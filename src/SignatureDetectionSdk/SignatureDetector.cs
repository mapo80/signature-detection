using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace SignatureDetectionSdk;

public class SignatureDetector : IDisposable
{
    private readonly InferenceSession _session;
    private readonly float[] _mean = {0.485f, 0.456f, 0.406f};
    private readonly float[] _std = {0.229f, 0.224f, 0.225f};
    public readonly record struct RobustParams(
        float AreaMin, float AreaMax,
        float ArMin, float ArMax,
        float Sigma, float DistScale,
        float Alpha, int NMin,
        float ScorePercentile);

    public static readonly RobustParams DetrParams = new(1200f, 150000f, 0.6f, 6.5f, 0.6f, 180f, 0.6f, 2, 0.50f);
    public static readonly RobustParams EnsembleParams = new(800f, 200000f, 0.5f, 8f, 0.4f, 120f, 0.5f, 1, 0.50f);
    // Dedicated geometry limits for dataset2 computed from the ground-truth 5th
    // and 95th percentiles
    public static readonly RobustParams Dataset2Params = new(
        1731f, 12928f,
        1.836f, 7.838f,
        0.4f, 80f,
        0.7f, 1,
        0.50f);

    public int InputSize { get; }

    public SignatureDetector(string modelPath, int inputSize = 640, SessionOptions? options = null)
    {
        InputSize = inputSize;
        _session = options is null ? new InferenceSession(modelPath) : new InferenceSession(modelPath, options);
    }

    public void Dispose()
    {
        _session.Dispose();
    }

    public float[][] Predict(string imagePath, out int robustCount,
        float scoreThreshold = 0.1f, RobustParams? parameters = null)
    {
        using var image = SKBitmap.Decode(imagePath);
        using var resized = image.Resize(new SKImageInfo(InputSize, InputSize), SKFilterQuality.High);
        var tensor = new DenseTensor<float>(new[] {1, 3, InputSize, InputSize});

        for (int y = 0; y < InputSize; y++)
        {
            for (int x = 0; x < InputSize; x++)
            {
                var color = resized.GetPixel(x, y);
                float r = color.Red / 255f;
                float g = color.Green / 255f;
                float b = color.Blue / 255f;
                tensor[0, 0, y, x] = (r - _mean[0]) / _std[0];
                tensor[0, 1, y, x] = (g - _mean[1]) / _std[1];
                tensor[0, 2, y, x] = (b - _mean[2]) / _std[2];
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", tensor)
        };
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
        var logits = results.First(r => r.Name == "logits").AsTensor<float>();
        var boxes = results.First(r => r.Name == "boxes").AsTensor<float>();

        var p = parameters ?? DetrParams;

        var scores = PostProcessing.FilterByScore(logits, scoreThreshold);
        var dets = PostProcessing.ToPixelBoxes(boxes, resized.Width, resized.Height, scores);
        dets = PostProcessing.FilterByGeometry(dets, p.AreaMin, p.AreaMax, p.ArMin, p.ArMax);
        var robust = PostProcessing.SoftNmsDistance(dets, p.Sigma, p.DistScale);
        robustCount = robust.Count;
        float dynamicThresh = 0.3f;
        if (robust.Count > 0)
        {
            var scoresList = robust.Select(b => b[4]).ToList();
            float perc = PostProcessing.Percentile(scoresList, p.ScorePercentile * 100f);
            dynamicThresh = p.Alpha * perc;
        }
        var filtered = robust.Where(b => b[4] >= dynamicThresh).ToList();
        var nms = PostProcessing.Nms(dets, 0.5f);
        List<float[]> final = filtered.Count < p.NMin ? nms : filtered;
        return final.ToArray();
    }

    public float[][] Predict(string imagePath, float scoreThreshold = 0.1f,
        RobustParams? parameters = null)
        => Predict(imagePath, out _, scoreThreshold, parameters);

    public float[][] PredictSimple(string imagePath,
        out bool usedFallback,
        float confThreshold = 0.6f,
        float fallbackThreshold = 0.3f,
        RobustParams? geometry = null)
    {
        using var image = SKBitmap.Decode(imagePath);
        using var resized = image.Resize(new SKImageInfo(InputSize, InputSize), SKFilterQuality.High);
        var tensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });

        for (int y = 0; y < InputSize; y++)
        {
            for (int x = 0; x < InputSize; x++)
            {
                var color = resized.GetPixel(x, y);
                float r = color.Red / 255f;
                float g = color.Green / 255f;
                float b = color.Blue / 255f;
                tensor[0, 0, y, x] = (r - _mean[0]) / _std[0];
                tensor[0, 1, y, x] = (g - _mean[1]) / _std[1];
                tensor[0, 2, y, x] = (b - _mean[2]) / _std[2];
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", tensor)
        };
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
        var logits = results.First(r => r.Name == "logits").AsTensor<float>();
        var boxes = results.First(r => r.Name == "boxes").AsTensor<float>();

        var scores = PostProcessing.FilterByScore(logits, confThreshold);
        var dets = PostProcessing.ToPixelBoxes(boxes, resized.Width, resized.Height, scores);
        if (geometry.HasValue)
            dets = PostProcessing.FilterByGeometry(dets,
                geometry.Value.AreaMin,
                geometry.Value.AreaMax,
                geometry.Value.ArMin,
                geometry.Value.ArMax);

        var nms = PostProcessing.Nms(dets, 0.5f);
        usedFallback = false;
        if (nms.Count == 0 && fallbackThreshold < confThreshold)
        {
            usedFallback = true;
            scores = PostProcessing.FilterByScore(logits, fallbackThreshold);
            dets = PostProcessing.ToPixelBoxes(boxes, resized.Width, resized.Height, scores);
            if (geometry.HasValue)
                dets = PostProcessing.FilterByGeometry(dets,
                    geometry.Value.AreaMin,
                    geometry.Value.AreaMax,
                    geometry.Value.ArMin,
                    geometry.Value.ArMax);
            nms = PostProcessing.Nms(dets, 0.5f);
        }

        return nms.ToArray();
    }

    public float[][] PredictSimpleMetrics(string imagePath,
        out bool usedFallback,
        out int nBefore,
        out int nAfterThresh,
        out int nAfterNms,
        float confThreshold = 0.6f,
        float fallbackThreshold = 0.3f,
        RobustParams? geometry = null)
    {
        using var image = SKBitmap.Decode(imagePath);
        using var resized = image.Resize(new SKImageInfo(InputSize, InputSize), SKFilterQuality.High);
        var tensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });

        for (int y = 0; y < InputSize; y++)
        {
            for (int x = 0; x < InputSize; x++)
            {
                var color = resized.GetPixel(x, y);
                float r = color.Red / 255f;
                float g = color.Green / 255f;
                float b = color.Blue / 255f;
                tensor[0, 0, y, x] = (r - _mean[0]) / _std[0];
                tensor[0, 1, y, x] = (g - _mean[1]) / _std[1];
                tensor[0, 2, y, x] = (b - _mean[2]) / _std[2];
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("pixel_values", tensor)
        };
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
        var logits = results.First(r => r.Name == "logits").AsTensor<float>();
        var boxes = results.First(r => r.Name == "boxes").AsTensor<float>();

        nBefore = logits.Dimensions[1];

        var scores = PostProcessing.FilterByScore(logits, confThreshold);
        nAfterThresh = scores.Count;
        var dets = PostProcessing.ToPixelBoxes(boxes, resized.Width, resized.Height, scores);
        if (geometry.HasValue)
            dets = PostProcessing.FilterByGeometry(dets,
                geometry.Value.AreaMin,
                geometry.Value.AreaMax,
                geometry.Value.ArMin,
                geometry.Value.ArMax);

        var nms = PostProcessing.Nms(dets, 0.5f);
        nAfterNms = nms.Count;
        usedFallback = false;
        if (nms.Count == 0 && fallbackThreshold < confThreshold)
        {
            usedFallback = true;
            scores = PostProcessing.FilterByScore(logits, fallbackThreshold);
            dets = PostProcessing.ToPixelBoxes(boxes, resized.Width, resized.Height, scores);
            if (geometry.HasValue)
                dets = PostProcessing.FilterByGeometry(dets,
                    geometry.Value.AreaMin,
                    geometry.Value.AreaMax,
                    geometry.Value.ArMin,
                    geometry.Value.ArMax);
            nms = PostProcessing.Nms(dets, 0.5f);
        }

        return nms.ToArray();
    }

}
