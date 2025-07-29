using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace SignatureDetectionSdk;

public class SignatureDetector : IDisposable
{
    private readonly InferenceSession _session;
    private readonly float[] _mean = {0.485f, 0.456f, 0.406f};
    private readonly float[] _std = {0.229f, 0.224f, 0.225f};
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

    public float[][] Predict(string imagePath, float scoreThreshold = 0.1f)
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

        var scores = PostProcessing.FilterByScore(logits, scoreThreshold);
        var dets = PostProcessing.ToPixelBoxes(boxes, resized.Width, resized.Height, scores);
        dets = PostProcessing.FilterByGeometry(dets, 800f, 400000f, 0.5f, 6f);
        var robust = PostProcessing.SoftNmsDistance(dets, 0.5f, 150f);
        float dynamicThresh = 0.3f;
        if (robust.Count > 0)
        {
            var ordered = robust.Select(b => b[4]).OrderBy(v => v).ToList();
            float median = ordered[ordered.Count / 2];
            dynamicThresh = 0.6f * median;
        }
        var filtered = robust.Where(b => b[4] >= dynamicThresh).ToList();
        var nms = PostProcessing.Nms(dets, 0.5f);
        List<float[]> final = filtered.Count < 2 ? nms : filtered;
        return final.ToArray();
    }

}
