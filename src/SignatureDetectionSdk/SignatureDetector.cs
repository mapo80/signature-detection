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
        var logits = results.First(r => r.Name == "logits").AsEnumerable<float>().ToArray();
        var boxes = results.First(r => r.Name == "boxes").AsEnumerable<float>().ToArray();
        int numPreds = logits.Length; // single class with sigmoid activation
        var detections = new List<float[]>();
        for (int i = 0; i < numPreds; i++)
        {
            float score = Sigmoid(logits[i]);
            if (score <= scoreThreshold) continue;
            float cx = boxes[i*4];
            float cy = boxes[i*4 + 1];
            float w = boxes[i*4 + 2];
            float h = boxes[i*4 + 3];
            float x1 = (cx - w/2) * resized.Width;
            float y1 = (cy - h/2) * resized.Height;
            float x2 = (cx + w/2) * resized.Width;
            float y2 = (cy + h/2) * resized.Height;
            detections.Add(new[]{x1,y1,x2,y2,score});
        }
        return detections.ToArray();
    }

    private static float Sigmoid(float value)
    {
        return 1f / (1f + MathF.Exp(-value));
    }
}
