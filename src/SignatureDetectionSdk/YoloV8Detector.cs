using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace SignatureDetectionSdk;

public class YoloV8Detector : IDisposable
{
    private readonly InferenceSession _session;
    public int InputSize { get; }

    public YoloV8Detector(string modelPath, int inputSize = 640)
    {
        InputSize = inputSize;
        _session = new InferenceSession(modelPath);
    }

    public void Dispose()
    {
        _session.Dispose();
    }

    public float[][] Predict(string imagePath, float scoreThreshold = 0.25f)
    {
        using var image = SKBitmap.Decode(imagePath);
        using var resized = image.Resize(new SKImageInfo(InputSize, InputSize), SKFilterQuality.High);
        var tensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });

        for (int y = 0; y < InputSize; y++)
        {
            for (int x = 0; x < InputSize; x++)
            {
                var color = resized.GetPixel(x, y);
                tensor[0, 0, y, x] = color.Red / 255f;
                tensor[0, 1, y, x] = color.Green / 255f;
                tensor[0, 2, y, x] = color.Blue / 255f;
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("images", tensor)
        };
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);
        var output = results.First().AsTensor<float>();

        int attrs = output.Dimensions[1];
        int boxes = output.Dimensions[2];
        bool transpose = false;

        if (attrs != 6 && boxes == 6)
        {
            // some exports use (1, boxes, 6)
            transpose = true;
            boxes = output.Dimensions[1];
            attrs = 6;
        }

        var dets = new List<float[]>();
        for (int i = 0; i < boxes; i++)
        {
            float cx, cy, w, h, obj, cls;
            if (transpose)
            {
                cx = output[0, i, 0];
                cy = output[0, i, 1];
                w  = output[0, i, 2];
                h  = output[0, i, 3];
                obj = output[0, i, 4];
                cls = attrs > 5 ? output[0, i, 5] : 1f;
            }
            else
            {
                cx = output[0, 0, i];
                cy = output[0, 1, i];
                w  = output[0, 2, i];
                h  = output[0, 3, i];
                obj = output[0, 4, i];
                cls = attrs > 5 ? output[0, 5, i] : 1f;
            }

            float score = obj * cls;
            if (score < scoreThreshold) continue;

            float x1 = (cx - w / 2f) * image.Width / InputSize;
            float y1 = (cy - h / 2f) * image.Height / InputSize;
            float x2 = (cx + w / 2f) * image.Width / InputSize;
            float y2 = (cy + h / 2f) * image.Height / InputSize;
            dets.Add(new[] { x1, y1, x2, y2, score });
        }

        return dets.ToArray();
    }
}
