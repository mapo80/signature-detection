using SkiaSharp;
using SignatureDetectionSdk;

string Root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../"));
string modelPath = Path.Combine(Root, "yolov8s.onnx");
string dataset = Environment.GetEnvironmentVariable("DATASET_SUBDIR") ?? "dataset1";
string imagesDir = Path.Combine(Root, "dataset", dataset, "images");
string labelsDir = Path.Combine(Root, "dataset", dataset, "labels");
string outputDir = Path.Combine(Root, "samples", "yolov8s");

if (!File.Exists(modelPath))
{
    Console.WriteLine($"Model not found: {modelPath}");
    return;
}
Directory.CreateDirectory(outputDir);

using var detector = new YoloV8Detector(modelPath);
var images = Directory.GetFiles(imagesDir, "*.jpg").OrderBy(f => f);
foreach (var img in images)
{
    using var bitmap = SKBitmap.Decode(img);
    using var canvas = new SKCanvas(bitmap);
    using var gtPaint = new SKPaint { Color = SKColors.Red, Style = SKPaintStyle.Stroke, StrokeWidth = 3 };
    using var detPaint = new SKPaint { Color = SKColors.Lime, Style = SKPaintStyle.Stroke, StrokeWidth = 3 };
    using var textPaint = new SKPaint { Color = SKColors.Yellow, TextSize = 24, IsAntialias = true };

    var labelFile = Path.Combine(labelsDir, Path.GetFileNameWithoutExtension(img) + ".txt");
    if (File.Exists(labelFile))
    {
        foreach (var line in File.ReadLines(labelFile))
        {
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length >= 5)
            {
                float cx = float.Parse(parts[1]);
                float cy = float.Parse(parts[2]);
                float w = float.Parse(parts[3]);
                float h = float.Parse(parts[4]);
                float x1 = (cx - w / 2f) * bitmap.Width;
                float y1 = (cy - h / 2f) * bitmap.Height;
                float x2 = (cx + w / 2f) * bitmap.Width;
                float y2 = (cy + h / 2f) * bitmap.Height;
                canvas.DrawRect(SKRect.Create(x1, y1, x2 - x1, y2 - y1), gtPaint);
                canvas.DrawText("GT", x1, Math.Max(0, y1 - 5), textPaint);
            }
        }
    }

    var dets = detector.Predict(img);
    foreach (var det in dets)
    {
        float x1 = det[0];
        float y1 = det[1];
        float x2 = det[2];
        float y2 = det[3];
        float score = det[4];
        canvas.DrawRect(SKRect.Create(x1, y1, x2 - x1, y2 - y1), detPaint);
        canvas.DrawText($"{score:0.00}", x1, Math.Max(0, y1 - 25), textPaint);
    }

    var outPath = Path.Combine(outputDir, Path.GetFileName(img));
    using var image = SKImage.FromBitmap(bitmap);
    using var data = image.Encode(SKEncodedImageFormat.Jpeg, 90);
    using var fs = File.OpenWrite(outPath);
    data.SaveTo(fs);
    Console.WriteLine($"Saved {outPath}");
}
