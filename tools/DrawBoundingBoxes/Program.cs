using SkiaSharp;
using SignatureDetectionSdk;
using System.Linq;
using System.IO;

string Root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../.."));
string dataset = Environment.GetEnvironmentVariable("DATASET_SUBDIR") ?? "dataset1";
string imagesDir = Path.Combine(Root, "dataset", dataset, "images");
string labelsDir = Path.Combine(Root, "dataset", dataset, "labels");
string outputDetr = Path.Combine(Root, "samples", "detr");
string outputYolo = Path.Combine(Root, "samples", "yolov8s");
Directory.CreateDirectory(outputDetr);
Directory.CreateDirectory(outputYolo);
string onnxPath = Path.Combine(Root, "conditional_detr_signature.onnx");
string yoloOnnxPath = Path.Combine(Root, "yolov8s.onnx");
EnsureModel();

using var detector = new SignatureDetector(onnxPath);
using var yolo = File.Exists(yoloOnnxPath) && new FileInfo(yoloOnnxPath).Length > 1000000 ? new YoloV8Detector(yoloOnnxPath) : null;

var files = Directory.GetFiles(imagesDir, "*.jpg")
    .OrderBy(f => f)
    .Take(20)
    .Select(Path.GetFileName)
    .ToArray();

foreach (var file in files)
{
    var imagePath = Path.Combine(imagesDir, file);
    var labelPath = Path.Combine(labelsDir, Path.GetFileNameWithoutExtension(file) + ".txt");
    if (!File.Exists(imagePath) || !File.Exists(labelPath))
    {
        Console.WriteLine($"Skipping {file} (missing data)");
        continue;
    }

    var parts = File.ReadAllText(labelPath).Split(' ', StringSplitOptions.RemoveEmptyEntries);
    float[]? gt = null;
    using (var tmp = SKBitmap.Decode(imagePath))
    {
        if (parts.Length >= 5)
        {
            float cx = float.Parse(parts[1]);
            float cy = float.Parse(parts[2]);
            float w = float.Parse(parts[3]);
            float h = float.Parse(parts[4]);
            float x1 = (cx - w / 2f) * tmp.Width;
            float y1 = (cy - h / 2f) * tmp.Height;
            float x2 = (cx + w / 2f) * tmp.Width;
            float y2 = (cy + h / 2f) * tmp.Height;
            gt = new[] { x1, y1, x2, y2 };
        }
    }

    using (var detBitmap = SKBitmap.Decode(imagePath))
    using (var canvas = new SKCanvas(detBitmap))
    using (var gtPaint = new SKPaint { Color = SKColors.Red, Style = SKPaintStyle.Stroke, StrokeWidth = 3 })
    using (var detPaint = new SKPaint { Color = SKColors.Lime, Style = SKPaintStyle.Stroke, StrokeWidth = 3 })
    using (var textPaint = new SKPaint { Color = SKColors.Yellow, TextSize = 24, IsAntialias = true })
    {
        if (gt != null)
        {
            canvas.DrawRect(SKRect.Create(gt[0], gt[1], gt[2] - gt[0], gt[3] - gt[1]), gtPaint);
            canvas.DrawText("GT", gt[0], Math.Max(0, gt[1] - 5), textPaint);
        }

        var detections = detector.Predict(imagePath);
        if (detections.Length > 0)
        {
            var best = detections.OrderByDescending(d => d[4]).First();
            canvas.DrawRect(SKRect.Create(best[0], best[1], best[2] - best[0], best[3] - best[1]), detPaint);
            canvas.DrawText($"DETR {best[4]:0.00}", best[0], Math.Max(0, best[1] - 25), textPaint);
        }

        var outPath = Path.Combine(outputDetr, file);
        using var image = SKImage.FromBitmap(detBitmap);
        using var data = image.Encode(SKEncodedImageFormat.Jpeg, 90);
        using var fs = File.OpenWrite(outPath);
        data.SaveTo(fs);
        Console.WriteLine($"Saved {outPath}");
    }

    if (yolo != null)
    {
        using var yBitmap = SKBitmap.Decode(imagePath);
        using var canvas = new SKCanvas(yBitmap);
        using var gtPaint = new SKPaint { Color = SKColors.Red, Style = SKPaintStyle.Stroke, StrokeWidth = 3 };
        using var yPaint = new SKPaint { Color = SKColors.Cyan, Style = SKPaintStyle.Stroke, StrokeWidth = 3 };
        using var textPaint = new SKPaint { Color = SKColors.Yellow, TextSize = 24, IsAntialias = true };

        if (gt != null)
        {
            canvas.DrawRect(SKRect.Create(gt[0], gt[1], gt[2] - gt[0], gt[3] - gt[1]), gtPaint);
            canvas.DrawText("GT", gt[0], Math.Max(0, gt[1] - 5), textPaint);
        }

        var yoloDet = yolo.Predict(imagePath);
        if (yoloDet.Length > 0)
        {
            var best = yoloDet.OrderByDescending(d => d[4]).First();
            canvas.DrawRect(SKRect.Create(best[0], best[1], best[2] - best[0], best[3] - best[1]), yPaint);
            canvas.DrawText($"YOLO {best[4]:0.00}", best[0], Math.Max(0, best[1] - 25), textPaint);
        }

        var outPath = Path.Combine(outputYolo, file);
        using var image = SKImage.FromBitmap(yBitmap);
        using var data = image.Encode(SKEncodedImageFormat.Jpeg, 90);
        using var fs = File.OpenWrite(outPath);
        data.SaveTo(fs);
        Console.WriteLine($"Saved {outPath}");
    }
}

void EnsureModel()
{
    if (File.Exists(onnxPath)) return;
    var part1 = Path.Combine(Root, "conditional_detr_signature.onnx_parte_1");
    var part2 = Path.Combine(Root, "conditional_detr_signature.onnx_parte_2");
    using var output = File.Create(onnxPath);
    foreach (var part in new[] { part1, part2 })
    using (var fs = File.OpenRead(part))
        fs.CopyTo(output);
}
