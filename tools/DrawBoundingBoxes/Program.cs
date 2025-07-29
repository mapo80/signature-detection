using SkiaSharp;
using SignatureDetectionSdk;
using System.Linq;

string Root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../.."));
string imagesDir = Path.Combine(Root, "dataset", "images");
string labelsDir = Path.Combine(Root, "dataset", "labels");
string outputDetr = Path.Combine(Root, "samples", "detr");
string outputYolo = Path.Combine(Root, "samples", "yolov8s");
Directory.CreateDirectory(outputDetr);
Directory.CreateDirectory(outputYolo);
string onnxPath = Path.Combine(Root, "conditional_detr_signature.onnx");
string yoloOnnxPath = Path.Combine(Root, "yolov8s.onnx");
EnsureModel();

using var detector = new SignatureDetector(onnxPath);
using var yolo = File.Exists(yoloOnnxPath) && new FileInfo(yoloOnnxPath).Length > 1000000 ? new YoloV8Detector(yoloOnnxPath) : null;

string[] files = {
    "00101001_png_jpg.rf.27db3f0cbf1a1ef078dcca2fdc2874af.jpg",
    "00101027_png_jpg.rf.a92770147b74d58b15829954bbba6ac6.jpg",
    "00101029_png_jpg.rf.14639faea024ffc684cd71be406650dc.jpg",
    "00104001_png_jpg.rf.bfafcce0144b089dc34bc63f05c4ea12.jpg",
    "00104027_png_jpg.rf.a0812b28f188bed93538a071edc42b73.jpg",
    "001_15_PNG_jpg.rf.7ae3c04130de9c0e178fa2c1feb8eca9.jpg",
    "00205002_png_jpg.rf.c64a564d90ed620839808566c8ae60bc.jpg",
    "00205002_png_jpg.rf.edc16c394577e472cd95c93f73a616e4.jpg",
    "002_02_PNG_jpg.rf.036f32c4fafd37313d3efbf30e330a90.jpg",
    "002_11_PNG_jpg.rf.74c78f2735867cd2f42cf4550d9d7993.jpg",
    "002_15_PNG_jpg.rf.505a2e55fcdd82ca86042fe97b59d1b7.jpg",
    "02205002_png_jpg.rf.c491e313d0f62c95e2990f664fe44c8b.jpg",
    "02302023_png_jpg.rf.7b59991fc80b082bb1925a5071c22464.jpg",
    "02302070_png_jpg.rf.5db163a7de9ae621c56c1a86c3de2d84.jpg",
    "02305070_png_jpg.rf.d7ecd0ef0984bbe479271ab32d0888af.jpg",
    "02403024_png_jpg.rf.234c51b41d237cc3246c71e4fae0e0e0.jpg",
    "02601026_png_jpg.rf.2e55a766ff4dc7a77260ab10c910bca5.jpg",
    "02701027_png_jpg.rf.cbb3219446cb316a4c42533c35249aef.jpg",
    "02702027_png_jpg.rf.819a7dc18c7f3c8ce22710a3ed5abc08.jpg",
    "02703027_png_jpg.rf.b76da2e8d9524be6951e25848b1add1a.jpg"
};

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
