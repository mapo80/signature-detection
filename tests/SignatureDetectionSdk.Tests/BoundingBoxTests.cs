using SignatureDetectionSdk;
using Xunit;

namespace SignatureDetectionSdk.Tests;

public class BoundingBoxTests
{
    private static string Root => Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../"));
    private static string OnnxPath => Path.Combine(Root, "conditional_detr_signature.onnx");

    private static void EnsureModel()
    {
        if (File.Exists(OnnxPath)) return;
        var part1 = Path.Combine(Root, "conditional_detr_signature.onnx_parte_1");
        var part2 = Path.Combine(Root, "conditional_detr_signature.onnx_parte_2");
        using var output = File.Create(OnnxPath);
        foreach (var part in new[] { part1, part2 })
        using (var fs = File.OpenRead(part))
            fs.CopyTo(output);
    }

    public static IEnumerable<object[]> SampleImages()
    {
        var dataset = Environment.GetEnvironmentVariable("DATASET_SUBDIR") ?? "dataset1";
        var imagesDir = Path.Combine(Root, "dataset", dataset, "images");
        return Directory.GetFiles(imagesDir).OrderBy(f => f)
            .Select(f => new object[] { f });
    }

    [Theory]
    [MemberData(nameof(SampleImages))]
    public void DetectionMatchesLabel(string imagePath)
    {
        EnsureModel();
        using var detector = new SignatureDetector(OnnxPath);
        var detections = detector.Predict(imagePath, 0.1f);
        var dataset = Environment.GetEnvironmentVariable("DATASET_SUBDIR") ?? "dataset1";
        var labelPath = Path.Combine(Root, "dataset", dataset, "labels",
            Path.GetFileNameWithoutExtension(imagePath) + ".txt");
        var labelLines = File.ReadAllLines(labelPath);
        if (labelLines.Length == 0) return; // no label for this image
        Assert.NotEmpty(detections);
        var bestIoU = BestIoU(detections, labelLines);
        Assert.True(bestIoU > 0.25, $"IoU too low: {bestIoU}");
    }

    private static float BestIoU(float[][] dets, IEnumerable<string> labels)
    {
        float best = 0f;
        foreach (var line in labels)
        {
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 5) continue;
            float cx = float.Parse(parts[1]);
            float cy = float.Parse(parts[2]);
            float w = float.Parse(parts[3]);
            float h = float.Parse(parts[4]);
            float lx1 = (cx - w / 2) * 640;
            float ly1 = (cy - h / 2) * 640;
            float lx2 = (cx + w / 2) * 640;
            float ly2 = (cy + h / 2) * 640;
            foreach (var d in dets)
            {
                float iou = IoU(d[0], d[1], d[2], d[3], lx1, ly1, lx2, ly2);
                if (iou > best) best = iou;
            }
        }
        return best;
    }

    private static float IoU(float x1, float y1, float x2, float y2,
                             float x1b, float y1b, float x2b, float y2b)
    {
        float xx1 = MathF.Max(x1, x1b);
        float yy1 = MathF.Max(y1, y1b);
        float xx2 = MathF.Min(x2, x2b);
        float yy2 = MathF.Min(y2, y2b);
        float inter = MathF.Max(0, xx2 - xx1) * MathF.Max(0, yy2 - yy1);
        float area = MathF.Max(0, x2 - x1) * MathF.Max(0, y2 - y1);
        float areab = MathF.Max(0, x2b - x1b) * MathF.Max(0, y2b - y1b);
        float union = area + areab - inter;
        return union <= 0 ? 0 : inter / union;
    }
}
