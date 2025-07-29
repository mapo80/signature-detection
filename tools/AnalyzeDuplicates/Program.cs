using SignatureDetectionSdk;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;

string Root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../.."));
string OnnxPath = Path.Combine(Root, "conditional_detr_signature.onnx");
string dataset = args.Length > 0 ? args[0] : "dataset1";

void EnsureModel()
{
    if (File.Exists(OnnxPath)) return;
    var part1 = Path.Combine(Root, "conditional_detr_signature.onnx_parte_1");
    var part2 = Path.Combine(Root, "conditional_detr_signature.onnx_parte_2");
    using var output = File.Create(OnnxPath);
    foreach (var part in new[] { part1, part2 })
    using (var fs = File.OpenRead(part))
        fs.CopyTo(output);
}

EnsureModel();
var imagesDir = Path.Combine(Root, "dataset", dataset, "images");
var labelsDir = Path.Combine(Root, "dataset", dataset, "labels");
var images = Directory.GetFiles(imagesDir).OrderBy(f => f).ToArray();
using var detector = new SignatureDetector(OnnxPath);

Console.WriteLine("image,detections,maxIoU,minCenterDist");
foreach (var img in images)
{
    var preds = detector.Predict(img, 0.1f);
    if (preds.Length <= 1) continue;
    float maxIoU = 0f;
    float minDist = float.MaxValue;
    for (int i = 0; i < preds.Length; i++)
    {
        for (int j = i+1; j < preds.Length; j++)
        {
            var a = preds[i];
            var b = preds[j];
            float iou = IoU(a,b);
            if (iou > maxIoU) maxIoU = iou;
            float dist = CenterDistance(a,b);
            if (dist < minDist) minDist = dist;
        }
    }
    Console.WriteLine($"{Path.GetFileName(img)},{preds.Length},{maxIoU:F2},{minDist:F1}");
}

static float IoU(float[] a, float[] b)
{
    float xx1 = MathF.Max(a[0], b[0]);
    float yy1 = MathF.Max(a[1], b[1]);
    float xx2 = MathF.Min(a[2], b[2]);
    float yy2 = MathF.Min(a[3], b[3]);
    float inter = MathF.Max(0, xx2 - xx1) * MathF.Max(0, yy2 - yy1);
    float areaA = MathF.Max(0, a[2]-a[0]) * MathF.Max(0, a[3]-a[1]);
    float areaB = MathF.Max(0, b[2]-b[0]) * MathF.Max(0, b[3]-b[1]);
    float union = areaA + areaB - inter;
    return union <= 0 ? 0 : inter / union;
}

static float CenterDistance(float[] a, float[] b)
{
    float cxA = (a[0] + a[2]) / 2f;
    float cyA = (a[1] + a[3]) / 2f;
    float cxB = (b[0] + b[2]) / 2f;
    float cyB = (b[1] + b[3]) / 2f;
    float dx = cxA - cxB;
    float dy = cyA - cyB;
    return MathF.Sqrt(dx*dx + dy*dy);
}
