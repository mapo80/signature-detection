using SignatureDetectionSdk;
using System.Linq;
using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;

string Root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../"));
string OnnxPath = Path.Combine(Root, "conditional_detr_signature.onnx");

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

float IoU(float x1, float y1, float x2, float y2, float x1b, float y1b, float x2b, float y2b)
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

string dataset = "dataset1";
bool useYolo = false;
bool optimizeSession = false;
bool ensemble = false;
int max = int.MaxValue;
foreach (var a in args)
{
    if (a == "--yolo") useYolo = true;
    else if (a == "--ensemble") { useYolo = true; ensemble = true; }
    else if (a == "--ort-opt") optimizeSession = true;
    else if (a.StartsWith("--max=")) max = int.Parse(a.Substring(6));
    else dataset = a;
}

var imagesDir = Path.Combine(Root, "dataset", dataset, "images");
var labelsDir = Path.Combine(Root, "dataset", dataset, "labels");
var images = Directory.GetFiles(imagesDir).OrderBy(f => f).Take(max).ToArray();
var rows = new List<string>();
if (!useYolo) EnsureModel();
string yoloPath = Path.Combine(Root, "yolov8s.onnx");
SessionOptions? opts = null;
if (optimizeSession)
{
    opts = new SessionOptions();
    opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
}
IDisposable detectorObj;
if (ensemble)
{
    var detrParams = dataset == "dataset2" ? SignatureDetector.Dataset2Params : SignatureDetector.DetrParams;
    var ensParams = dataset == "dataset2" ? SignatureDetector.Dataset2Params : SignatureDetector.EnsembleParams;
    int tLow = dataset == "dataset2" ? 18 : 5;
    float wbfIou = dataset == "dataset2" ? 0.50f : 0.55f;
    float wbfScore = dataset == "dataset2" ? 0.35f : 0.45f;
    detectorObj = new EnsembleDetector(OnnxPath, yoloPath, true, tLow,
        wbfIou, wbfScore, detrParams, ensParams);
}
else if (useYolo)
    detectorObj = new YoloV8Detector(yoloPath);
else
{
    detectorObj = new SignatureDetector(OnnxPath, 640, opts);
}
dynamic detector = detectorObj;
double totalMs = 0;
foreach (var img in images)
{
    var sw = Stopwatch.StartNew();
    float[][] preds;
    if (detectorObj is YoloV8Detector y)
    {
        var p = dataset == "dataset2" ? SignatureDetector.Dataset2Params : SignatureDetector.EnsembleParams;
        preds = y.Predict(img, 0.25f, p);
    }
    else if (detectorObj is SignatureDetector d)
    {
        var p = dataset == "dataset2" ? SignatureDetector.Dataset2Params : SignatureDetector.DetrParams;
        preds = d.Predict(img, out _, 0.1f, p);
    }
    else if (detectorObj is EnsembleDetector ensDet)
        preds = ensDet.Predict(img);
    else
        preds = detector.Predict(img);
    sw.Stop();
    totalMs += sw.Elapsed.TotalMilliseconds;
    var labelPath = Path.Combine(labelsDir, Path.GetFileNameWithoutExtension(img) + ".txt");
    var labelText = File.ReadAllText(labelPath).Trim();
    int numLabels = 0;
    double diff = 100.0;
    if (!string.IsNullOrWhiteSpace(labelText))
    {
    var parts = labelText.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
        numLabels = parts.Length / 5;
        float cx = float.Parse(parts[1]);
        float cy = float.Parse(parts[2]);
        float w = float.Parse(parts[3]);
        float h = float.Parse(parts[4]);
        float lx1 = (cx - w/2) * 640;
        float ly1 = (cy - h/2) * 640;
        float lx2 = (cx + w/2) * 640;
        float ly2 = (cy + h/2) * 640;
        float best = 0f;
        foreach (var p in preds)
        {
            float iou = IoU(p[0], p[1], p[2], p[3], lx1, ly1, lx2, ly2);
            if (iou > best) best = iou;
        }
        diff = (1 - best) * 100.0;
    }
    rows.Add($"{Path.GetFileName(img)},{numLabels},{preds.Length},{diff:F2},{sw.Elapsed.TotalMilliseconds:F0}");
}
string suffix = ensemble ? "_ensemble" : useYolo ? "_yolo" : "";
string outFile = dataset == "dataset1" ? $"dataset_report{suffix}.csv" : $"dataset_report_{dataset}{suffix}.csv";
File.WriteAllLines(Path.Combine(Root, outFile), rows);
Console.WriteLine($"Average inference ms: {totalMs / images.Length:F1}");
if (ensemble && detectorObj is EnsembleDetector ed)
{
    Console.WriteLine($"Ensemble triggered on {ed.UsedCount}/{ed.TotalCount} images ({ed.UsedCount * 100.0 / ed.TotalCount:F1}%)");
    int totFuse = ed.FusionAccepted + ed.FusionRejected;
    if (totFuse > 0)
        Console.WriteLine($"Fusion accepted {ed.FusionAccepted}/{totFuse} ({ed.FusionAccepted * 100.0 / totFuse:F1}%)");
}
