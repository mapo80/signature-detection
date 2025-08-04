using SignatureDetectionSdk;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.Processing;

string root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../"));
string modelPath = Path.Combine(root, "conditional_detr_signature.onnx");

void EnsureModel()
{
    if (File.Exists(modelPath)) return;
    var part1 = Path.Combine(root, "conditional_detr_signature.onnx_parte_1");
    var part2 = Path.Combine(root, "conditional_detr_signature.onnx_parte_2");
    using var output = File.Create(modelPath);
    foreach (var p in new[]{part1, part2})
    using (var fs = File.OpenRead(p))
        fs.CopyTo(output);
}

string datasetPath = args.Length > 0 ? args[0] : "/dataset";
if (!Path.IsPathRooted(datasetPath))
    datasetPath = Path.Combine(root, datasetPath);
EnsureModel();
using var detector = new SignatureDetector(modelPath, 640);
foreach (var img in Directory.GetFiles(datasetPath))
{
    var preds = detector.Predict(img);
    Console.WriteLine($"{Path.GetFileName(img)}:");
    using var image = Image.Load(img);
    foreach (var p in preds)
    {
        Console.WriteLine($"  [{p[0]:F1},{p[1]:F1},{p[2]:F1},{p[3]:F1}] score={p[4]:F2}");
        var rect = new SixLabors.ImageSharp.Drawing.RectangularPolygon(p[0], p[1], p[2] - p[0], p[3] - p[1]);
        image.Mutate(ctx => ctx.Draw(Color.Red, 2, rect));
    }
    using var ms = new MemoryStream();
    var ext = Path.GetExtension(img).ToLowerInvariant();
    if (ext == ".jpg" || ext == ".jpeg")
        image.SaveAsJpeg(ms);
    else
        image.SaveAsPng(ms);
    File.WriteAllText(img + ".base64", Convert.ToBase64String(ms.ToArray()));
}
