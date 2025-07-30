using SignatureDetectionSdk;
using SkiaSharp;
using System.Globalization;

record struct Detection(float X1, float Y1, float X2, float Y2, float Score, int Image);
record struct GroundTruth(float X1, float Y1, float X2, float Y2, int Image);
record struct Metrics(float Precision, float Recall, float F1, float mAP);

class Program
{
    static string Root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../"));
    static string OnnxPath = Path.Combine(Root, "conditional_detr_signature.onnx");
    static string YoloPath = Path.Combine(Root, "yolov8s.onnx");

    static void EnsureModel()
    {
        if (File.Exists(OnnxPath)) return;
        var part1 = Path.Combine(Root, "conditional_detr_signature.onnx_parte_1");
        var part2 = Path.Combine(Root, "conditional_detr_signature.onnx_parte_2");
        using var output = File.Create(OnnxPath);
        foreach (var p in new[] { part1, part2 })
        using (var fs = File.OpenRead(p))
            fs.CopyTo(output);
    }

    static void Main(string[] args)
    {
        string dataset = args.Length > 0 ? args[0] : "dataset1";
        EnsureModel();
        string imagesDir = Path.Combine(Root, "dataset", dataset, "images");
        string labelsDir = Path.Combine(Root, "dataset", dataset, "labels");
        var images = Directory.GetFiles(imagesDir, "*.jpg").OrderBy(f => f).ToArray();
        if (int.TryParse(Environment.GetEnvironmentVariable("MAX_IMAGES"), out int max) && max > 0 && max < images.Length)
            images = images.Take(max).ToArray();

        var detr = EvaluateDetr(images, labelsDir);
        var yolo = EvaluateYolo(images, labelsDir);

        Console.WriteLine($"DETR    Precision: {detr.Precision:F3} Recall: {detr.Recall:F3} F1: {detr.F1:F3} mAP: {detr.mAP:F3}");
        Console.WriteLine($"YOLOv8 Precision: {yolo.Precision:F3} Recall: {yolo.Recall:F3} F1: {yolo.F1:F3} mAP: {yolo.mAP:F3}");
    }

    static Metrics EvaluateDetr(string[] images, string labelsDir)
    {
        using var det = new SignatureDetector(OnnxPath);
        return Evaluate(img => det.Predict(img), images, labelsDir, convertFromDetr: true);
    }

    static Metrics EvaluateYolo(string[] images, string labelsDir)
    {
        if (!File.Exists(YoloPath))
            return new Metrics();
        using var det = new YoloV8Detector(YoloPath);
        return Evaluate(img => det.Predict(img), images, labelsDir, convertFromDetr: false);
    }

    static Metrics Evaluate(Func<string, float[][]> predict, string[] images, string labelsDir, bool convertFromDetr)
    {
        var preds = new List<Detection>();
        var gts = new List<GroundTruth>();
        int idx = 0;
        foreach (var img in images)
        {
            using var bitmap = SKBitmap.Decode(img);
            int w = bitmap.Width;
            int h = bitmap.Height;

            var labelPath = Path.Combine(labelsDir, Path.GetFileNameWithoutExtension(img) + ".txt");
            if (File.Exists(labelPath))
            {
                foreach (var line in File.ReadLines(labelPath))
                {
                    var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 5)
                    {
                        float cx = float.Parse(parts[1], CultureInfo.InvariantCulture);
                        float cy = float.Parse(parts[2], CultureInfo.InvariantCulture);
                        float bw = float.Parse(parts[3], CultureInfo.InvariantCulture);
                        float bh = float.Parse(parts[4], CultureInfo.InvariantCulture);
                        float x1 = (cx - bw / 2f) * w;
                        float y1 = (cy - bh / 2f) * h;
                        float x2 = (cx + bw / 2f) * w;
                        float y2 = (cy + bh / 2f) * h;
                        gts.Add(new GroundTruth(x1, y1, x2, y2, idx));
                    }
                }
            }

            var dets = predict(img);
            foreach (var d in dets)
            {
                float x1 = d[0];
                float y1 = d[1];
                float x2 = d[2];
                float y2 = d[3];
                float score = d[4];
                if (convertFromDetr)
                {
                    x1 = x1 / 640f * w;
                    y1 = y1 / 640f * h;
                    x2 = x2 / 640f * w;
                    y2 = y2 / 640f * h;
                }
                preds.Add(new Detection(x1, y1, x2, y2, score, idx));
            }

            idx++;
        }

        return ComputeMetrics(preds, gts);
    }

    static Metrics ComputeMetrics(List<Detection> preds, List<GroundTruth> gts, float iouThr = 0.5f)
    {
        var sorted = preds.OrderByDescending(p => p.Score).ToArray();
        var used = new bool[gts.Count];
        float[] tp = new float[sorted.Length];
        float[] fp = new float[sorted.Length];

        for (int i = 0; i < sorted.Length; i++)
        {
            var p = sorted[i];
            float best = iouThr;
            int match = -1;
            for (int j = 0; j < gts.Count; j++)
            {
                if (used[j] || gts[j].Image != p.Image) continue;
                float iou = IoU(p, gts[j]);
                if (iou >= best)
                {
                    best = iou;
                    match = j;
                }
            }
            if (match != -1)
            {
                tp[i] = 1;
                used[match] = true;
            }
            else fp[i] = 1;
        }

        float[] cumTP = new float[sorted.Length];
        float[] cumFP = new float[sorted.Length];
        for (int i = 0; i < sorted.Length; i++)
        {
            cumTP[i] = (i > 0 ? cumTP[i - 1] : 0) + tp[i];
            cumFP[i] = (i > 0 ? cumFP[i - 1] : 0) + fp[i];
        }

        float[] rec = new float[sorted.Length];
        float[] prec = new float[sorted.Length];
        for (int i = 0; i < sorted.Length; i++)
        {
            rec[i] = gts.Count > 0 ? cumTP[i] / gts.Count : 0;
            prec[i] = (cumTP[i] + cumFP[i]) > 0 ? cumTP[i] / (cumTP[i] + cumFP[i]) : 0;
        }

        var mRec = new List<float> { 0 };
        mRec.AddRange(rec);
        mRec.Add(1);
        var mPre = new List<float> { prec.Length > 0 ? prec[0] : 0 };
        mPre.AddRange(prec);
        mPre.Add(0);
        for (int i = mPre.Count - 2; i >= 0; i--)
            mPre[i] = MathF.Max(mPre[i], mPre[i + 1]);

        float ap = 0;
        for (int i = 0; i < mRec.Count - 1; i++)
            ap += (mRec[i + 1] - mRec[i]) * mPre[i + 1];

        float P = sorted.Length > 0 ? cumTP[^1] / sorted.Length : 0;
        float R = gts.Count > 0 ? cumTP[^1] / gts.Count : 0;
        float F1 = (P + R) > 0 ? 2 * P * R / (P + R) : 0;
        return new Metrics(P, R, F1, ap);
    }

    static float IoU(Detection d, GroundTruth g)
    {
        float xx1 = MathF.Max(d.X1, g.X1);
        float yy1 = MathF.Max(d.Y1, g.Y1);
        float xx2 = MathF.Min(d.X2, g.X2);
        float yy2 = MathF.Min(d.Y2, g.Y2);
        float inter = MathF.Max(0, xx2 - xx1) * MathF.Max(0, yy2 - yy1);
        if (inter <= 0) return 0f;
        float areaA = MathF.Max(0, d.X2 - d.X1) * MathF.Max(0, d.Y2 - d.Y1);
        float areaB = MathF.Max(0, g.X2 - g.X1) * MathF.Max(0, g.Y2 - g.Y1);
        float union = areaA + areaB - inter;
        return union > 0 ? inter / union : 0f;
    }
}
