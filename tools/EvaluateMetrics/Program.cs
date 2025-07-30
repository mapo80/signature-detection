using SignatureDetectionSdk;
using SkiaSharp;
using System.Globalization;
using System.Text.Json;

record struct Detection(float X1, float Y1, float X2, float Y2, float Score, int Image);
record struct GroundTruth(float X1, float Y1, float X2, float Y2, int Image);

record struct ImageDetail(int Image, double Ms, int Pred, int GroundTruth, int TP, int FP, int FN, float[] IoUs);

record struct Aggregate(
    float Precision, float Recall, float F1,
    float AP50, float mAP,
    float IoUAvg, float IoU50, float IoU75, float IoU90,
    float CenterErr, float CornerErr,
    float AreaRatioMean, float AreaRatioStd,
    float AspectDiffMean, float AspectDiffStd,
    double AvgMs, double MedianMs, double StdMs,
    double P50Ms, double P90Ms, double P99Ms,
    int FP, int FN, double FPS);

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
        if (images.Length > 100)
            images = images.Take(100).ToArray();

        var detr = EvaluateDetr(images, labelsDir);
        var yolo = EvaluateYolo(images, labelsDir);

        Console.WriteLine($"DETR    Precision: {detr.Precision:F3} Recall: {detr.Recall:F3} F1: {detr.F1:F3} mAP50: {detr.AP50:F3} mAP: {detr.mAP:F3} FPS: {detr.FPS:F1}");
        Console.WriteLine($"YOLOv8 Precision: {yolo.Precision:F3} Recall: {yolo.Recall:F3} F1: {yolo.F1:F3} mAP50: {yolo.AP50:F3} mAP: {yolo.mAP:F3} FPS: {yolo.FPS:F1}");
    }

    static Aggregate EvaluateDetr(string[] images, string labelsDir)
    {
        using var det = new SignatureDetector(OnnxPath);
        return Evaluate(img => det.Predict(img), images, labelsDir, true, "detr");
    }

    static Aggregate EvaluateYolo(string[] images, string labelsDir)
    {
        if (!File.Exists(YoloPath))
            return new Aggregate();
        using var det = new YoloV8Detector(YoloPath);
        return Evaluate(img => det.Predict(img), images, labelsDir, false, "yolo");
    }

    static Aggregate Evaluate(Func<string, float[][]> predict, string[] images, string labelsDir, bool convertFromDetr, string prefix)
    {
        var preds = new List<Detection>();
        var gts = new List<GroundTruth>();
        var times = new List<double>();
        var matchedIoU = new List<float>();
        var centerErr = new List<float>();
        var cornerErr = new List<float>();
        var areaRatio = new List<float>();
        var aspectDiff = new List<float>();
        var perImageFP = new List<int>();
        var details = new List<ImageDetail>();
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
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var dets = predict(img);
            sw.Stop();
            times.Add(sw.Elapsed.TotalMilliseconds);
            var localPreds = new List<Detection>();
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
                var detRec = new Detection(x1, y1, x2, y2, score, idx);
                preds.Add(detRec);
                localPreds.Add(detRec);
            }

            var localGTs = gts.Where(g => g.Image == idx).ToList();
            var usedLocal = new bool[localGTs.Count];
            int localTP = 0;
            var localIoU = new List<float>();
            foreach (var p in localPreds)
            {
                float best = 0.5f;
                int match = -1;
                for (int j = 0; j < localGTs.Count; j++)
                {
                    if (usedLocal[j]) continue;
                    float iou = IoU(p, localGTs[j]);
                    if (iou >= best)
                    {
                        best = iou;
                        match = j;
                    }
                }
                if (match != -1)
                {
                    usedLocal[match] = true;
                    localTP++;
                    localIoU.Add(best);
                }
            }
            int localFP = localPreds.Count - localTP;
            int localFN = localGTs.Count - localTP;
            details.Add(new ImageDetail(idx, sw.Elapsed.TotalMilliseconds, localPreds.Count, localGTs.Count, localTP, localFP, localFN, localIoU.ToArray()));

            idx++;
        }

        var agg = ComputeMetrics(preds, gts, times, matchedIoU, centerErr, cornerErr, areaRatio, aspectDiff, perImageFP);
        var payload = new { metrics = agg, details, times, ious = matchedIoU };
        File.WriteAllText(Path.Combine(Root, $"metrics_{prefix}.json"), JsonSerializer.Serialize(payload));
        return agg;
    }

    static Aggregate ComputeMetrics(List<Detection> preds, List<GroundTruth> gts,
        List<double> times, List<float> matchedIoU, List<float> centerErr, List<float> cornerErr,
        List<float> areaRatio, List<float> aspectDiff, List<int> perImageFP,
        float iouThr = 0.5f)
    {
        var sorted = preds.OrderByDescending(p => p.Score).ToArray();
        var used = new bool[gts.Count];
        float[] tp = new float[sorted.Length];
        float[] fp = new float[sorted.Length];
        var fpPerImageTemp = new Dictionary<int, int>();

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
                matchedIoU.Add(best);
                var gt = gts[match];
                float pcx = (p.X1 + p.X2) / 2f;
                float pcy = (p.Y1 + p.Y2) / 2f;
                float gcx = (gt.X1 + gt.X2) / 2f;
                float gcy = (gt.Y1 + gt.Y2) / 2f;
                centerErr.Add(MathF.Sqrt((pcx - gcx)*(pcx - gcx) + (pcy - gcy)*(pcy - gcy)));
                float pe1 = MathF.Pow(p.X1 - gt.X1,2) + MathF.Pow(p.Y1 - gt.Y1,2);
                float pe2 = MathF.Pow(p.X2 - gt.X2,2) + MathF.Pow(p.Y1 - gt.Y1,2);
                float pe3 = MathF.Pow(p.X2 - gt.X2,2) + MathF.Pow(p.Y2 - gt.Y2,2);
                float pe4 = MathF.Pow(p.X1 - gt.X1,2) + MathF.Pow(p.Y2 - gt.Y2,2);
                cornerErr.Add(MathF.Sqrt(pe1 + pe2 + pe3 + pe4));
                float areaP = MathF.Max(0, p.X2 - p.X1) * MathF.Max(0, p.Y2 - p.Y1);
                float areaG = MathF.Max(0, gt.X2 - gt.X1) * MathF.Max(0, gt.Y2 - gt.Y1);
                areaRatio.Add(areaG > 0 ? areaP / areaG : 0);
                float arP = (p.X2 - p.X1) / MathF.Max(1e-6f, (p.Y2 - p.Y1));
                float arG = (gt.X2 - gt.X1) / MathF.Max(1e-6f, (gt.Y2 - gt.Y1));
                aspectDiff.Add(arP - arG);
            }
            else
            {
                fp[i] = 1;
                if (!fpPerImageTemp.ContainsKey(p.Image)) fpPerImageTemp[p.Image] = 0;
                fpPerImageTemp[p.Image]++;
            }
        }

        int totalImages = gts.Select(g => g.Image).Concat(preds.Select(p => p.Image)).DefaultIfEmpty(0).Max() + 1;
        for (int i = 0; i < totalImages; i++)
            perImageFP.Add(fpPerImageTemp.TryGetValue(i, out int v) ? v : 0);

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

        float ap50 = 0;
        for (int i = 0; i < mRec.Count - 1; i++)
            ap50 += (mRec[i + 1] - mRec[i]) * mPre[i + 1];

        float map = 0f;
        for (float thr = 0.5f; thr <= 0.95f + 1e-5; thr += 0.05f)
            map += APAtThreshold(preds, gts, thr);
        map /= 10f;

        float P = sorted.Length > 0 ? cumTP[^1] / sorted.Length : 0;
        float R = gts.Count > 0 ? cumTP[^1] / gts.Count : 0;
        float F1 = (P + R) > 0 ? 2 * P * R / (P + R) : 0;

        int tpCount = (int)cumTP[^1];
        int fnCount = gts.Count - tpCount;

        double avgMs = times.Average();
        var sortedT = times.OrderBy(v => v).ToArray();
        double medianMs = sortedT.Length > 0 ? sortedT[sortedT.Length/2] : 0;
        double stdMs = Math.Sqrt(times.Select(t => Math.Pow(t - avgMs,2)).Sum() / times.Count);
        double p50 = Percentile(sortedT,0.5);
        double p90 = Percentile(sortedT,0.9);
        double p99 = Percentile(sortedT,0.99);
        double fps = times.Sum() > 0 ? times.Count * 1000.0 / times.Sum() : 0;

        float iouAvg = matchedIoU.Count > 0 ? matchedIoU.Average() : 0f;
        float iou50 = matchedIoU.Count > 0 ? matchedIoU.Count(m => m > 0.50f) / (float)matchedIoU.Count : 0f;
        float iou75 = matchedIoU.Count > 0 ? matchedIoU.Count(m => m > 0.75f) / (float)matchedIoU.Count : 0f;
        float iou90 = matchedIoU.Count > 0 ? matchedIoU.Count(m => m > 0.90f) / (float)matchedIoU.Count : 0f;

        float centerMean = centerErr.Count > 0 ? centerErr.Average() : 0f;
        float cornerMean = cornerErr.Count > 0 ? cornerErr.Average() : 0f;
        float areaMean = areaRatio.Count > 0 ? areaRatio.Average() : 0f;
        float areaStd = areaRatio.Count > 0 ? StdDev(areaRatio.Select(v=>(double)v)) : 0f;
        float aspectMean = aspectDiff.Count > 0 ? aspectDiff.Average() : 0f;
        float aspectStd = aspectDiff.Count > 0 ? StdDev(aspectDiff.Select(v=>(double)v)) : 0f;

        int fpCount = fp.Sum(f => (int)f);

        return new Aggregate(P,R,F1,ap50,map,iouAvg,iou50,iou75,iou90,centerMean,
            cornerMean,areaMean,areaStd,aspectMean,aspectStd,avgMs,medianMs,stdMs,
            p50,p90,p99,fpCount,fnCount,fps);
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

    static float APAtThreshold(List<Detection> preds, List<GroundTruth> gts, float thr)
    {
        var sorted = preds.OrderByDescending(p => p.Score).ToArray();
        var used = new bool[gts.Count];
        float[] tp = new float[sorted.Length];
        float[] fp = new float[sorted.Length];
        for (int i = 0; i < sorted.Length; i++)
        {
            var p = sorted[i];
            float best = thr;
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

        var mRec = new List<float> { 0 };
        var mPre = new List<float> { cumTP.Length > 0 ? cumTP[0] / (cumTP[0] + cumFP[0]) : 0 };
        for (int i = 0; i < sorted.Length; i++)
        {
            float rec = gts.Count > 0 ? cumTP[i] / gts.Count : 0;
            float pre = (cumTP[i] + cumFP[i]) > 0 ? cumTP[i] / (cumTP[i] + cumFP[i]) : 0;
            mRec.Add(rec);
            mPre.Add(pre);
        }
        mRec.Add(1); mPre.Add(0);
        for (int i = mPre.Count - 2; i >= 0; i--)
            mPre[i] = MathF.Max(mPre[i], mPre[i + 1]);

        float ap = 0;
        for (int i = 0; i < mRec.Count - 1; i++)
            ap += (mRec[i + 1] - mRec[i]) * mPre[i + 1];
        return ap;
    }

    static double Percentile(double[] sorted, double p)
    {
        if (sorted.Length == 0) return 0;
        double pos = p * (sorted.Length + 1);
        int idx = (int)pos;
        if (idx < 1) return sorted[0];
        if (idx >= sorted.Length) return sorted[^1];
        double diff = pos - idx;
        return sorted[idx - 1] + diff * (sorted[idx] - sorted[idx - 1]);
    }

    static float StdDev(IEnumerable<double> values)
    {
        var arr = values.ToArray();
        if (arr.Length == 0) return 0f;
        double avg = arr.Average();
        double sum = arr.Sum(v => (v - avg) * (v - avg));
        return (float)Math.Sqrt(sum / arr.Length);
    }
}
