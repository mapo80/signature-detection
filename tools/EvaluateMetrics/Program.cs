using SignatureDetectionSdk;
using SkiaSharp;
using System.Globalization;
using System.Text.Json;
using System.Linq;
using System.Collections.Generic;
using System.Text;

record struct Detection(float X1, float Y1, float X2, float Y2, float Score, int Image);
record struct GroundTruth(float X1, float Y1, float X2, float Y2, int Image);

record struct ImageDetail(int Image, double InfMs, double PostMs, int Pred, int GroundTruth, int TP, int FP, int FN,
    float OverlapMean, int OverlapMax, float Brightness, float Noise, float AvgBoxArea,
    float[] IoUs, float[] ConfTP, float[] ConfFP);

record struct Aggregate(
    float Precision, float Recall, float F1,
    float AP50, float mAP,
    float IoUAvg, float IoU50, float IoU75, float IoU90,
    float CenterErr, float CornerErr,
    float AreaRatioMean, float AreaRatioStd,
    float AspectDiffMean, float AspectDiffStd,
    double AvgInfMs, double MedianInfMs, double StdInfMs,
    double P50InfMs, double P90InfMs, double P99InfMs,
    double AvgPostMs, double MedianPostMs, double StdPostMs,
    int FP, int FN, double FPS);

record struct ThresholdResult(float Threshold, float Precision, float Recall, float F1, float AP50, float mAP);

record class PipelineConfig
{
    public bool EnableYoloV8 { get; init; } = true;
    public bool EnableDetr { get; init; } = true;
    public string Strategy { get; init; } = "SequentialFallback"; // or Parallel
    public float YoloConfidenceThreshold { get; init; } = 0.6f;
    public float YoloNmsIoU { get; init; } = 0.3f;
    public float DetrConfidenceThreshold { get; init; } = 0.3f;
    public int FallbackFp { get; init; } = 2;
    public int FallbackFn { get; init; } = 0;
}

class Program
{
    static string Root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../"));
    static string OnnxPath = Path.Combine(Root, "conditional_detr_signature.onnx");
    static string YoloPath = Path.Combine(Root, "yolov8s.onnx");

    static PipelineConfig LoadConfig(string path)
    {
        if (File.Exists(path))
        {
            try
            {
                var text = File.ReadAllText(path);
                var cfg = JsonSerializer.Deserialize<PipelineConfig>(text);
                if (cfg != null) return cfg;
            }
            catch { }
        }
        return new PipelineConfig();
    }

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
        var cfg = LoadConfig(Path.Combine(Root, "config.json"));
        for(int i=1;i<args.Length-1;i+=2)
        {
            var key = args[i].TrimStart('-');
            var val = args[i+1];
            switch(key)
            {
                case "enable-yolov8": cfg = cfg with { EnableYoloV8 = bool.Parse(val) }; break;
                case "enable-detr": cfg = cfg with { EnableDetr = bool.Parse(val) }; break;
                case "strategy": cfg = cfg with { Strategy = val }; break;
                case "yoloConfidenceThreshold": cfg = cfg with { YoloConfidenceThreshold = float.Parse(val,CultureInfo.InvariantCulture) }; break;
                case "yoloNmsIoU": cfg = cfg with { YoloNmsIoU = float.Parse(val,CultureInfo.InvariantCulture) }; break;
                case "detrConfidenceThreshold": cfg = cfg with { DetrConfidenceThreshold = float.Parse(val,CultureInfo.InvariantCulture) }; break;
            }
        }

        EnsureModel();
        string imagesDir = Path.Combine(Root, "dataset", dataset, "images");
        string labelsDir = Path.Combine(Root, "dataset", dataset, "labels");
        var images = Directory.GetFiles(imagesDir, "*.jpg").OrderBy(f => f).ToArray();
        if (images.Length > 100)
            images = images.Take(100).ToArray();

        EvalData? detr = null;
        EvalData? yolo = null;
        if(cfg.EnableDetr)
            detr = RunModel(img => new SignatureDetector(OnnxPath).Predict(img, cfg.DetrConfidenceThreshold), images, labelsDir, true, cfg);
        if(cfg.EnableYoloV8 && File.Exists(YoloPath))
            yolo = RunModel(img => new YoloV8Detector(YoloPath).Predict(img, cfg.YoloConfidenceThreshold), images, labelsDir, false, cfg);

        var detrAgg = detr.HasValue ? ComputeMetrics(detr.Value.Preds, detr.Value.GTs, detr.Value.InfTimes, detr.Value.PostTimes, detr.Value.MatchedIoU, detr.Value.CenterErr, detr.Value.CornerErr, detr.Value.AreaRatio, detr.Value.AspectDiff, detr.Value.PerImageFP) : new Aggregate();
        var yoloAgg = yolo.HasValue ? ComputeMetrics(yolo.Value.Preds, yolo.Value.GTs, yolo.Value.InfTimes, yolo.Value.PostTimes, yolo.Value.MatchedIoU, yolo.Value.CenterErr, yolo.Value.CornerErr, yolo.Value.AreaRatio, yolo.Value.AspectDiff, yolo.Value.PerImageFP) : new Aggregate();

        var thresholds = new[] {0.25f,0.5f,0.75f,0.9f};
        var detrTh = detr.HasValue ? thresholds.Select(t => ThresholdMetrics(detr.Value.Preds, detr.Value.GTs, t)).ToArray() : Array.Empty<ThresholdResult>();
        var yoloTh = yolo.HasValue ? thresholds.Select(t => ThresholdMetrics(yolo.Value.Preds, yolo.Value.GTs, t)).ToArray() : Array.Empty<ThresholdResult>();

        if (detr.HasValue) SaveJson("detr", detrAgg, detr.Value.Details, detr.Value.InfTimes, detr.Value.PostTimes, detr.Value.MatchedIoU, detrTh);
        if (yolo.HasValue) SaveJson("yolo", yoloAgg, yolo.Value.Details, yolo.Value.InfTimes, yolo.Value.PostTimes, yolo.Value.MatchedIoU, yoloTh);

        if (detr.HasValue && yolo.HasValue)
        {
            List<Detection> combined;
            if (cfg.Strategy.StartsWith("Parallel", StringComparison.OrdinalIgnoreCase))
                combined = detr.Value.Preds.Concat(yolo.Value.Preds).ToList();
            else
                combined = CombineFallback(detr.Value, yolo.Value, cfg);
            var combAgg = ComputeMetrics(combined, detr.Value.GTs, new List<double>(), new List<double>(), new List<float>(), new List<float>(), new List<float>(), new List<float>(), new List<float>(), new List<int>());
            Console.WriteLine($"Combined Precision: {combAgg.Precision:F3} Recall: {combAgg.Recall:F3} F1: {combAgg.F1:F3} mAP: {combAgg.mAP:F3}");
        }

        if(detr.HasValue)
            Console.WriteLine($"DETR    Precision: {detrAgg.Precision:F3} Recall: {detrAgg.Recall:F3} F1: {detrAgg.F1:F3} mAP50: {detrAgg.AP50:F3} mAP: {detrAgg.mAP:F3} FPS: {detrAgg.FPS:F1}");
        if (yolo.HasValue)
            Console.WriteLine($"YOLOv8 Precision: {yoloAgg.Precision:F3} Recall: {yoloAgg.Recall:F3} F1: {yoloAgg.F1:F3} mAP50: {yoloAgg.AP50:F3} mAP: {yoloAgg.mAP:F3} FPS: {yoloAgg.FPS:F1}");
    }

    record struct EvalData(List<Detection> Preds, List<GroundTruth> GTs, List<ImageDetail> Details,
        List<double> InfTimes, List<double> PostTimes, List<float> MatchedIoU, List<float> CenterErr, List<float> CornerErr,
        List<float> AreaRatio, List<float> AspectDiff, List<int> PerImageFP);

    static EvalData RunModel(Func<string, float[][]> predict, string[] images, string labelsDir, bool convert, PipelineConfig cfg)
    {
        var preds = new List<Detection>();
        var gts = new List<GroundTruth>();
        var details = new List<ImageDetail>();
        var times = new List<double>();
        var postTimes = new List<double>();
        var matchedIoU = new List<float>();
        var centerErr = new List<float>();
        var cornerErr = new List<float>();
        var areaRatio = new List<float>();
        var aspectDiff = new List<float>();
        var perImageFP = new List<int>();

        int idx = 0;
        foreach (var img in images)
        {
            using var original = SKBitmap.Decode(img);
            using var bmp = Preprocess(original, cfg);
            int w = bmp.Width;
            int h = bmp.Height;

            double sumV=0, sumV2=0;
            for(int y=0;y<h;y++)
            {
                for(int x=0;x<w;x++)
                {
                    var c = bmp.GetPixel(x,y);
                    c.ToHsv(out _, out _, out float v);
                    sumV += v;
                    sumV2 += v*v;
                }
            }
            float brightness = (float)(sumV/(w*h));
            float noise = (float)(sumV2/(w*h) - brightness*brightness);

            var labelPath = Path.Combine(labelsDir, Path.GetFileNameWithoutExtension(img) + ".txt");
            var gtForImg = new List<GroundTruth>();
            if (File.Exists(labelPath))
            {
                foreach (var line in File.ReadLines(labelPath))
                {
                    var p = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (p.Length >= 5)
                    {
                        float cx = float.Parse(p[1], CultureInfo.InvariantCulture);
                        float cy = float.Parse(p[2], CultureInfo.InvariantCulture);
                        float bw = float.Parse(p[3], CultureInfo.InvariantCulture);
                        float bh = float.Parse(p[4], CultureInfo.InvariantCulture);
                        float x1 = (cx - bw/2f) * w;
                        float y1 = (cy - bh/2f) * h;
                        float x2 = (cx + bw/2f) * w;
                        float y2 = (cy + bh/2f) * h;
                        var gt = new GroundTruth(x1,y1,x2,y2,idx);
                        gts.Add(gt);
                        gtForImg.Add(gt);
                    }
                }
            }
            float avgGtArea = gtForImg.Count>0 ? gtForImg.Average(g => (g.X2-g.X1)*(g.Y2-g.Y1)) : 0f;

            var swInf = System.Diagnostics.Stopwatch.StartNew();
            var result = predict(img);
            swInf.Stop();
            times.Add(swInf.Elapsed.TotalMilliseconds);
            var swPost = System.Diagnostics.Stopwatch.StartNew();

        var tempPreds = new List<Detection>();
        foreach (var d in result)
        {
                float x1 = d[0];
                float y1 = d[1];
                float x2 = d[2];
                float y2 = d[3];
                float sc = d[4];
                if (convert)
                {
                    x1 = x1 / 640f * w;
                    y1 = y1 / 640f * h;
                    x2 = x2 / 640f * w;
                    y2 = y2 / 640f * h;
                }
            var det = new Detection(x1,y1,x2,y2,sc,idx);
            tempPreds.Add(det);
        }
        if(!convert)
            tempPreds = ApplyNms(tempPreds, cfg.YoloNmsIoU);
        var imagePreds = tempPreds;
        preds.AddRange(imagePreds);

            var used = new bool[gtForImg.Count];
            var ious = new List<float>();
            var confTP = new List<float>();
            var confFP = new List<float>();
            int tp=0;
            int fp=0;
            foreach (var p in imagePreds)
            {
                float best=0.5f; int match=-1; float bestIoU=0f;
                for(int j=0;j<gtForImg.Count;j++)
                {
                    if(used[j]) continue;
                    float iou = IoU(p, gtForImg[j]);
                    if(iou>=best)
                    {
                        best=iou; match=j; bestIoU=iou;
                    }
                }
                if(match!=-1)
                {
                    used[match]=true; tp++; ious.Add(bestIoU); confTP.Add(p.Score);
                }
                else { fp++; confFP.Add(p.Score); }
            }
            int fn = gtForImg.Count - tp;

            // overlap counts
            int overlaps=0; int pairs=0; int maxOverlap=0;
            for(int i=0;i<imagePreds.Count;i++)
            {
                int local=0;
                for(int j=i+1;j<imagePreds.Count;j++)
                {
                    if(IoU(imagePreds[i], imagePreds[j])>0.3f) { overlaps++; local++; }
                }
                if(local>maxOverlap) maxOverlap=local;
                pairs++;
            }
            float overlapMean = pairs>0 ? (float)overlaps/pairs : 0f;

            swPost.Stop();
            postTimes.Add(swPost.Elapsed.TotalMilliseconds);

            details.Add(new ImageDetail(idx, swInf.Elapsed.TotalMilliseconds, swPost.Elapsed.TotalMilliseconds,
                imagePreds.Count, gtForImg.Count,
                tp, fp, fn, overlapMean, maxOverlap, brightness, noise, avgGtArea,
                ious.ToArray(), confTP.ToArray(), confFP.ToArray()));

            idx++;
        }

        return new EvalData(preds,gts,details,times,postTimes,matchedIoU,centerErr,cornerErr,areaRatio,aspectDiff,perImageFP);
    }

    static void SaveJson(string prefix, Aggregate metrics, List<ImageDetail> details, List<double> infTimes, List<double> postTimes, List<float> ious, ThresholdResult[] thresholds)
    {
        var payload = new { metrics, details, infTimes, postTimes, ious, thresholds };
        File.WriteAllText(Path.Combine(Root, $"metrics_{prefix}.json"), JsonSerializer.Serialize(payload));
    }

    static ThresholdResult ThresholdMetrics(List<Detection> preds, List<GroundTruth> gts, float thr)
    {
        var filtered = preds.Where(p => p.Score >= thr).ToList();
        var agg = ComputeMetrics(filtered, gts, new List<double>(), new List<double>(), new List<float>(), new List<float>(), new List<float>(), new List<float>(), new List<float>(), new List<int>());
        return new ThresholdResult(thr, agg.Precision, agg.Recall, agg.F1, agg.AP50, agg.mAP);
    }


    static Aggregate ComputeMetrics(List<Detection> preds, List<GroundTruth> gts,
        List<double> infTimes, List<double> postTimes, List<float> matchedIoU, List<float> centerErr, List<float> cornerErr,
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

        int tpCount = sorted.Length > 0 ? (int)cumTP[^1] : 0;
        int fnCount = gts.Count - tpCount;

        double avgInf = infTimes.Count>0 ? infTimes.Average() : 0;
        var sortedInf = infTimes.OrderBy(v => v).ToArray();
        double medianInf = sortedInf.Length > 0 ? sortedInf[sortedInf.Length/2] : 0;
        double stdInf = infTimes.Count>0 ? Math.Sqrt(infTimes.Select(t => Math.Pow(t - avgInf,2)).Sum() / infTimes.Count) : 0;
        double p50Inf = Percentile(sortedInf,0.5);
        double p90Inf = Percentile(sortedInf,0.9);
        double p99Inf = Percentile(sortedInf,0.99);
        double fps = infTimes.Sum() > 0 ? infTimes.Count * 1000.0 / infTimes.Sum() : 0;

        double avgPost = postTimes.Count>0 ? postTimes.Average() : 0;
        var sortedPost = postTimes.OrderBy(v => v).ToArray();
        double medianPost = sortedPost.Length>0 ? sortedPost[sortedPost.Length/2] : 0;
        double stdPost = postTimes.Count>0 ? Math.Sqrt(postTimes.Select(t => Math.Pow(t - avgPost,2)).Sum()/postTimes.Count) : 0;

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
            cornerMean,areaMean,areaStd,aspectMean,aspectStd,
            avgInf,medianInf,stdInf,p50Inf,p90Inf,p99Inf,
            avgPost,medianPost,stdPost,
            fpCount,fnCount,fps);
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

    static float IoU(Detection a, Detection b)
    {
        float xx1 = MathF.Max(a.X1, b.X1);
        float yy1 = MathF.Max(a.Y1, b.Y1);
        float xx2 = MathF.Min(a.X2, b.X2);
        float yy2 = MathF.Min(a.Y2, b.Y2);
        float inter = MathF.Max(0, xx2 - xx1) * MathF.Max(0, yy2 - yy1);
        if (inter <= 0) return 0f;
        float areaA = MathF.Max(0, a.X2 - a.X1) * MathF.Max(0, a.Y2 - a.Y1);
        float areaB = MathF.Max(0, b.X2 - b.X1) * MathF.Max(0, b.Y2 - b.Y1);
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

    static SKBitmap Preprocess(SKBitmap input, PipelineConfig cfg)
    {
        if (!cfg.EnableDetr && !cfg.EnableYoloV8) return input;
        var bmp = new SKBitmap(input.Info.Width, input.Info.Height);
        input.CopyTo(bmp);
        if (cfg != null)
        {
            // simple 3x3 median
            var rWin = new byte[9];
            var gWin = new byte[9];
            var bWin = new byte[9];
            for (int y = 1; y < bmp.Height - 1; y++)
            {
                for (int x = 1; x < bmp.Width - 1; x++)
                {
                    int k = 0;
                    for (int yy = -1; yy <= 1; yy++)
                        for (int xx = -1; xx <= 1; xx++)
                        {
                            var c = input.GetPixel(x + xx, y + yy);
                            rWin[k] = c.Red; gWin[k] = c.Green; bWin[k] = c.Blue; k++;
                        }
                    Array.Sort(rWin); Array.Sort(gWin); Array.Sort(bWin);
                    bmp.SetPixel(x, y, new SKColor(rWin[4], gWin[4], bWin[4]));
                }
            }

            // histogram equalization on V channel
            int[] hist = new int[256];
            for (int y = 0; y < bmp.Height; y++)
                for (int x = 0; x < bmp.Width; x++)
                {
                    bmp.GetPixel(x, y).ToHsv(out _, out _, out float v);
                    int idx = (int)(v * 255f);
                    hist[idx]++;
                }
            int total = bmp.Width * bmp.Height;
            float[] cdf = new float[256];
            int sum = 0;
            for (int i = 0; i < 256; i++) { sum += hist[i]; cdf[i] = sum / (float)total; }
            for (int y = 0; y < bmp.Height; y++)
                for (int x = 0; x < bmp.Width; x++)
                {
                    var c = bmp.GetPixel(x, y);
                    c.ToHsv(out float h, out float s, out float v);
                    int idx = (int)(v * 255f);
                    float nv = cdf[idx];
                    var newC = SKColor.FromHsv(h, s, nv);
                    bmp.SetPixel(x, y, newC);
                }
        }
        return bmp;
    }

    static List<Detection> ApplyNms(List<Detection> dets, float iou)
    {
        var result = new List<Detection>();
        var sorted = dets.OrderByDescending(d => d.Score).ToList();
        var removed = new bool[sorted.Count];
        for (int i = 0; i < sorted.Count; i++)
        {
            if (removed[i]) continue;
            var a = sorted[i];
            result.Add(a);
            for (int j = i + 1; j < sorted.Count; j++)
            {
                if (removed[j]) continue;
                if (IoU(a, sorted[j]) > iou) removed[j] = true;
            }
        }
        return result;
    }

    static List<Detection> CombineFallback(EvalData detr, EvalData yolo, PipelineConfig cfg)
    {
        var list = new List<Detection>();
        for (int i = 0; i < yolo.Details.Count; i++)
        {
            var yd = yolo.Details[i];
            bool useDetr = yd.FP > cfg.FallbackFp || yd.FN > cfg.FallbackFn;
            var preds = useDetr ? detr.Preds.Where(p => p.Image == i) : yolo.Preds.Where(p => p.Image == i);
            list.AddRange(preds);
        }
        return list;
    }
}
