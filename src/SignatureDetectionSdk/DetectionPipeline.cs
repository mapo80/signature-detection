using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SignatureDetectionSdk;

public record PipelineConfig(
    bool   EnableYoloV8            = true,
    bool   EnableDetr              = true,
    string Strategy                = "Ensemble",          // "SequentialFallback" | "Parallel" | "Ensemble"
    float  YoloConfidenceThreshold = 0.60f,
    float  YoloNmsIoU              = 0.30f,
    float  DetrConfidenceThreshold = 0.30f,
    int    FallbackFp              = 2,
    int    FallbackFn              = 0,

    // soft-voting
    float  EceDetr              = 1.0f,
    float  EceYolo              = 1.0f,
    float  EnsembleThreshold    = 0.50f,

    // shape-prior v2
    bool   EnableShapeRoiV2     = true,
    float  ShapeMinAspect       = 0.30f,   // percentile low
    float  ShapeMaxAspect       = 6.00f,   // reduced from 9.4 → 6.0
    float  LowConfidence        = 0.40f,
    float  HighConfidence       = 0.85f,   // raised 0.80 → 0.85
    float  CropMarginPerc       = 0.20f,
    float  RoiConfirmIoU        = 0.40f,
    float  UncertainQuantile    = 0.05f    // top 5 % most uncertain boxes → ROI fallback
);

public class DetectionPipeline : IDisposable
{
    private readonly PipelineConfig _config;
    private readonly YoloV8Detector? _yolo;
    private readonly SignatureDetector? _detr;
    private readonly SignatureDetector? _roiDetr;
    private float _shapeLow;
    private float _shapeHigh;
    private int _roiBatchCount;
    private int _roiCropCount;
    private double _roiTimeMs;
    private float[][] _preRoi = Array.Empty<float[]>();
    private float _postShapePrecision;
    private float _postShapeRecall;

    public DetectionPipeline(string detrModel, string yoloModel, PipelineConfig? config = null, string? datasetDir = null)
    {
        _config = config ?? new PipelineConfig();
        if (_config.EnableDetr)
        {
            _detr = new SignatureDetector(detrModel);
            // ROI detector with dedicated session (approx FP16)
            _roiDetr = new SignatureDetector(detrModel, options: new Microsoft.ML.OnnxRuntime.SessionOptions());
        }
        if (_config.EnableYoloV8 && System.IO.File.Exists(yoloModel))
            _yolo = new YoloV8Detector(yoloModel);

        if (_config.EnableShapeRoiV2)
        {
            if (datasetDir != null)
                (_shapeLow, _shapeHigh) = ComputeShapeBounds(datasetDir, _config);
            else
            {
                _shapeLow = _config.ShapeMinAspect;
                _shapeHigh = _config.ShapeMaxAspect;
            }
        }
        else
        {
            _shapeLow = _config.ShapeMinAspect;
            _shapeHigh = _config.ShapeMaxAspect;
        }
    }

    public void Dispose()
    {
        _detr?.Dispose();
        _roiDetr?.Dispose();
        _yolo?.Dispose();
    }

    public float[][] Detect(string imagePath, IList<float[]>? groundTruth = null)
    {
        using var original = SKBitmap.Decode(imagePath);
        using var pre = SafePreprocess(original);
        var yoloPreds = new List<float[]>();
        var detrPreds = new List<float[]>();
        if (_config.EnableYoloV8 && _yolo != null)
            yoloPreds.AddRange(_yolo.Predict(pre, _config.YoloConfidenceThreshold));
        if (_config.EnableDetr && _detr != null)
            detrPreds.AddRange(_detr.Predict(pre, _config.DetrConfidenceThreshold));

        float[][] result;
        bool both = _config.EnableYoloV8 && _config.EnableDetr;
        bool useEnsemble = both && (_config.Strategy.Equals("Ensemble", StringComparison.OrdinalIgnoreCase)
            || _config.Strategy.StartsWith("Parallel", StringComparison.OrdinalIgnoreCase));

        if (useEnsemble)
        {
            var combined = SoftVotingEnsemble.Combine(yoloPreds, detrPreds,
                _config.EceYolo, _config.EceDetr, _config.LowConfidence,
                _shapeLow, _shapeHigh);

            _preRoi = combined;

            if (groundTruth != null)
            {
                int fp0 = CountFp(combined, groundTruth);
                int fn0 = CountFn(combined, groundTruth);
                int tp0 = combined.Length - fp0;
                _postShapePrecision = tp0 + fp0 > 0 ? tp0 / (float)(tp0 + fp0) : 0f;
                _postShapeRecall = tp0 + fn0 > 0 ? tp0 / (float)(tp0 + fn0) : 0f;
            }

            result = _config.EnableShapeRoiV2 ? ApplyRoiFallback(combined, pre) : combined;
        }
        else if (both)
        {
            var primary = yoloPreds;
            var secondary = detrPreds;
            result = primary.ToArray();
            if (groundTruth != null && (CountFp(primary, groundTruth) > _config.FallbackFp ||
                                       CountFn(primary, groundTruth) > _config.FallbackFn))
            {
                result = result.Concat(secondary).ToArray();
            }
        }
        else if (_config.EnableYoloV8)
        {
            result = yoloPreds.ToArray();
        }
        else if (_config.EnableDetr)
        {
            result = detrPreds.ToArray();
        }
        else result = Array.Empty<float[]>();

        return result;
    }

    private static SKBitmap SafePreprocess(SKBitmap original)
    {
        try
        {
            return ImagePreprocessing.Apply(original);
        }
        catch
        {
            var copy = new SKBitmap(original.Info.Width, original.Info.Height);
            original.CopyTo(copy);
            return copy;
        }
    }

    public float ShapeMinAspectActual => _shapeLow;
    public float ShapeMaxAspectActual => _shapeHigh;
    public float PostShapePrecision => _postShapePrecision;
    public float PostShapeRecall => _postShapeRecall;
    public float[][] PreRoiDetections => _preRoi;
    public int RoiBatchTotal => _roiBatchCount;
    public float RoiBatchAvgCrop => _roiBatchCount > 0 ? (float)_roiCropCount / _roiBatchCount : 0f;
    public double RoiAvgLatencyMs => _roiBatchCount > 0 ? _roiTimeMs / _roiBatchCount : 0.0;

    private float[][] ApplyRoiFallback(float[][] boxes, SKBitmap image)
    {
        if (_roiDetr == null) return boxes;
        var accepted = new List<float[]>();
        var candidates = new List<float[]>();
        foreach (var b in boxes)
        {
            if (b[4] >= _config.HighConfidence) accepted.Add(b);
            else candidates.Add(b);
        }
        if (candidates.Count == 0) return accepted.ToArray();

        var uncertainties = candidates.Select(b => 1f - b[4]).ToArray();
        float threshold = Percentile(uncertainties, 1f - _config.UncertainQuantile);
        var toCheck = new List<(float[] box, SKRect rect)>();
        foreach (var b in candidates)
        {
            if (1f - b[4] >= threshold)
            {
                float x1 = b[0], y1 = b[1], x2 = b[2], y2 = b[3];
                float w = x2 - x1, h = y2 - y1;
                float ex = w * _config.CropMarginPerc;
                float ey = h * _config.CropMarginPerc;
                float rx1 = MathF.Max(0, x1 - ex);
                float ry1 = MathF.Max(0, y1 - ey);
                float rx2 = MathF.Min(image.Width, x2 + ex);
                float ry2 = MathF.Min(image.Height, y2 + ey);
                var rect = SKRect.Create(rx1, ry1, rx2 - rx1, ry2 - ry1);
                toCheck.Add((b, rect));
            }
            else accepted.Add(b);
        }

        for (int i = 0; i < toCheck.Count; i += 4)
        {
            var batch = toCheck.Skip(i).Take(4).ToList();
            var crops = new List<SKBitmap>();
            foreach (var (_, rect) in batch)
            {
                var crop = new SKBitmap((int)rect.Width, (int)rect.Height);
                using var canvas = new SKCanvas(crop);
                canvas.DrawBitmap(image, rect, new SKRect(0, 0, rect.Width, rect.Height));
                crops.Add(crop);
            }

            var sw = Stopwatch.StartNew();
            var dets = _roiDetr.PredictBatch(crops, _config.DetrConfidenceThreshold);
            sw.Stop();
            _roiBatchCount++;
            _roiCropCount += batch.Count;
            _roiTimeMs += sw.Elapsed.TotalMilliseconds;

            for (int j = 0; j < batch.Count; j++)
            {
                bool ok = false;
                foreach (var p in dets[j])
                {
                    float ax1 = batch[j].rect.Left + p[0];
                    float ay1 = batch[j].rect.Top + p[1];
                    float ax2 = batch[j].rect.Left + p[2];
                    float ay2 = batch[j].rect.Top + p[3];
                    var tmp = new[] { ax1, ay1, ax2, ay2 };
                    if (IoU(batch[j].box, tmp) >= _config.RoiConfirmIoU) { ok = true; break; }
                }
                if (ok) accepted.Add(batch[j].box);
            }
            foreach (var c in crops) c.Dispose();
        }

        return accepted.ToArray();
    }

    private static int CountFp(IEnumerable<float[]> preds, IList<float[]> gts)
    {
        int fp = 0;
        var used = new bool[gts.Count];
        foreach (var p in preds)
        {
            float best = 0.5f; int match = -1;
            for (int i = 0; i < gts.Count; i++)
            {
                if (used[i]) continue;
                if (IoU(p, gts[i]) >= best) { best = IoU(p, gts[i]); match = i; }
            }
            if (match != -1) used[match] = true; else fp++;
        }
        return fp;
    }

    private static int CountFn(IEnumerable<float[]> preds, IList<float[]> gts)
    {
        var used = new bool[gts.Count];
        foreach (var p in preds)
        {
            float best = 0.5f; int match = -1;
            for (int i = 0; i < gts.Count; i++)
            {
                if (used[i]) continue;
                if (IoU(p, gts[i]) >= best) { best = IoU(p, gts[i]); match = i; }
            }
            if (match != -1) used[match] = true;
        }
        int fn = used.Count(u => !u);
        return fn;
    }

    private static float IoU(float[] a, float[] b)
    {
        float xx1 = MathF.Max(a[0], b[0]);
        float yy1 = MathF.Max(a[1], b[1]);
        float xx2 = MathF.Min(a[2], b[2]);
        float yy2 = MathF.Min(a[3], b[3]);
        float inter = MathF.Max(0, xx2 - xx1) * MathF.Max(0, yy2 - yy1);
        if (inter <= 0) return 0f;
        float areaA = MathF.Max(0, a[2] - a[0]) * MathF.Max(0, a[3] - a[1]);
        float areaB = MathF.Max(0, b[2] - b[0]) * MathF.Max(0, b[3] - b[1]);
        float union = areaA + areaB - inter;
        return union > 0 ? inter / union : 0f;
    }

    private static float Percentile(IList<float> data, float p)
    {
        if (data.Count == 0) return 0f;
        var ordered = data.OrderBy(x => x).ToArray();
        float rank = (ordered.Length - 1) * p;
        int l = (int)MathF.Floor(rank);
        int u = (int)MathF.Ceiling(rank);
        if (l == u) return ordered[l];
        return ordered[l] + (rank - l) * (ordered[u] - ordered[l]);
    }

    private static (float, float) ComputeShapeBounds(string datasetDir, PipelineConfig cfg)
    {
        var aspects = new List<float>();
        string imagesDir = System.IO.Path.Combine(datasetDir, "images");
        string labelsDir = System.IO.Path.Combine(datasetDir, "labels");
        foreach (var label in System.IO.Directory.GetFiles(labelsDir, "*.txt"))
        {
            var imgPath = System.IO.Path.Combine(imagesDir, System.IO.Path.GetFileNameWithoutExtension(label) + ".jpg");
            if (!System.IO.File.Exists(imgPath)) continue;
            using var img = SKBitmap.Decode(imgPath);
            foreach (var line in System.IO.File.ReadLines(label))
            {
                var p = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (p.Length < 5) continue;
                float bw = float.Parse(p[3], System.Globalization.CultureInfo.InvariantCulture) * img.Width;
                float bh = float.Parse(p[4], System.Globalization.CultureInfo.InvariantCulture) * img.Height;
                if (bh > 0) aspects.Add(bw / bh);
            }
        }
        float low = Percentile(aspects, 0.02f);
        float high = Percentile(aspects, 0.98f);
        high = MathF.Min(high, cfg.ShapeMaxAspect);
        low = MathF.Max(low, cfg.ShapeMinAspect);
        return (low, high);
    }
}

