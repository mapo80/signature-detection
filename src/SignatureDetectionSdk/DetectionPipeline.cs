using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;

namespace SignatureDetectionSdk;

public record PipelineConfig(
    bool EnableYoloV8 = true,
    bool EnableDetr = true,
    string Strategy = "SequentialFallback",
    float YoloConfidenceThreshold = 0.6f,
    float YoloNmsIoU = 0.3f,
    float DetrConfidenceThreshold = 0.3f,
    int FallbackFp = 2,
    int FallbackFn = 0,
    float EceDetr = 1.0f,
    float EceYolo = 1.0f,
    float EnsembleThreshold = 0.5f,
    float HighScore = 0.75f,
    int RoiBatchSize = 4
);

public class DetectionPipeline : IDisposable
{
    private readonly PipelineConfig _config;
    private readonly YoloV8Detector? _yolo;
    private readonly SignatureDetector? _detr;

    public DetectionPipeline(string detrModel, string yoloModel, PipelineConfig? config = null)
    {
        _config = config ?? new PipelineConfig();
        if (_config.EnableDetr)
            _detr = new SignatureDetector(detrModel);
        if (_config.EnableYoloV8 && System.IO.File.Exists(yoloModel))
            _yolo = new YoloV8Detector(yoloModel);
    }

    public void Dispose()
    {
        _detr?.Dispose();
        _yolo?.Dispose();
    }

    public float[][] Detect(string imagePath, IList<float[]>? groundTruth = null)
    {
        using var original = SKBitmap.Decode(imagePath);
        using var pre = ImagePreprocessing.Apply(original);
        var yoloPreds = new List<float[]>();
        var detrPreds = new List<float[]>();
        if (_config.EnableYoloV8 && _yolo != null)
            yoloPreds.AddRange(_yolo.Predict(pre, _config.YoloConfidenceThreshold));
        if (_config.EnableDetr && _detr != null)
            detrPreds.AddRange(_detr.Predict(pre, _config.DetrConfidenceThreshold));

        List<float[]> final = new();
        bool both = _config.EnableYoloV8 && _config.EnableDetr;
        bool useEnsemble = both && (_config.Strategy.Equals("Ensemble", StringComparison.OrdinalIgnoreCase)
            || _config.Strategy.StartsWith("Parallel", StringComparison.OrdinalIgnoreCase));

        if (useEnsemble)
        {
            var combined = SoftVotingEnsemble.Combine(yoloPreds, detrPreds,
                _config.EceYolo, _config.EceDetr, _config.EnsembleThreshold);
            final.AddRange(ApplyRoiFallback(combined, pre));
        }
        else if (both)
        {
            var primary = yoloPreds;
            var secondary = detrPreds;
            final.AddRange(primary);
            if (groundTruth != null && (CountFp(primary, groundTruth) > _config.FallbackFp ||
                                       CountFn(primary, groundTruth) > _config.FallbackFn))
            {
                final.AddRange(secondary);
            }
        }
        else if (_config.EnableYoloV8)
        {
            final.AddRange(yoloPreds);
        }
        else if (_config.EnableDetr)
        {
            final.AddRange(detrPreds);
        }
        return final.ToArray();
    }

    private float[][] ApplyRoiFallback(float[][] boxes, SKBitmap image)
    {
        if (_detr == null) return boxes;
        var accepted = new List<float[]>();
        var borderline = new List<float[]>();
        foreach (var b in boxes)
        {
            if (b[4] >= _config.HighScore) accepted.Add(b);
            else if (b[4] >= _config.EnsembleThreshold) borderline.Add(b);
        }
        if (borderline.Count == 0) return accepted.ToArray();

        for (int i = 0; i < borderline.Count; i += _config.RoiBatchSize)
        {
            var batch = borderline.Skip(i).Take(_config.RoiBatchSize).ToList();
            var crops = new List<SKBitmap>();
            var rects = new List<SKRect>();
            foreach (var b in batch)
            {
                float x1 = b[0], y1 = b[1], x2 = b[2], y2 = b[3];
                float w = x2 - x1, h = y2 - y1;
                float ex = w * 0.1f, ey = h * 0.1f;
                float rx1 = MathF.Max(0, x1 - ex);
                float ry1 = MathF.Max(0, y1 - ey);
                float rx2 = MathF.Min(image.Width, x2 + ex);
                float ry2 = MathF.Min(image.Height, y2 + ey);
                var rect = SKRect.Create(rx1, ry1, rx2 - rx1, ry2 - ry1);
                rects.Add(rect);
                var crop = new SKBitmap((int)rect.Width, (int)rect.Height);
                using var canvas = new SKCanvas(crop);
                canvas.DrawBitmap(image, rect, new SKRect(0, 0, rect.Width, rect.Height));
                crops.Add(crop);
            }

            var dets = _detr.PredictBatch(crops, _config.DetrConfidenceThreshold);
            for (int j = 0; j < batch.Count; j++)
            {
                bool ok = false;
                foreach (var p in dets[j])
                {
                    float ax1 = rects[j].Left + p[0];
                    float ay1 = rects[j].Top + p[1];
                    float ax2 = rects[j].Left + p[2];
                    float ay2 = rects[j].Top + p[3];
                    var tmp = new[] { ax1, ay1, ax2, ay2 };
                    if (IoU(batch[j], tmp) >= 0.5f) { ok = true; break; }
                }
                if (ok) accepted.Add(batch[j]);
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
}

