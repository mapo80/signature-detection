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
    int FallbackFn = 0
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
        if (_config.EnableYoloV8 && _config.EnableDetr)
        {
            if (_config.Strategy.StartsWith("Parallel", StringComparison.OrdinalIgnoreCase))
            {
                final.AddRange(yoloPreds);
                final.AddRange(detrPreds);
            }
            else // SequentialFallback
            {
                var primary = yoloPreds;
                var secondary = detrPreds;
                final.AddRange(primary);
                if (groundTruth != null && (CountFp(primary, groundTruth) > _config.FallbackFp || CountFn(primary, groundTruth) > _config.FallbackFn))
                {
                    final.AddRange(secondary);
                }
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

