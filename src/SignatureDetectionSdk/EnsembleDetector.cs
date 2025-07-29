using System;
using System.IO;
using System.Linq;
namespace SignatureDetectionSdk;

public class EnsembleDetector : IDisposable
{
    private readonly SignatureDetector _detr;
    private readonly YoloV8Detector _yolo;
    private readonly bool _enabled;
    private readonly int _tLow;
    private readonly float _vHigh;
    private readonly int _topK;
    private readonly float _wbfIou;
    private readonly float _wbfScore;
    private int _used;
    private int _total;
    private int _fusedAccepted;
    private int _fusedRejected;

    public int UsedCount => _used;
    public int TotalCount => _total;
    public int FusionAccepted => _fusedAccepted;
    public int FusionRejected => _fusedRejected;
    public bool LastUsedEnsemble { get; private set; }
    public int LastRobustCount { get; private set; }
    public float LastScoreVariance { get; private set; }

    public EnsembleDetector(string detrPath, string yoloPath, bool enabled = true,
        int tLow = 1, float vHigh = 0.1f, int topK = 3,
        float wbfIou = 0.55f, float wbfScore = 0.45f)
    {
        _enabled = enabled && File.Exists(yoloPath);
        _detr = new SignatureDetector(detrPath);
        _yolo = _enabled ? new YoloV8Detector(yoloPath) : null!;
        _tLow = tLow;
        _vHigh = vHigh;
        _topK = topK;
        _wbfIou = wbfIou;
        _wbfScore = wbfScore;
    }

    public void Dispose()
    {
        _detr.Dispose();
        if (_enabled)
            _yolo.Dispose();
    }

    public float[][] Predict(string imagePath, float scoreThreshold = 0.1f)
    {
        _total++;
        var detrBoxes = _detr.Predict(imagePath, scoreThreshold);
        LastRobustCount = detrBoxes.Length;

        if (!_enabled)
        {
            LastUsedEnsemble = false;
            return detrBoxes;
        }

        float std = 0f;
        if (detrBoxes.Length > 0)
        {
            var top = detrBoxes.OrderByDescending(b => b[4]).Take(_topK).ToArray();
            float mean = top.Average(b => b[4]);
            std = top.Length > 1 ? top.Sum(b => (b[4] - mean) * (b[4] - mean)) / top.Length : 0f;
        }
        LastScoreVariance = std;
        bool use = detrBoxes.Length < _tLow && std < _vHigh;
        LastUsedEnsemble = use;
        if (!use)
            return detrBoxes;

        _used++;
        var yoloBoxes = _yolo.Predict(imagePath, 0.25f, SignatureDetector.EnsembleParams);
        int acc, rej;
        var fused = PostProcessing.WeightedBoxFusion(detrBoxes, yoloBoxes, _wbfIou, _wbfScore, out acc, out rej);
        _fusedAccepted += acc;
        _fusedRejected += rej;

        fused = PostProcessing.FilterByGeometry(fused,
            SignatureDetector.EnsembleParams.AreaMin,
            SignatureDetector.EnsembleParams.AreaMax,
            SignatureDetector.EnsembleParams.ArMin,
            SignatureDetector.EnsembleParams.ArMax);
        fused = PostProcessing.SoftNmsDistance(fused,
            SignatureDetector.EnsembleParams.Sigma,
            SignatureDetector.EnsembleParams.DistScale);

        float dynamicThresh = 0.3f;
        if (fused.Count > 0)
        {
            var ordered = fused.Select(b => b[4]).OrderBy(v => v).ToList();
            float median = ordered[ordered.Count / 2];
            dynamicThresh = SignatureDetector.EnsembleParams.Alpha * median;
        }
        var filtered = fused.Where(b => b[4] >= dynamicThresh).ToList();
        var nms = PostProcessing.Nms(fused, 0.5f);
        var finalList = filtered.Count < SignatureDetector.EnsembleParams.NMin ? nms : filtered;

        if (finalList.Count == 0)
        {
            if (use)
            {
                finalList = detrBoxes.ToList();
            }
            else if (_enabled)
            {
                var yoloOnly = _yolo.Predict(imagePath, 0.3f, SignatureDetector.EnsembleParams);
                finalList = PostProcessing.Nms(yoloOnly, 0.5f);
            }
        }

        return finalList.ToArray();
    }
}
