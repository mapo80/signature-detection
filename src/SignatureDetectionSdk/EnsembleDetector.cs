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

    private readonly SignatureDetector.RobustParams _detrParams;
    private readonly SignatureDetector.RobustParams _ensParams;

    public EnsembleDetector(string detrPath, string yoloPath, bool enabled = true,
        int tLow = 1,
        float wbfIou = 0.55f, float wbfScore = 0.45f,
        SignatureDetector.RobustParams? detrParams = null,
        SignatureDetector.RobustParams? ensembleParams = null)
    {
        _enabled = enabled && File.Exists(yoloPath);
        _detr = new SignatureDetector(detrPath);
        _yolo = _enabled ? new YoloV8Detector(yoloPath) : null!;
        _tLow = tLow;
        _wbfIou = wbfIou;
        _wbfScore = wbfScore;
        _detrParams = detrParams ?? SignatureDetector.DetrParams;
        _ensParams = ensembleParams ?? SignatureDetector.EnsembleParams;
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
        var detrBoxes = _detr.Predict(imagePath, out int robustCount, scoreThreshold, _detrParams);
        LastRobustCount = robustCount;

        if (!_enabled)
        {
            LastUsedEnsemble = false;
            return detrBoxes;
        }

        if (robustCount == 0)
        {
            LastUsedEnsemble = false;
            return detrBoxes;
        }

        bool use = robustCount < _tLow;
        LastUsedEnsemble = use;
        if (!use)
            return detrBoxes;

        _used++;
        var yoloBoxes = _yolo.Predict(imagePath, 0.25f, _ensParams);
        int acc, rej;
        var fused = PostProcessing.WeightedBoxFusion(detrBoxes, yoloBoxes, _wbfIou, _wbfScore, out acc, out rej);
        _fusedAccepted += acc;
        _fusedRejected += rej;

        fused = PostProcessing.FilterByGeometry(fused,
            _ensParams.AreaMin,
            _ensParams.AreaMax,
            _ensParams.ArMin,
            _ensParams.ArMax);
        fused = PostProcessing.SoftNmsDistance(fused,
            _ensParams.Sigma,
            _ensParams.DistScale);

        float dynamicThresh = 0.3f;
        if (fused.Count > 0)
        {
            var scoresList = fused.Select(b => b[4]).ToList();
            float perc = PostProcessing.Percentile(scoresList, _ensParams.ScorePercentile * 100f);
            dynamicThresh = _ensParams.Alpha * perc;
        }
        var filtered = fused.Where(b => b[4] >= dynamicThresh).ToList();
        var nms = PostProcessing.Nms(fused, 0.5f);
        var finalList = filtered.Count < _ensParams.NMin ? nms : filtered;

        if (finalList.Count == 0)
            finalList = PostProcessing.Nms(detrBoxes, 0.5f);

        return finalList.ToArray();
    }
}
