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
    private readonly float _sLow;
    private int _used;
    private int _total;
    private int _fusedAccepted;
    private int _fusedRejected;

    public int UsedCount => _used;
    public int TotalCount => _total;
    public int FusionAccepted => _fusedAccepted;
    public int FusionRejected => _fusedRejected;
    public bool LastUsedEnsemble { get; private set; }

    public EnsembleDetector(string detrPath, string yoloPath, bool enabled = true,
        int tLow = 2, float sLow = 0.5f)
    {
        _enabled = enabled && File.Exists(yoloPath);
        _detr = new SignatureDetector(detrPath);
        _yolo = _enabled ? new YoloV8Detector(yoloPath) : null!;
        _tLow = tLow;
        _sLow = sLow;
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

        if (!_enabled)
        {
            LastUsedEnsemble = false;
            return detrBoxes;
        }

        float avgScore = detrBoxes.Length == 0 ? 0f : detrBoxes.Average(b => b[4]);
        bool use = detrBoxes.Length < _tLow || avgScore < _sLow;
        LastUsedEnsemble = use;
        if (!use)
            return detrBoxes;

        _used++;
        var yoloBoxes = _yolo.Predict(imagePath, 0.25f);
        int acc, rej;
        var fused = PostProcessing.WeightedBoxFusion(detrBoxes, yoloBoxes, 0.45f, 0.35f, out acc, out rej);
        _fusedAccepted += acc;
        _fusedRejected += rej;
        fused = PostProcessing.FilterByGeometry(fused, 1200f, 150000f, 0.6f, 6.5f);
        fused = PostProcessing.SoftNmsDistance(fused, 0.6f, 180f);
        return fused.ToArray();
    }
}
