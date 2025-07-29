using System;
using System.IO;
namespace SignatureDetectionSdk;

public class EnsembleDetector : IDisposable
{
    private readonly SignatureDetector _detr;
    private readonly YoloV8Detector _yolo;
    private readonly bool _enabled;

    public EnsembleDetector(string detrPath, string yoloPath, bool enabled = true)
    {
        _enabled = enabled && File.Exists(yoloPath);
        _detr = new SignatureDetector(detrPath);
        _yolo = _enabled ? new YoloV8Detector(yoloPath) : null!;
    }

    public void Dispose()
    {
        _detr.Dispose();
        if (_enabled)
            _yolo.Dispose();
    }

    public float[][] Predict(string imagePath, float scoreThreshold = 0.1f)
    {
        var detrBoxes = _detr.Predict(imagePath, scoreThreshold);
        if (!_enabled)
            return detrBoxes;
        var yoloBoxes = _yolo.Predict(imagePath, 0.25f);
        var fused = PostProcessing.WeightedBoxFusion(detrBoxes, yoloBoxes, 0.5f);
        fused = PostProcessing.FilterByGeometry(fused, 800f, 400000f, 0.5f, 6f);
        fused = PostProcessing.SoftNmsDistance(fused, 0.5f, 150f);
        return fused.ToArray();
    }
}
