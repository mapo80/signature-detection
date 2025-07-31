using System;
using System.Collections.Generic;
using System.Linq;

namespace SignatureDetectionSdk;

public static class SoftVotingEnsemble
{
    public static float[][] Combine(IList<float[]> yolo, IList<float[]> detr,
        float eceYolo, float eceDetr, float threshold)
    {
        var all = new List<float[]>(yolo.Count + detr.Count);
        foreach (var p in yolo)
        {
            var c = (float[])p.Clone();
            c[4] *= eceYolo;
            all.Add(c);
        }
        foreach (var p in detr)
        {
            var c = (float[])p.Clone();
            c[4] *= eceDetr;
            all.Add(c);
        }

        var groups = new List<List<float[]>>();
        foreach (var det in all)
        {
            bool added = false;
            foreach (var g in groups)
            {
                if (g.Any(o => IoU(o, det) >= 0.5f))
                {
                    g.Add(det);
                    added = true;
                    break;
                }
            }
            if (!added)
                groups.Add(new List<float[]> { det });
        }

        var result = new List<float[]>();
        foreach (var g in groups)
        {
            float sum = g.Sum(d => d[4]);
            float x1 = g.Sum(d => d[0] * d[4]) / sum;
            float y1 = g.Sum(d => d[1] * d[4]) / sum;
            float x2 = g.Sum(d => d[2] * d[4]) / sum;
            float y2 = g.Sum(d => d[3] * d[4]) / sum;
            float score = sum / g.Count;
            if (score >= threshold)
                result.Add(new[] { x1, y1, x2, y2, score });
        }
        return result.ToArray();
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
