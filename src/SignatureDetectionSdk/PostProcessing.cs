using System;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;

namespace SignatureDetectionSdk;

public readonly record struct QueryScore(int QueryIndex, int ClassIndex, float Score);

public static class PostProcessing
{
    public static IReadOnlyList<QueryScore> FilterByScore(Tensor<float> logits, float confThreshold)
    {
        var dims = logits.Dimensions;
        if (dims.Length != 3)
            throw new ArgumentException($"Expected logits with 3 dimensions, got {dims.Length}");
        int numQueries = dims[1];
        int numClasses = dims[2];
        var keep = new List<QueryScore>();

        if (numClasses == 1)
        {
            for (int q = 0; q < numQueries; q++)
            {
                float score = Sigmoid(logits[0, q, 0]);
                if (score >= confThreshold)
                    keep.Add(new QueryScore(q, 0, score));
            }
            return keep;
        }

        for (int q = 0; q < numQueries; q++)
        {
            float maxLogit = float.NegativeInfinity;
            for (int c = 0; c < numClasses - 1; c++)
            {
                float l = logits[0, q, c];
                if (l > maxLogit) maxLogit = l;
            }
            float sum = 0f;
            var probs = new float[numClasses - 1];
            for (int c = 0; c < numClasses - 1; c++)
            {
                float e = MathF.Exp(logits[0, q, c] - maxLogit);
                probs[c] = e;
                sum += e;
            }
            float bestProb = 0f;
            int bestClass = -1;
            for (int c = 0; c < numClasses - 1; c++)
            {
                float p = probs[c] / sum;
                if (p > bestProb)
                {
                    bestProb = p;
                    bestClass = c;
                }
            }
            if (bestProb >= confThreshold)
                keep.Add(new QueryScore(q, bestClass, bestProb));
        }
        return keep;
    }

    public static List<float[]> ToPixelBoxes(Tensor<float> boxes, int width, int height, IReadOnlyList<QueryScore> keep)
    {
        var dims = boxes.Dimensions;
        if (dims.Length != 3)
            throw new ArgumentException($"Expected boxes with 3 dimensions, got {dims.Length}");
        var result = new List<float[]>(keep.Count);
        foreach (var k in keep)
        {
            int q = k.QueryIndex;
            float cx = boxes[0, q, 0];
            float cy = boxes[0, q, 1];
            float w  = boxes[0, q, 2];
            float h  = boxes[0, q, 3];
            float x1 = (cx - w / 2f) * width;
            float y1 = (cy - h / 2f) * height;
            float x2 = (cx + w / 2f) * width;
            float y2 = (cy + h / 2f) * height;
            result.Add(new[]{x1,y1,x2,y2,k.Score});
        }
        return result;
    }

    public static List<float[]> Nms(IReadOnlyList<float[]> boxes, float iouThreshold)
    {
        var sorted = boxes.OrderByDescending(b => b[4]).ToList();
        var keep = new List<float[]>();
        while (sorted.Count > 0)
        {
            var box = sorted[0];
            sorted.RemoveAt(0);
            keep.Add(box);
            sorted.RemoveAll(b => IoU(box, b) > iouThreshold);
        }
        return keep;
    }

    public static List<float[]> FilterByGeometry(IReadOnlyList<float[]> boxes,
        float areaMin, float areaMax, float arMin, float arMax)
    {
        var keep = new List<float[]>();
        foreach (var b in boxes)
        {
            float w = b[2] - b[0];
            float h = b[3] - b[1];
            float area = w * h;
            if (area < areaMin || area > areaMax) continue;
            float ar = w / h;
            if (ar < arMin || ar > arMax) continue;
            keep.Add(b);
        }
        return keep;
    }

    public static List<float[]> SoftNmsDistance(List<float[]> boxes,
        float sigma, float distanceScale)
    {
        var work = boxes.OrderByDescending(b => b[4]).Select(b => (float[])b.Clone()).ToList();
        var keep = new List<float[]>();
        while (work.Count > 0)
        {
            work.Sort((a,b) => b[4].CompareTo(a[4]));
            var current = work[0];
            work.RemoveAt(0);
            keep.Add(current);
            for (int i = 0; i < work.Count; i++)
            {
                float iou = IoU(current, work[i]);
                float dist = CentroidDistance(current, work[i]);
                float decay = MathF.Exp(- (iou * iou) / sigma - (dist * dist) / (distanceScale * distanceScale));
                work[i][4] *= decay;
            }
        }
        return keep;
    }

    public static List<float[]> WeightedBoxFusion(IReadOnlyList<float[]> a,
        IReadOnlyList<float[]> b, float iouThreshold)
    {
        var all = a.Concat(b).OrderByDescending(x => x[4]).Select(x => (float[])x.Clone()).ToList();
        var result = new List<float[]>();
        while (all.Count > 0)
        {
            var current = all[0];
            all.RemoveAt(0);
            var cluster = new List<float[]> { current };
            for (int i = all.Count - 1; i >= 0; i--)
            {
                if (IoU(current, all[i]) > iouThreshold)
                {
                    cluster.Add(all[i]);
                    all.RemoveAt(i);
                }
            }

            float sumScore = cluster.Sum(c => c[4]);
            float x1 = cluster.Sum(c => c[0] * c[4]) / sumScore;
            float y1 = cluster.Sum(c => c[1] * c[4]) / sumScore;
            float x2 = cluster.Sum(c => c[2] * c[4]) / sumScore;
            float y2 = cluster.Sum(c => c[3] * c[4]) / sumScore;
            float score = cluster.Max(c => c[4]);
            result.Add(new[] { x1, y1, x2, y2, score });
        }
        return result;
    }

    private static float IoU(float[] a, float[] b)
    {
        float xx1 = MathF.Max(a[0], b[0]);
        float yy1 = MathF.Max(a[1], b[1]);
        float xx2 = MathF.Min(a[2], b[2]);
        float yy2 = MathF.Min(a[3], b[3]);
        float inter = MathF.Max(0, xx2 - xx1) * MathF.Max(0, yy2 - yy1);
        float areaA = MathF.Max(0, a[2] - a[0]) * MathF.Max(0, a[3] - a[1]);
        float areaB = MathF.Max(0, b[2] - b[0]) * MathF.Max(0, b[3] - b[1]);
        float union = areaA + areaB - inter;
        return union <= 0 ? 0 : inter / union;
    }

    private static float CentroidDistance(float[] a, float[] b)
    {
        float cx1 = (a[0] + a[2]) / 2f;
        float cy1 = (a[1] + a[3]) / 2f;
        float cx2 = (b[0] + b[2]) / 2f;
        float cy2 = (b[1] + b[3]) / 2f;
        float dx = cx1 - cx2;
        float dy = cy1 - cy2;
        return MathF.Sqrt(dx * dx + dy * dy);
    }

    private static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));
}
