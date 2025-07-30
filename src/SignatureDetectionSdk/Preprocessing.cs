using SkiaSharp;

namespace SignatureDetectionSdk;

public static class ImagePreprocessing
{
    public static SKBitmap Apply(SKBitmap input)
    {
        var bmp = new SKBitmap(input.Info.Width, input.Info.Height);
        input.CopyTo(bmp);

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
        return bmp;
    }
}

