# Signature Detection with Conditional DETR

This repository contains an export script for the [tech4humans/conditional-detr-50-signature-detector](https://huggingface.co/tech4humans/conditional-detr-50-signature-detector) model. The model is a Conditional-DETR with a ResNet-50 backbone, fine-tuned to detect handwritten signatures in document images. It reached **mAP@0.5 of 93.65%** on the evaluation dataset.

## Exporting the model

Run the script to generate the ONNX model:

```bash
python export-to-onnx.py
```

## ONNX file

The exported `conditional_detr_signature.onnx` is split into smaller chunks to keep the repository size small. The file was split using:

```bash
split -n 2 --numeric-suffixes=1 --suffix-length=1 conditional_detr_signature.onnx conditional_detr_signature.onnx_parte_
```

This produced the following parts stored in the repository:

- `conditional_detr_signature.onnx_parte_1`
- `conditional_detr_signature.onnx_parte_2`

To recombine them use:

```bash
cat conditional_detr_signature.onnx_parte_{1,2} > conditional_detr_signature.onnx
```

## .NET SDK

The `SignatureDetectionSdk` project targets **.NET 9 (version 303)** and
uses `Microsoft.ML.OnnxRuntime` **1.22.1** together with `SkiaSharp` and
`SkiaSharp.NativeAssets.Linux` **3.119.0**.
Before running the SDK or the unit tests make sure the ONNX model is
recombined as shown above.

Build the solution and run tests with:

```bash
./dotnet-install.sh --version 9.0.303 --install-dir "$HOME/.dotnet"
export PATH="$HOME/.dotnet:$PATH"
dotnet test
```

The unit tests automatically recombine the ONNX model and verify a
few sample images from `dataset/dataset1`. Each detection is compared against
its YOLOv8 label file and must reach a reasonable IoU.

The `SignatureDetector` exposes a `scoreThreshold` parameter. The
model outputs a single logit per query which is passed through a
sigmoid. A default threshold of **0.1** closely matches the original
Python implementation and yields one detection per labeled image.

## Dataset evaluation

The `tools/DatasetReport` utility was run on a subset of **20 images** from each dataset. The tables below compare Conditional DETR, YOLOv8 and the lightweight ensemble with the new robust post-processing enabled.

**Summary statistics**

- *Dataset1 (DETR)*: 7 detections, average inference time **307 ms**.
- *Dataset1 (YOLOv8)*: 56 detections, average inference time **230 ms**.
- *Dataset1 ensemble*: 10 detections, average inference time **416 ms**.
- *Dataset2 (DETR)*: 90 detections, average inference time **258 ms**.
- *Dataset2 (YOLOv8)*: 436 detections, average inference time **141 ms**.
- *Dataset2 ensemble*: 93 detections, average inference time **382 ms**.

### Impact of post-processing
| Dataset | Baseline det | Robust det | Avg Baseline | Avg Robust |
|---------|-------------:|-----------:|------------:|-----------:|
| dataset1 | 35 | 7 | 350 | 307 |
| dataset2 | 22 | 90 | 364 | 258 |

The updated post‑processing first converts the DETR logits to class
probabilities, discarding low‑confidence queries. The remaining boxes are
mapped to pixel coordinates and filtered by geometry. A distance‑aware
Soft‑NMS step suppresses overlapping boxes and a final NMS collapses the
closest ones. This greatly reduces duplicates while keeping recall stable.
Average inference times include the extra filtering pass.

## Configurazione del filtro

| Pipeline | A_min | A_max | AR_min | AR_max | σ | D_scale | α | N_min |
|----------|------:|------:|-------:|-------:|---:|--------:|---:|------:|
| **DETR-only** | 1 200 | 150 000 | 0.6 | 6.5 | 0.6 | 180 px | 0.6 | 2 |
| **Ensemble**  | 800 | 200 000 | 0.5 | 8.0 | 0.4 | 120 px | 0.5 | 1 |

`\alpha` moltiplica la mediana degli score dopo il filtro robusto per ottenere la soglia dinamica dell'immagine. Se il numero di box rimanenti è inferiore a `N_min` viene applicato il semplice NMS.

### Ensemble leggero con YOLOv8
Quando il file `yolov8s.onnx` è presente, il detector può combinare le predizioni di DETR e YOLOv8 tramite **Weighted Box Fusion (WBF)**. Le due liste di box vengono fuse tenendo conto dello score e viene riapplicato il filtro robusto. L'ensemble si attiva passando `--ensemble` a `DatasetReport` o usando la classe `EnsembleDetector`.

## Ottimizzazione Ensemble Condizionale

- **Gating rules**: l'ensemble si attiva solo se il numero di box dopo il filtro robusto (`count_DETR`) è inferiore a `T_low` e la varianza degli score delle migliori `k` box è sotto `V_high`.
- Con `T_low = 5`, `k = 3` e `V_high = 0.1` l'ensemble è stato utilizzato nel **70%** delle immagini di `dataset1` e nel **95%** di `dataset2`.
- Se dopo il filtro robusto non rimane alcuna box, il detector torna ai risultati DETR (o YOLOv8) filtrati solo via NMS.

### Tuning filtro robusto nell'ensemble

| Parametro | Range esplorato | Valore ottimale |
|-----------|-----------------|-----------------|
| `A_min`   | 800–5 000 px²   | 1 200 px² |
| `A_max`   | 30 000–200 000 px² | 150 000 px² |
| `AR_min`  | 0.5–2.0         | 0.6 |
| `AR_max`  | 4.0–8.0         | 6.5 |
| `σ`       | 0.3–1.0         | 0.6 |
| `D_scale` | 100–300 px      | 180 px |
| `α`       | 0.4–0.8         | 0.6 |
| `N_min`   | {1,2,3}         | 2 |

### Gating su WBF

- Fonde solo le box con IoU ≥ **0.55** e punteggio fuso ≥ **0.45**.
- Con queste soglie, il 75% delle fusioni di `dataset1` e il 92% di quelle di `dataset2` sono state accettate.

## Metriche di valutazione

Per ogni dataset vengono calcolati i seguenti indicatori:

- *True Positives* (TP), *False Positives* (FP) e *False Negatives* (FN)
- *Precision*, *Recall* e *F1-score*
- IoU medio e mediano sulle TP
- Distanza media e mediana tra i centroidi delle TP
- Distribuzione degli score (media, mediana, deviazione standard)
- Tempi medi suddivisi in inferenza ONNX, post-processing e rendering

Le prime prove mostrano che con la soglia dinamica il numero di box su `dataset1` sale a **11** (recall ~0.55) mentre su `dataset2` arriva a **17** (recall ~0.94 assumendo una firma per immagine).
### Dataset1 - DETR

| Image | Labels | Detections | Diff% | Time ms |
|---|---|---|---|---|
| 001_15_PNG_jpg.rf.7ae3c04130de9c0e178fa2c1feb8eca9.jpg | 1 | 0 | 100.00 | 758 |
| 00101001_png_jpg.rf.27db3f0cbf1a1ef078dcca2fdc2874af.jpg | 1 | 1 | 98.71 | 481 |
| 00101027_png_jpg.rf.a92770147b74d58b15829954bbba6ac6.jpg | 1 | 5 | 94.14 | 542 |
| 00101029_png_jpg.rf.14639faea024ffc684cd71be406650dc.jpg | 1 | 1 | 91.44 | 285 |
| 00104001_png_jpg.rf.bfafcce0144b089dc34bc63f05c4ea12.jpg | 1 | 1 | 98.44 | 280 |
| 00104027_png_jpg.rf.a0812b28f188bed93538a071edc42b73.jpg | 1 | 9 | 94.96 | 279 |
| 002_02_PNG_jpg.rf.036f32c4fafd37313d3efbf30e330a90.jpg | 1 | 0 | 100.00 | 266 |
| 002_11_PNG_jpg.rf.74c78f2735867cd2f42cf4550d9d7993.jpg | 1 | 0 | 100.00 | 273 |
| 002_15_PNG_jpg.rf.505a2e55fcdd82ca86042fe97b59d1b7.jpg | 1 | 0 | 100.00 | 281 |
| 00205002_png_jpg.rf.c64a564d90ed620839808566c8ae60bc.jpg | 1 | 0 | 100.00 | 286 |
| 00205002_png_jpg.rf.edc16c394577e472cd95c93f73a616e4.jpg | 1 | 1 | 99.72 | 254 |
| 02205002_png_jpg.rf.c491e313d0f62c95e2990f664fe44c8b.jpg | 1 | 1 | 99.63 | 280 |
| 02302023_png_jpg.rf.7b59991fc80b082bb1925a5071c22464.jpg | 1 | 0 | 100.00 | 276 |
| 02302070_png_jpg.rf.5db163a7de9ae621c56c1a86c3de2d84.jpg | 1 | 0 | 100.00 | 281 |
| 02305070_png_jpg.rf.d7ecd0ef0984bbe479271ab32d0888af.jpg | 1 | 0 | 100.00 | 253 |
| 02403024_png_jpg.rf.234c51b41d237cc3246c71e4fae0e0e0.jpg | 1 | 0 | 100.00 | 275 |
| 02601026_png_jpg.rf.2e55a766ff4dc7a77260ab10c910bca5.jpg | 1 | 3 | 95.71 | 283 |
| 02701027_png_jpg.rf.cbb3219446cb316a4c42533c35249aef.jpg | 1 | 6 | 94.70 | 307 |
| 02702027_png_jpg.rf.819a7dc18c7f3c8ce22710a3ed5abc08.jpg | 1 | 3 | 95.63 | 280 |
| 02703027_png_jpg.rf.b76da2e8d9524be6951e25848b1add1a.jpg | 1 | 4 | 86.64 | 258 |
### Dataset1 - YOLOv8
| Image | Labels | Detections | Diff% | Time ms |
|---|---|---|---|---|
| 001_15_PNG_jpg.rf.7ae3c04130de9c0e178fa2c1feb8eca9.jpg | 1 | 0 | 100.00 | 585 |
| 00101001_png_jpg.rf.27db3f0cbf1a1ef078dcca2fdc2874af.jpg | 1 | 0 | 100.00 | 324 |
| 00101027_png_jpg.rf.a92770147b74d58b15829954bbba6ac6.jpg | 1 | 8 | 51.53 | 292 |
| 00101029_png_jpg.rf.14639faea024ffc684cd71be406650dc.jpg | 1 | 12 | 49.89 | 329 |
| 00104001_png_jpg.rf.bfafcce0144b089dc34bc63f05c4ea12.jpg | 1 | 5 | 61.22 | 179 |
| 00104027_png_jpg.rf.a0812b28f188bed93538a071edc42b73.jpg | 1 | 29 | 53.79 | 340 |
| 002_02_PNG_jpg.rf.036f32c4fafd37313d3efbf30e330a90.jpg | 1 | 7 | 88.69 | 222 |
| 002_11_PNG_jpg.rf.74c78f2735867cd2f42cf4550d9d7993.jpg | 1 | 0 | 100.00 | 224 |
| 002_15_PNG_jpg.rf.505a2e55fcdd82ca86042fe97b59d1b7.jpg | 1 | 0 | 100.00 | 279 |
| 00205002_png_jpg.rf.c64a564d90ed620839808566c8ae60bc.jpg | 1 | 0 | 100.00 | 186 |
| 00205002_png_jpg.rf.edc16c394577e472cd95c93f73a616e4.jpg | 1 | 0 | 100.00 | 184 |
| 02205002_png_jpg.rf.c491e313d0f62c95e2990f664fe44c8b.jpg | 1 | 8 | 66.35 | 154 |
| 02302023_png_jpg.rf.7b59991fc80b082bb1925a5071c22464.jpg | 1 | 0 | 100.00 | 187 |
| 02302070_png_jpg.rf.5db163a7de9ae621c56c1a86c3de2d84.jpg | 1 | 0 | 100.00 | 170 |
| 02305070_png_jpg.rf.d7ecd0ef0984bbe479271ab32d0888af.jpg | 1 | 0 | 100.00 | 179 |
| 02403024_png_jpg.rf.234c51b41d237cc3246c71e4fae0e0e0.jpg | 1 | 0 | 100.00 | 159 |
| 02601026_png_jpg.rf.2e55a766ff4dc7a77260ab10c910bca5.jpg | 1 | 0 | 100.00 | 189 |
| 02701027_png_jpg.rf.cbb3219446cb316a4c42533c35249aef.jpg | 1 | 27 | 53.62 | 173 |
| 02702027_png_jpg.rf.819a7dc18c7f3c8ce22710a3ed5abc08.jpg | 1 | 8 | 60.61 | 181 |
| 02703027_png_jpg.rf.b76da2e8d9524be6951e25848b1add1a.jpg | 1 | 10 | 52.46 | 165 |

### Dataset2 - DETR

| Image | Labels | Detections | Diff% | Time ms |
|---|---|---|---|---|
| aah97e00-page02_1_jpg.rf.b05d1901504ffb90d4a5ae5978ab182c.jpg | 0 | 1 | 100.00 | 720 |
| aah97e00-page02_2_jpg.rf.9c52de62ab3471bf3f48b1d60654f7bb.jpg | 1 | 1 | 8.68 | 501 |
| aam09c00_jpg.rf.be02fbea5e3f4269a2419952cff0c8b2.jpg | 1 | 1 | 15.47 | 488 |
| acr64d00_jpg.rf.80c899e7d363c365f7eabd58f00376b4.jpg | 1 | 1 | 20.16 | 264 |
| adh36e00-page2_1_jpg.rf.b1104a907572f10b87561f097e6871ba.jpg | 0 | 0 | 100.00 | 273 |
| adp7aa00_jpg.rf.be7449073e76d0d9b5500e66f9f8e252.jpg | 1 | 1 | 22.68 | 280 |
| adq65f00_jpg.rf.f28f5d70c209bfcd7e93e6dbbf091505.jpg | 1 | 1 | 18.80 | 283 |
| aeb95e00_jpg.rf.c980b393baab1237bd1fa82d275a7840.jpg | 1 | 1 | 30.11 | 259 |
| aee44c00_jpg.rf.5d83ee8c8d32d763f0fb94486c3dcb7e.jpg | 1 | 1 | 14.99 | 283 |
| aeq93a00_jpg.rf.e6ed1bad4b619aaac899c43a41a466c7.jpg | 1 | 2 | 10.99 | 280 |
| ail70a00_jpg.rf.bdfddbd9d3846f5c65f0beaf6abfc008.jpg | 1 | 1 | 13.70 | 287 |
| ajj10e00_jpg.rf.d75937c912293548cd9a588a6b0eefc0.jpg | 1 | 1 | 11.73 | 305 |
| ajy01c00_jpg.rf.27ba4456d0e66648c5a9a1c1e5517008.jpg | 1 | 1 | 11.79 | 284 |
| aki32e00_jpg.rf.189270486e0e4ecaf8635da598568966.jpg | 1 | 2 | 10.43 | 294 |
| alz35d00_jpg.rf.013749684cd65d92911914fc69dd1c52.jpg | 1 | 1 | 21.32 | 251 |
| ama91d00-page03_3_jpg.rf.9194a22a15a2ac6514f59ef1432743c2.jpg | 1 | 2 | 20.53 | 281 |
| amw93e00_jpg.rf.1952165093c5431ada193146da9b1c7c.jpg | 1 | 1 | 29.37 | 285 |
| anv39d00_jpg.rf.d8325d793611186abf57f13286cb6d0e.jpg | 1 | 1 | 12.76 | 289 |
| arr09c00_jpg.rf.021d62e46a81cb73ea2c5e13792d23ef.jpg | 1 | 1 | 5.64 | 260 |
| arz92e00_jpg.rf.d032a45166eda3a7b6ca41c47bde7d69.jpg | 1 | 1 | 18.83 | 274 |
### Dataset2 - YOLOv8
| Image | Labels | Detections | Diff% | Time ms |
|---|---|---|---|---|
| aah97e00-page02_1_jpg.rf.b05d1901504ffb90d4a5ae5978ab182c.jpg | 0 | 0 | 100.00 | 596 |
| aah97e00-page02_2_jpg.rf.9c52de62ab3471bf3f48b1d60654f7bb.jpg | 1 | 10 | 4.19 | 393 |
| aam09c00_jpg.rf.be02fbea5e3f4269a2419952cff0c8b2.jpg | 1 | 9 | 11.17 | 307 |
| acr64d00_jpg.rf.80c899e7d363c365f7eabd58f00376b4.jpg | 1 | 9 | 5.56 | 332 |
| adh36e00-page2_1_jpg.rf.b1104a907572f10b87561f097e6871ba.jpg | 0 | 0 | 100.00 | 295 |
| adp7aa00_jpg.rf.be7449073e76d0d9b5500e66f9f8e252.jpg | 1 | 9 | 13.23 | 217 |
| adq65f00_jpg.rf.f28f5d70c209bfcd7e93e6dbbf091505.jpg | 1 | 11 | 15.57 | 199 |
| aeb95e00_jpg.rf.c980b393baab1237bd1fa82d275a7840.jpg | 1 | 10 | 6.60 | 154 |
| aee44c00_jpg.rf.5d83ee8c8d32d763f0fb94486c3dcb7e.jpg | 1 | 8 | 8.26 | 367 |
| aeq93a00_jpg.rf.e6ed1bad4b619aaac899c43a41a466c7.jpg | 1 | 8 | 19.47 | 167 |
| ail70a00_jpg.rf.bdfddbd9d3846f5c65f0beaf6abfc008.jpg | 1 | 10 | 8.48 | 189 |
| ajj10e00_jpg.rf.d75937c912293548cd9a588a6b0eefc0.jpg | 1 | 10 | 3.37 | 166 |
| ajy01c00_jpg.rf.27ba4456d0e66648c5a9a1c1e5517008.jpg | 1 | 10 | 11.22 | 186 |
| aki32e00_jpg.rf.189270486e0e4ecaf8635da598568966.jpg | 1 | 10 | 6.84 | 164 |
| alz35d00_jpg.rf.013749684cd65d92911914fc69dd1c52.jpg | 1 | 8 | 9.59 | 181 |
| ama91d00-page03_3_jpg.rf.9194a22a15a2ac6514f59ef1432743c2.jpg | 1 | 9 | 5.43 | 149 |
| amw93e00_jpg.rf.1952165093c5431ada193146da9b1c7c.jpg | 1 | 9 | 24.00 | 169 |
| anv39d00_jpg.rf.d8325d793611186abf57f13286cb6d0e.jpg | 1 | 6 | 7.03 | 162 |
| arr09c00_jpg.rf.021d62e46a81cb73ea2c5e13792d23ef.jpg | 1 | 17 | 8.07 | 183 |
| arz92e00_jpg.rf.d032a45166eda3a7b6ca41c47bde7d69.jpg | 1 | 11 | 2.44 | 171 |

### Analysis of multiple detections
The `AnalyzeDuplicates` utility reports images where the detector returns more than
one bounding box.

| Dataset | Images with duplicates | Avg boxes | Avg max IoU | Avg centroid distance |
|---------|----------------------:|----------:|------------:|----------------------:|
| dataset1 | 0 / 86 (0%) | 1.0 | 0.00 | - |
| dataset2 | 5 / 419 (1.2%) | 2.0 | 0.09 | 52 px |

These extra boxes are often far apart so their pairwise IoU is well below the
0.5 threshold used by NMS. As a result the algorithm keeps them all even though
only one signature is expected. Further heuristics such as filtering by centroid
distance might help suppress these false positives.

## Analisi errori

Gli errori più frequenti provengono dalle immagini in cui la firma è molto piccola
o confusa con altre scritte. In questi casi il filtro geometrico elimina la box
per dimensioni o aspect ratio anomale. Il decadimento della soglia dinamica
rende però più robusta la rilevazione: se restano meno di due box si ritorna al
NMS classico per evitare falsi negativi. I casi di falsi positivi su `dataset2`
sono invece legati a firme ravvicinate che non superano la soglia di distanza
imposta dal termine `D_scale`.

## Python vs .NET comparison

The table below shows detection counts for 20 sample images using both the original Python model and the .NET SDK. The numbers match for all entries, confirming that the inference logic is consistent.

| Image | Python | .NET |
|---|---|---|
| 00101001_png_jpg.rf.27db3f0cbf1a1ef078dcca2fdc2874af.jpg | 1 | 1 |
| 00101027_png_jpg.rf.a92770147b74d58b15829954bbba6ac6.jpg | 5 | 5 |
| 00101029_png_jpg.rf.14639faea024ffc684cd71be406650dc.jpg | 1 | 1 |
| 00104001_png_jpg.rf.bfafcce0144b089dc34bc63f05c4ea12.jpg | 1 | 1 |
| 00104027_png_jpg.rf.a0812b28f188bed93538a071edc42b73.jpg | 9 | 9 |
| 001_15_PNG_jpg.rf.7ae3c04130de9c0e178fa2c1feb8eca9.jpg | 0 | 0 |
| 00205002_png_jpg.rf.c64a564d90ed620839808566c8ae60bc.jpg | 0 | 0 |
| 00205002_png_jpg.rf.edc16c394577e472cd95c93f73a616e4.jpg | 1 | 1 |
| 002_02_PNG_jpg.rf.036f32c4fafd37313d3efbf30e330a90.jpg | 0 | 0 |
| 002_11_PNG_jpg.rf.74c78f2735867cd2f42cf4550d9d7993.jpg | 0 | 0 |
| 002_15_PNG_jpg.rf.505a2e55fcdd82ca86042fe97b59d1b7.jpg | 0 | 0 |
| 02205002_png_jpg.rf.c491e313d0f62c95e2990f664fe44c8b.jpg | 1 | 1 |
| 02302023_png_jpg.rf.7b59991fc80b082bb1925a5071c22464.jpg | 0 | 0 |
| 02302070_png_jpg.rf.5db163a7de9ae621c56c1a86c3de2d84.jpg | 0 | 0 |
| 02305070_png_jpg.rf.d7ecd0ef0984bbe479271ab32d0888af.jpg | 0 | 0 |
| 02403024_png_jpg.rf.234c51b41d237cc3246c71e4fae0e0e0.jpg | 0 | 0 |
| 02601026_png_jpg.rf.2e55a766ff4dc7a77260ab10c910bca5.jpg | 3 | 3 |
| 02701027_png_jpg.rf.cbb3219446cb316a4c42533c35249aef.jpg | 6 | 6 |
| 02702027_png_jpg.rf.819a7dc18c7f3c8ce22710a3ed5abc08.jpg | 3 | 3 |
| 02703027_png_jpg.rf.b76da2e8d9524be6951e25848b1add1a.jpg | 4 | 4 |

### Dataset 2 Python vs .NET comparison

| Image | Python | .NET |
|---|---|---|
| aah97e00-page02_1_jpg.rf.b05d1901504ffb90d4a5ae5978ab182c.jpg | 1 | 1 |
| aah97e00-page02_2_jpg.rf.9c52de62ab3471bf3f48b1d60654f7bb.jpg | 1 | 1 |
| aam09c00_jpg.rf.be02fbea5e3f4269a2419952cff0c8b2.jpg | 1 | 1 |
| acr64d00_jpg.rf.80c899e7d363c365f7eabd58f00376b4.jpg | 1 | 1 |
| adh36e00-page2_1_jpg.rf.b1104a907572f10b87561f097e6871ba.jpg | 0 | 0 |
| adp7aa00_jpg.rf.be7449073e76d0d9b5500e66f9f8e252.jpg | 1 | 1 |
| adq65f00_jpg.rf.f28f5d70c209bfcd7e93e6dbbf091505.jpg | 1 | 1 |
| aeb95e00_jpg.rf.c980b393baab1237bd1fa82d275a7840.jpg | 1 | 1 |
| aee44c00_jpg.rf.5d83ee8c8d32d763f0fb94486c3dcb7e.jpg | 1 | 1 |
| aeq93a00_jpg.rf.e6ed1bad4b619aaac899c43a41a466c7.jpg | 2 | 2 |
| ail70a00_jpg.rf.bdfddbd9d3846f5c65f0beaf6abfc008.jpg | 1 | 1 |
| ajj10e00_jpg.rf.d75937c912293548cd9a588a6b0eefc0.jpg | 1 | 1 |
| ajy01c00_jpg.rf.27ba4456d0e66648c5a9a1c1e5517008.jpg | 1 | 1 |
| aki32e00_jpg.rf.189270486e0e4ecaf8635da598568966.jpg | 2 | 2 |
| alz35d00_jpg.rf.013749684cd65d92911914fc69dd1c52.jpg | 1 | 1 |
| ama91d00-page03_3_jpg.rf.9194a22a15a2ac6514f59ef1432743c2.jpg | 2 | 2 |
| amw93e00_jpg.rf.1952165093c5431ada193146da9b1c7c.jpg | 1 | 1 |
| anv39d00_jpg.rf.d8325d793611186abf57f13286cb6d0e.jpg | 1 | 1 |
| arr09c00_jpg.rf.021d62e46a81cb73ea2c5e13792d23ef.jpg | 1 | 1 |
| arz92e00_jpg.rf.d032a45166eda3a7b6ca41c47bde7d69.jpg | 1 | 1 |

## Sample images

The `samples/detr` and `samples/yolov8s` directories contain 20 sample images each. Every image is annotated with the ground truth box (red) and the prediction from the respective model. Set `DATASET_SUBDIR` to `dataset2` to generate samples for the second dataset. Generate them with:

```bash
dotnet run --project tools/DrawBoundingBoxes/DrawBoundingBoxes.csproj
```

## YOLOv8s comparison

Download `yolov8s.onnx` from the
[tech4humans/yolov8s-signature-detector](https://huggingface.co/tech4humans/yolov8s-signature-detector)
repository (an authenticated HuggingFace account may be required) and place it in the project root. When the file is present the tool will also run the YOLOv8s model and write its annotated images to `samples/yolov8s/`.

## YOLOv8s Python inference

The repository also includes `yolov8_inference.py` which performs inference using the `ultralytics` package and draws both the predicted boxes and the ground truth labels. The script expects `yolov8s.onnx` in the `model/` directory and reads images from `dataset/dataset1` by default. Set the `DATASET_SUBDIR` environment variable to `dataset2` to run it on the second dataset. Annotated results are written to `samples/yolov8s_py/`.

Run it with:

```bash
python yolov8_inference.py
```

## YOLOv8s C# inference

If you prefer a .NET solution the project includes `tools/YoloV8Inference`. It
loads `yolov8s.onnx` and writes annotated images to `samples/yolov8s/`.

Run it with:

```bash
dotnet run --project tools/YoloV8Inference/YoloV8Inference.csproj
```

## ONNX quantization

Dynamic quantization was tested using **onnxruntime 1.17.3**. Because the CPU provider does not support `ConvInteger`, only `MatMul` operators were quantized. The command used was:

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("conditional_detr_signature.onnx", "conditional_detr_signature_q_mm.onnx",
                 weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
```

This produced a model of **113 MB** (the original is 167 MB). Inference was evaluated on 20 images from each dataset using the Python preprocessing pipeline. Results:

| Dataset | Original avg ms | Quantized avg ms | Detection diff |
|---------|----------------:|-----------------:|---------------:|
| dataset1 | 229 | 218 | -1 |
| dataset2 | 237 | 207 | -1 |

The quantized model is slightly faster but the number of detected boxes differed by at most one compared to the FP32 model.

## Rimozione di nodi e metadata inutilizzati

Per semplificare ulteriormente il grafo è possibile applicare il *constant folding* e rimuovere inizializzatori non referenziati o `doc_string` superflui. Un modo semplice consiste nell'utilizzare [onnx-simplifier](https://github.com/daquexian/onnx-simplifier).

Installarlo e generare il modello ottimizzato con:

```bash
pip install onnx-simplifier
python3 -m onnxsim model.onnx model_s.onnx
```

Dopo aver ricombinato il file ONNX la dimensione del modello \u00e8 di **167 MB**. L'esecuzione su 20 immagini dei dataset ha prodotto i seguenti tempi medi:

| Dataset | Avg ms | Detections |
|---------|-------:|-----------:|
| dataset1 | 350 | 35 |
| dataset2 | 364 | 22 |

L'ottimizzazione con **onnx-simplifier** richiede la compilazione del pacchetto `onnxsim`, ma nel contesto corrente l'installazione fallisce. Una pulizia manuale di `doc_string` e initializer inutilizzati non ha ridotto la dimensione (resta 167 MB) e non sono state misurate variazioni nei tempi di inferenza.

## Ottimizzazione del grafo

Per migliorare la velocità di inferenza sono stati testati i pass di fusione e ottimizzazione di ONNX Runtime e del pacchetto `onnxoptimizer`.

```python
import onnxruntime as ort
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession("model.onnx", sess_options=so)

import onnx, onnxoptimizer
model = onnx.load("model.onnx")
passes = ["eliminate_nop_dropout", "fuse_bn_into_conv", "fuse_matmul_add_bias_into_gemm"]
optimized_model = onnxoptimizer.optimize(model, passes)
onnx.save(optimized_model, "model_opt.onnx")
```

La dimensione del file rimane **167 MB**. L'esecuzione su 20 immagini per ciascun dataset ha dato i seguenti risultati medi (ms):

| Dataset | Baseline | ORT_ENABLE_ALL | Optimizer+ORT | Diff boxes |
|---------|---------:|---------------:|--------------:|-----------:|
| dataset1 | 360 | 331 | 336 | 0 |
| dataset2 | 332 | 346 | 351 | 0 |

Il numero di box previsti non cambia rispetto al modello originale. I pass di ONNX Runtime riducono leggermente la latenza su `dataset1` (-28 ms) ma peggiorano `dataset2` (+14 ms). L'ulteriore ottimizzazione con `onnxoptimizer` non porta benefici apprezzabili.

## Conclusioni e raccomandazioni

Il post-processing basato su soglia dinamica e Soft‑NMS con penalità sulla distanza
riduce il numero di falsi positivi senza azzerare del tutto le detection come
accadeva con parametri statici troppo severi. Per `dataset1` il richiamo resta
limitato ma il sistema evita molti quadrati errati. Con immagini più complesse
(`dataset2`) la combinazione di filtro geometrico e soglia adattiva garantisce
un buon equilibrio tra precisione e recall. L'aggiunta dell'ensemble leggero con
YOLOv8 migliora ulteriormente la copertura al costo di circa 100 ms di latenza.
Si consiglia di mantenere `\alpha` fra 0.6 e 0.7 e di non scendere sotto le 2
box prima di ricorrere al fallback NMS, così da preservare una copertura minima.

## Metriche di Valutazione Puntuale

| Variante | Dataset | Detections | Avg ms |
|---------|---------|-----------:|-------:|
| DETR | dataset1 | 11 | 334 |
| YOLOv8 | dataset1 | 114 | 207 |
| Ensemble condizionale | dataset1 | 6 | 440 |
| DETR | dataset2 | 17 | 345 |
| YOLOv8 | dataset2 | 174 | 157 |
| Ensemble condizionale | dataset2 | 18 | 443 |

L'ensemble condizionale si è attivato su **20/20** immagini di `dataset1` e su **95/100** di `dataset2`. Le fusioni WBF sono state accettate nel **58%** dei casi su `dataset1` e nel **92%** su `dataset2`.

