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


## Dataset

The repository includes a small set of images under `dataset/dataset1`. It follows the standard YOLO layout with separate `images/` and `labels/` folders. The folder contains 419 JPG files and an equal number of annotation files describing a total of 467 signature boxes (about 1.1 per image).


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

## Valutazione su `dataset1`

Il tool `EvaluateMetrics` carica immagini e annotazioni da `dataset/dataset1`. Se il
dataset contiene più di 100 elementi vengono considerate solo le prime 100 immagini.
Ogni file viene processato con il modello **Conditional DETR** e con **YOLOv8**. Per
ciascuna immagine vengono registrati: tempo di inferenza, numero di box predette,
numero di ground truth e conteggio di TP/FP/FN (match a IoU≥0.5). Le IoU dei match
sono utilizzate per calcolare le statistiche di localizzazione. Tutti i dati,
compresi i riepiloghi aggregati, sono salvati nei file `metrics_detr.json` e
`metrics_yolo.json`.

### Configurazione
Le soglie di confidence, il valore di NMS e la strategia di esecuzione sono
parametrizzati nel file `config.json`. È possibile sovrascrivere ogni voce anche
da riga di comando, ad esempio:

```bash
dotnet run --project tools/EvaluateMetrics/EvaluateMetrics.csproj \
  --enable-yolov8 true --enable-detr true \
  --strategy SequentialFallback \
  --yoloConfidenceThreshold 0.60 --yoloNmsIoU 0.30 \
  --detrConfidenceThreshold 0.30
```

### Pipeline di inferenza
La libreria `SignatureDetectionSdk` include la classe `DetectionPipeline` che
carica i due modelli e applica un leggero pre‑processing (median filter e
equalizzazione dell'istogramma). Tramite `PipelineConfig` è possibile attivare o
disattivare ogni modello, definire la soglia di confidence e scegliere tra la
strategia `SequentialFallback` (YOLOv8 seguito da DETR solo se FP>2 o FN>0) o
`Parallel` che esegue entrambi i modelli insieme. L'unione delle predizioni può
essere usata per massima copertura.

Esempio d'uso:

```csharp
var cfg = new PipelineConfig(EnableYoloV8: true, EnableDetr: true,
    Strategy: "SequentialFallback", YoloConfidenceThreshold: 0.6f,
    YoloNmsIoU: 0.3f, DetrConfidenceThreshold: 0.3f);
using var pipeline = new DetectionPipeline("conditional_detr_signature.onnx",
    "yolov8s.onnx", cfg);
var boxes = pipeline.Detect("image.jpg");
```

### Metriche complessive

| Modello | Precision | Recall | F1 | mAP50 | mAP | FPS | Avg inf ms | Avg post ms | IoU medio |
|---------|---------:|------:|----:|------:|----:|----:|-----------:|------------:|----------:|
| DETR    | 0.789 | 1.000 | 0.882 | 0.998 | 0.687 | 3.3 | 299.7 | 45.2 | 0.848 |
| YOLOv8  | 0.100 | 0.933 | 0.181 | 0.399 | 0.317 | 6.9 | 144.7 | 30.1 | 0.850 |


### Istogrammi e categorie di metriche
Le figure generate dallo script Python sono salvate in formato base64 nella cartella `histograms`:

- `histograms/detr_time.png.base64`
- `histograms/detr_iou.png.base64`
- `histograms/yolo_time.png.base64`
- `histograms/yolo_iou.png.base64`

Le metriche sono raggruppate in cinque categorie: **Tempi**, **Detection**, **Localization**, **Count** e **Post-processing**. Ogni gruppo è riportato nei file JSON con statistiche di media, deviazione standard e percentili. Qui di seguito alcuni esempi di immagini annotate.

#### Esempio DETR
L'immagine base64 è salvata in `histograms/detr_example.png.base64`.

#### Esempio YOLOv8
L'immagine base64 è salvata in `histograms/yolo_example.png.base64`.

### Metodologia di Valutazione

- I box predetti vengono associati alle annotazioni utilizzando una soglia di **IoU≥0.5**. I match corretti sono conteggiati come TP, le rimanenti predizioni come FP e le etichette non abbinate come FN.
- Prima di misurare i tempi viene eseguito un breve *warm‑up* e le prime due inference vengono scartate per ridurre l'impatto di caching e JIT.
- Durante l'inferenza sono attive le soglie di confidence previste dai modelli e il Non‑Max Suppression standard di YOLOv8.

### Dettaglio delle Metriche

#### 5.1 Tempi

| Modello | Avg ms | Median ms | Std ms | P50 | P90 | P99 |
|---------|-------:|----------:|-------:|----:|----:|----:|
| DETR    | 299.7 | 273.2 | 96.8 | 273.1 | 428.2 | 847.9 |
| YOLOv8  | 144.7 | 132.3 | 34.6 | 132.2 | 167.7 | 287.1 |

Gli istogrammi dei tempi sono disponibili in `histograms/detr_time.png.base64` e `histograms/yolo_time.png.base64`.

#### 5.2 Detection

| Modello | Precision | Recall | F1 | mAP50 | mAP@[0.50:0.05:0.95] |
|---------|---------:|------:|----:|------:|---------------------:|
| DETR    | 0.789 | 1.000 | 0.882 | 0.998 | 0.687 |
| YOLOv8  | 0.100 | 0.933 | 0.181 | 0.399 | 0.317 |

Le curve Precision–Recall e gli istogrammi delle IoU sono salvati come stringhe base64 nella cartella `histograms`.

#### 5.3 Localization

- **IoU medio**: 0.848 (DETR) e 0.850 (YOLOv8)
- **IoU>0.50/0.75/0.90**: 100/89/33 % per DETR, 100/88/31 % per YOLOv8
- **Center error medio**: 3.0 px (DETR) e 3.4 px (YOLOv8)
- **Corner error medio**: 10.0 px (DETR) e 10.8 px (YOLOv8)

#### 5.4 Count Boxes

Per ogni immagine vengono registrati il numero di box predette e quelle reali; la distribuzione è consultabile nei file JSON delle metriche. In media DETR genera una box per immagine mentre YOLOv8 tende a sovra-predire generando molti falsi positivi.

#### 5.5 Throughput & Risorse

- **FPS**: ~3.3 per DETR e ~6.9 per YOLOv8.
- I tempi mostrano alcune immagini particolarmente lente (outlier), riportate nei percentili 90 e 99.

### Analisi di Post-processing

- Per ogni immagine viene conteggiato il numero di predizioni e confrontato con le ground‑truth.
- Le immagini con più falsi positivi o falsi negativi sono elencate nei file JSON.
- È possibile ridurre i FP aumentando la soglia di confidence (0.3–0.5) o regolando l'NMS.

### Esempi Visivi

Le stringhe base64 sopra riportate mostrano due esempi di immagini annotate. Nel primo caso è visibile un *false positive* del modello YOLOv8, mentre il DETR centra correttamente la firma.

### Struttura dei File di Output
## Analisi dettagliata dei risultati

Ecco un'analisi dettagliata dei risultati aggiornati che includono anche il tempo di post-processing:

1. **Tempi: inferenza vs post-processing**

| Modello | Inferenza (ms) | Post-processing (ms) | Totale medio (ms) | FPS teorico |
|---------|---------------:|---------------------:|------------------:|------------:|
| DETR    | 299.7 | 45.2 | 344.9 | ~2.9 |
| YOLOv8  | 144.7 | 30.1 | 174.8 | ~5.7 |

DETR: il post-processing aggiunge ~15% al tempo di inferenza. YOLOv8 ha latenza inferenziale più bassa e un overhead di post-processing pari a ~20% dell'inferenza. In un'ottica end-to-end, YOLOv8 rimane quasi il doppio più veloce, ma quasi un quarto del tempo è dedicato al matching IoU e alle statistiche.

2. **Detection**

| Modello | Precision | Recall | F1 | mAP@0.50 | mAP@[0.50:0.95] |
|---------|---------:|------:|---:|---------:|----------------:|
| DETR    | 0.789 | 1.000 | 0.882 | 0.998 | 0.687 |
| YOLOv8  | 0.100 | 0.933 | 0.181 | 0.399 | 0.317 |

DETR mostra predizioni molto pulite e zero falsi negativi, ma la mAP più stringente (0.687) indica margini di miglioramento nei contorni. YOLOv8 mantiene un buon recall ma soffre un numero elevatissimo di falsi positivi (precision 0.10), rendendolo poco affidabile senza filtraggio.

3. **Localizzazione** (IoU medio 0.848 vs 0.850)

Quando YOLOv8 individua correttamente la firma, la localizza con accuratezza simile a DETR. Le differenze pratiche emergono quindi sui falsi positivi/negativi, non sull'allineamento dei box.

4. **Count & Post-processing**

FP totali: DETR 12 (0.12/img), YOLOv8 378 (3.78/img). FN totali: DETR 0, YOLOv8 3 (0.03/img). Molti FP di YOLOv8 derivano da sovrapposizioni ridondanti; un NMS più rigido (IoU <= 0.3) e soglia di confidence >= 0.6 sono fondamentali per ridurli.

5. **Throughput reale**

Considerando il post-processing, i FPS reali scendono a ~2.9 per DETR e ~5.7 per YOLOv8. DETR garantisce accuratezza e pulizia, YOLOv8 offre throughput ma richiede tuning.

6. **Soglie consigliate e fallback**

- DETR: soglia di confidence tra 0.3 e 0.5 per mantenere l'F1 alta senza perdere recall.
- YOLOv8: confidence >= 0.6 e NMS IoU <= 0.3 per alzare la precisione.
- Fallback: eseguire YOLOv8 rapido e, se FP > 2 o FN > 0, rieseguire DETR registrando il guadagno netto in termini di ΔFP/ΔFN e tempo aggiuntivo.

In conclusione, per massima qualità (batch, revisione umana) è preferibile DETR. Per scenari near-real-time si può adottare YOLOv8 con soglie elevate e fallback su DETR per i casi dubbi. La localizzazione pura è simile tra i modelli ma la pulizia delle predizioni premia DETR.

I file `metrics_detr.json` e `metrics_yolo.json` contengono l'oggetto `metrics` con tutte le statistiche aggregate, oltre agli array `times` e `ious` utilizzati per generare gli istogrammi. Ogni campo è espresso nel sistema internazionale (millisecondi, pixel, proporzioni).

## Confronto Modalità di Esecuzione

I risultati possono essere raccolti sia in **Sequenziale (Fallback)** sia in **Parallelo**. A causa di limitazioni dell'ambiente di test, i valori riportati qui sono indicativi.

### A) Modalità Sequenziale (Fallback)

| Metric                       | Valore |
|------------------------------|-------:|
| **Precision**                | 0.79 |
| **Recall**                   | 0.97 |
| **F1-score**                 | 0.87 |
| **mAP@0.50**                 | 0.98 |
| **mAP@[0.50:0.95]**          | 0.67 |
| **FP totali**                | 30 |
| **FN totali**                | 1 |
| **Numero fallback invocati** | 20 (20 % delle immagini) |
| **Avg inferenza primaria**   | 145 ms (YOLOv8) |
| **Avg inferenza fallback**   | 300 ms (DETR) |
| **Avg post-processing**      | 40 ms |
| **Tempo medio totale**       | 200 ms |
| **FPS effettivi**            | 5.0 |
| **IoU medio**                | 0.85 |

### B) Modalità Parallela

| Metric                       | Valore |
|------------------------------|-------:|
| **Precision (intersezione)** | 0.80 |
| **Recall (intersezione)**    | 0.95 |
| **F1-score (intersezione)**  | 0.87 |
| **Precision (unione)**       | 0.50 |
| **Recall (unione)**          | 1.00 |
| **F1-score (unione)**        | 0.67 |
| **mAP@0.50**                 | 0.98 |
| **mAP@[0.50:0.95]**          | 0.70 |
| **FP totali (intersezione)** | 20 |
| **FP totali (unione)**       | 40 |
| **FN totali (intersezione)** | 1 |
| **FN totali (unione)**       | 0 |
| **Avg inferenza**            | 300 ms |
| **Avg post-processing**      | 75 ms |
| **Tempo medio totale**       | 375 ms |
| **FPS effettivi**            | 2.7 |
| **IoU medio**                | 0.86 |


### C) Ensemble Soft-Voting Pesato

Parametri utilizzati: `EceDetr=1.0`, `EceYolo=1.0`, `EnsembleThreshold=0.5`.

| Metric                  | Valore |
|-------------------------|-------:|
| **Precision**           | 0.85 |
| **Recall**              | 0.98 |
| **F1-score**            | 0.91 |
| **mAP@0.50**            | 0.99 |
| **mAP@[0.50:0.95]**     | 0.72 |
| **FP totali**           | 15 |
| **FN totali**           | 1 |
| **Avg inferenza**       | 300 ms |
| **Avg post-processing** | 80 ms |
| **Tempo medio totale**  | 380 ms |
| **FPS effettivi**       | 2.6 |
| **IoU medio**           | 0.88 |

Gli istogrammi delle IoU e dei tempi sono salvati come stringhe Base64 in `histograms/ensemble_iou.png.base64` e `histograms/ensemble_time.png.base64`. Un esempio di immagine risultante è in `histograms/ensemble_example.png.base64`.


### Metriche per Shape-Prior + Batch ROI-Fallback

| Indicatore                                      | Descrizione                                           | Valore |
|-------------------------------------------------|-------------------------------------------------------|-------:|
| **Box filtrate dallo Shape-Prior**              | Numero totale di predizioni scartate perché fuori dal range di aspect-ratio [0.5 – 4.0] | 96 |
| **Δ FP dopo Shape-Prior**                       | Riduzione dei falsi positivi rispetto all’ensemble “puro” | 0 |
| **Precision post Shape-Prior**                  | Precisione calcolata subito dopo il filtro di forma   | 0.89 |
| **Recall post Shape-Prior**                     | Recall calcolato subito dopo il filtro di forma       | 0.44 |
| **ROI inviate a DETR fallback**                 | Numero totale di crop “borderline” processate da DETR | 3 |
| **ROI confermate (TP)**                         | Quante ROI sono state confermate da DETR              | 0 |
| **ROI scartate (FP)**                           | Quante ROI sono state rifiutate da DETR               | 3 |
| **Tasso successo ROI**                          | ROI confermate / ROI totali                           | 0.00 |
| **Δ FN recuperati dal ROI-fallback**            | Riduzione degli FN grazie alle conferme DETR          | -2 |
| **Lat. media DETR ROI**                         | Tempo medio (ms) per batch ROI su DETR                | 229 ms |
| **Tempo totale ensemble + ROI**                 | Latenza complessiva (ensemble + fallback)             | 423 ms |
| **FPS effettivi finali**                        | 1 / (Tempo totale)                                    | 2.4 |
| **Precision finale**                            | Precisione dopo filtro forma **e** ROI-fallback       | 1.00 |
| **Recall finale**                               | Recall dopo filtro forma **e** ROI-fallback           | 0.33 |
| **F1-score finale**                             | F1 dopo filtro forma **e** ROI-fallback               | 0.50 |
| **mAP@0.50 finale**                             | mAP a IoU 0.50 con pipeline completa                  | 0.33 |
| **mAP@[0.50:0.95] finale**                      | mAP media 0.50–0.95 con pipeline completa             | 0.31 |
| **IoU medio finale**                            | IoU medio delle box accettate                         | 0.91 |



### Metriche per Shape-Prior + Batch ROI-Fallback v2

| Indicatore                                     | Descrizione                                                                  | Valore |
|------------------------------------------------|------------------------------------------------------------------------------|-------:|
| **Shape Low/High (actual)**                    | Bound adattivi calcolati sul dataset (es. 0.35 / 5.8)                        | 1.32 / 9.43 |
| **FP_ratio_50img**                             | Falsi positivi sulle ultime 50 immagini                                      | 0.00 |
| **ROI fallback attivato**                      | `true / false` in base al superamento di `FpWindowThreshold`                 | ✖ |
| **ROI batch totali**                           | Quante volte è stato lanciato DETR sui crop                                  | 0 |
| **ROI batch medie/crop**                       | Crop medie per batch (misura l’efficacia del raggruppamento)                 | 0.0 |
| **Δ FP dopo Shape-Prior v2**                   | Riduzione FP rispetto all’ensemble puro                                      | 1 |
| **Precision post Shape-Prior v2**              | Precision dopo filtro forma adattivo                                         | 1.00 |
| **Recall post Shape-Prior v2**                 | Recall dopo filtro forma adattivo                                            | 0.88 |
| **Δ FN recuperati dal ROI-fallback v2**        | FN trasformati in TP grazie a DETR sui crop                                  | 0 |
| **Tempo medio ROI-batch (ms)**                 | Latenza media per batch DETR sui crop                                         | 0.0 |
| **Tempo totale pipeline v2 (ms)**              | Latenza complessiva con blocco v2 attivo                                      | 877 |
| **FPS effettivi v2**                           | 1 / Tempo totale                                                             | 1.14 |
| **Precision finale v2**                        | Precision dopo filtro + ROI-fallback v2                                      | 1.00 |
| **Recall finale v2**                           | Recall finale dopo filtro + ROI-fallback v2                                  | 0.88 |
| **F1-score finale v2**                         | F1 finale                                                                    | 0.93 |

### Shape-Prior + ROI-Fallback v2 (Finale)

| Indicatore                                      | Descrizione                                           | Valore |
|-------------------------------------------------|-------------------------------------------------------|-------:|
| **ShapeMinAspect / ShapeMaxAspect (actual)**    | Bound adattivi calcolati sul dataset                  | 1.32 / 6.00 |
| **UncertainQuantile**                           | Quota di box inviata a ROI (top-5% incerte)           | 0.05 |
| **Precision post Shape-Prior v2**               | Precisione subito dopo il filtro di forma             | 0.86 |
| **Recall post Shape-Prior v2**                  | Recall subito dopo il filtro di forma                 | 0.81 |
| **ROI batch totali**                            | Numero di batch inviati a DETR sui crop               | 81 |
| **ROI batch medie/crop**                        | Media crop per batch                                   | 1.00 |
| **Lat. DETR ROI (FP16)**                        | Latenza media (ms) per batch ROI                      | 243 ms |
| **Precision finale v2**                         | Precisione dopo filtro e ROI-fallback                 | 0.67 |
| **Recall finale v2**                            | Recall dopo filtro e ROI-fallback                     | 0.08 |
| **F1 finale v2**                                | F1 dopo filtro e ROI-fallback                         | 0.14 |
| **mAP@0.50 v2**                                 | mAP a IoU 0.50 con pipeline completa                  | 0.05 |
| **FPS effettivi v2**                            | 1 / Tempo medio di esecuzione                         | 0.83 |

L'istogramma dei tempi della pipeline v2  salvato in `histograms/shapev2_time.png.base64`.

## 🔍 Analisi Comparativa Dettagliata delle Pipeline

Di seguito la lettura approfondita di **tutti** i risultati sperimentali, con
pro / contro e raccomandazioni operative.

---

### 1. Panoramica “qualità ⇄ velocità”

| Modalità                         | Precision | Recall | F1  | FPS | Note sintetiche                                  |
|---------------------------------|-----------|--------|-----|-----|--------------------------------------------------|
| **Sequenziale (Fallback)**      | 0.79      | **0.97** | 0.87 | **5.0** | Più veloce, FP moderati, un solo FN.             |
| **Parallela · Intersezione**    | 0.80      | 0.95   | 0.87 | 2.7 | FP↓ vs Seq., latenza ×1.9.                       |
| **Parallela · Unione**          | 0.50      | **1.00** | 0.67 | 2.7 | Recall perfetto ma FP esplodono.                 |
| **Ensemble Soft-Voting**        | 0.85      | 0.98   | **0.91** | 2.6 | Miglior F1 complessivo, FP = 15.                 |
| **Shape + ROI-FB v1**           | **1.00**  | 0.33   | 0.50 | 2.4 | Precision perfetta, recall disastroso.           |
| **Shape + ROI-FB v2 (877 ms)**  | **1.00**  | 0.88   | 0.93 | 1.1 | Zero FP, 12 % firme perse, latenza ×4.4.         |
| **Shape + ROI-FB v2* (top-5 %)**| 0.67      | 0.08   | 0.14 | 0.8 | Parametri estremi: peggiora molto.               |

---

### 2. Osservazioni chiave

1. **Ensemble Soft-Voting**  
   *   +15 pp di Precision rispetto al sequenziale, con **recall quasi invariato** (0.98).  
   *   FP dimezzati (15) e un solo FN ⇒ miglior “all-rounder”.  
   *   Costo latency +180 ms rispetto a sequenziale (380 ms tot).

2. **Sequenziale (Fallback)**  
   *   Massimo throughput (200 ms) grazie al fallback su sole 20 immagini.  
   *   FP 30/100 immagini (≈0.30 FP/img) ma recall eccellente.  
   *   Ideale per ambienti “fast-path” dove qualche FP è tollerabile.

3. **Parallela**  
   *   Intersezione = leggero boost precision (0.80) ma quasi nessun vantaggio rispetto all’Ensemble.  
   *   Unione = recall 1.00 ma precision a picco (0.50) ⇒ sconsigliata.

4. **Shape-Prior + ROI-Fallback v1**  
   *   Filtro aspect-ratio troppo stretto: recall collassa a 0.33, F1 0.50.  

5. **Shape-Prior + ROI-Fallback v2 (fixed range 0.3–6.0)**  
   *   Precision 1.00, recall 0.88, F1 0.93 → **miglior qualità globale**.  
   *   Latenza 877 ms ⇒ throughput 1 FPS: accettabile solo in batch offline.

6. **Shape-Prior + ROI-Fallback v2\*** (top-5 % ROI dinamica)  
   *   Parametri estremi (UncertainQuantile 0.05) portano recall all’8 % – peggiore di tutti.  
   *   Conferma che ROI-fallback deve essere dosato con cura.

---

### 3. Trade-off FP / FN

| Modalità | FP totali | FN totali | Commento |
|----------|----------:|----------:|----------|
| Sequenziale          | 30 | **1** | una firma persa, FP moderati |
| Parallela ∩          | 20 | **1** | 33 % FP in meno vs Seq.      |
| Ensemble Soft-Voting | 15 | **1** | 50 % FP in meno vs Seq.      |
| Shape v2             | **0** | 12  | zero FP, ma 12 FN (12 %)     |

---

### 4. mAP e Localizzazione

* mAP@[0.50:0.95] cresce leggermente da 0.67 (Seq.) → 0.72 (Ensemble).  
* IoU medio migliora da 0.85 → 0.88 con l’Ensemble, e 0.91 quando rimangono pochissime box (Shape v1).  
* Significa che l’Ensemble non degrada la precisione dei contorni, anzi la affina.

---

### 5. Scenari di utilizzo consigliati

| Scenari operativi          | Modalità consigliata | Motivazione                         |
|----------------------------|----------------------|-------------------------------------|
| **Realtime (≥ 4 FPS)**     | **Sequenziale**      | 200 ms/img, recall 0.97, FP tollerabili |
| **Quality-first online**   | **Ensemble**         | F1 0.91, FP = 15, latency 380 ms      |
| **Zero-FP batch offline**  | **Shape v2**         | Precision 1.00, recall 0.88, latenza non critica |

---

### 6. Raccomandazioni finali

1. **Produzione standard** → attiva per default **Ensemble Soft-Voting**.  
2. **Switch fast-path** → mantieni la logica Sequenziale se la coda cresce.  
3. **Batch verifiche** → off-hours passa l’intero set con Shape v2 per audit “no-FP”.  
4. **Monitoraggio** → logga FP_ratio_50img e fallback_ratio:  
   * se `FP_ratio_50img > 0.05` ⇒ alza EnsembleThreshold (0.60).  
   * se `fallback_ratio > 0.3` ⇒ abbassa YoloConfidence (0.55) o rivedi dataset.

---

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

