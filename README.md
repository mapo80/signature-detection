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

