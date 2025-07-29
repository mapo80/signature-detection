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

The `tools/DatasetReport` utility was run on 20 images from each dataset. The tables below show the results for Conditional DETR and YOLOv8.

### Dataset1 - DETR
| Image | Labels | Detections | Diff% | Time ms |
|---|---|---|---|---|
| 001_15_PNG_jpg.rf.7ae3c04130de9c0e178fa2c1feb8eca9.jpg | 1 | 0 | 100.00 | 763 |
| 00101001_png_jpg.rf.27db3f0cbf1a1ef078dcca2fdc2874af.jpg | 1 | 1 | 98.71 | 445 |
| 00101027_png_jpg.rf.a92770147b74d58b15829954bbba6ac6.jpg | 1 | 5 | 94.14 | 550 |
| 00101029_png_jpg.rf.14639faea024ffc684cd71be406650dc.jpg | 1 | 1 | 91.44 | 315 |
| 00104001_png_jpg.rf.bfafcce0144b089dc34bc63f05c4ea12.jpg | 1 | 1 | 98.44 | 337 |
| 00104027_png_jpg.rf.a0812b28f188bed93538a071edc42b73.jpg | 1 | 9 | 94.96 | 307 |
| 002_02_PNG_jpg.rf.036f32c4fafd37313d3efbf30e330a90.jpg | 1 | 0 | 100.00 | 315 |
| 002_11_PNG_jpg.rf.74c78f2735867cd2f42cf4550d9d7993.jpg | 1 | 0 | 100.00 | 294 |
| 002_15_PNG_jpg.rf.505a2e55fcdd82ca86042fe97b59d1b7.jpg | 1 | 0 | 100.00 | 314 |
| 00205002_png_jpg.rf.c64a564d90ed620839808566c8ae60bc.jpg | 1 | 0 | 100.00 | 310 |
| 00205002_png_jpg.rf.edc16c394577e472cd95c93f73a616e4.jpg | 1 | 1 | 99.72 | 295 |
| 02205002_png_jpg.rf.c491e313d0f62c95e2990f664fe44c8b.jpg | 1 | 1 | 99.63 | 291 |
| 02302023_png_jpg.rf.7b59991fc80b082bb1925a5071c22464.jpg | 1 | 0 | 100.00 | 310 |
| 02302070_png_jpg.rf.5db163a7de9ae621c56c1a86c3de2d84.jpg | 1 | 0 | 100.00 | 292 |
| 02305070_png_jpg.rf.d7ecd0ef0984bbe479271ab32d0888af.jpg | 1 | 0 | 100.00 | 303 |
| 02403024_png_jpg.rf.234c51b41d237cc3246c71e4fae0e0e0.jpg | 1 | 0 | 100.00 | 293 |
| 02601026_png_jpg.rf.2e55a766ff4dc7a77260ab10c910bca5.jpg | 1 | 3 | 95.71 | 323 |
| 02701027_png_jpg.rf.cbb3219446cb316a4c42533c35249aef.jpg | 1 | 6 | 94.70 | 320 |
| 02702027_png_jpg.rf.819a7dc18c7f3c8ce22710a3ed5abc08.jpg | 1 | 3 | 95.63 | 339 |
| 02703027_png_jpg.rf.b76da2e8d9524be6951e25848b1add1a.jpg | 1 | 4 | 86.64 | 291 |

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
| aah97e00-page02_1_jpg.rf.b05d1901504ffb90d4a5ae5978ab182c.jpg | 0 | 1 | 100.00 | 747 |
| aah97e00-page02_2_jpg.rf.9c52de62ab3471bf3f48b1d60654f7bb.jpg | 1 | 1 | 8.68 | 553 |
| aam09c00_jpg.rf.be02fbea5e3f4269a2419952cff0c8b2.jpg | 1 | 1 | 15.47 | 536 |
| acr64d00_jpg.rf.80c899e7d363c365f7eabd58f00376b4.jpg | 1 | 1 | 20.16 | 296 |
| adh36e00-page2_1_jpg.rf.b1104a907572f10b87561f097e6871ba.jpg | 0 | 0 | 100.00 | 307 |
| adp7aa00_jpg.rf.be7449073e76d0d9b5500e66f9f8e252.jpg | 1 | 1 | 22.68 | 308 |
| adq65f00_jpg.rf.f28f5d70c209bfcd7e93e6dbbf091505.jpg | 1 | 1 | 18.80 | 324 |
| aeb95e00_jpg.rf.c980b393baab1237bd1fa82d275a7840.jpg | 1 | 1 | 30.11 | 297 |
| aee44c00_jpg.rf.5d83ee8c8d32d763f0fb94486c3dcb7e.jpg | 1 | 1 | 14.99 | 305 |
| aeq93a00_jpg.rf.e6ed1bad4b619aaac899c43a41a466c7.jpg | 1 | 2 | 10.99 | 302 |
| ail70a00_jpg.rf.bdfddbd9d3846f5c65f0beaf6abfc008.jpg | 1 | 1 | 13.70 | 303 |
| ajj10e00_jpg.rf.d75937c912293548cd9a588a6b0eefc0.jpg | 1 | 1 | 11.73 | 315 |
| ajy01c00_jpg.rf.27ba4456d0e66648c5a9a1c1e5517008.jpg | 1 | 1 | 11.79 | 310 |
| aki32e00_jpg.rf.189270486e0e4ecaf8635da598568966.jpg | 1 | 2 | 10.43 | 311 |
| alz35d00_jpg.rf.013749684cd65d92911914fc69dd1c52.jpg | 1 | 1 | 21.32 | 483 |
| ama91d00-page03_3_jpg.rf.9194a22a15a2ac6514f59ef1432743c2.jpg | 1 | 2 | 20.53 | 290 |
| amw93e00_jpg.rf.1952165093c5431ada193146da9b1c7c.jpg | 1 | 1 | 29.37 | 303 |
| anv39d00_jpg.rf.d8325d793611186abf57f13286cb6d0e.jpg | 1 | 1 | 12.76 | 306 |
| arr09c00_jpg.rf.021d62e46a81cb73ea2c5e13792d23ef.jpg | 1 | 1 | 5.64 | 312 |
| arz92e00_jpg.rf.d032a45166eda3a7b6ca41c47bde7d69.jpg | 1 | 1 | 18.83 | 364 |

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
