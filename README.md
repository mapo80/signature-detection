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
few sample images from `dataset/`. Each detection is compared against
its YOLOv8 label file and must reach a reasonable IoU.

The `SignatureDetector` exposes a `scoreThreshold` parameter. The
model outputs a single logit per query which is passed through a
sigmoid. A default threshold of **0.1** closely matches the original
Python implementation and yields one detection per labeled image.

## Dataset evaluation

The `tools/DatasetReport` utility processes every image in `dataset/` using the
SDK and compares the predicted bounding box with the corresponding label. The
table below lists for each image the number of labels, number of detected
signatures, the percentage difference (100% - IoU), and the inference time in
milliseconds. With the updated threshold the average inference time is
**~626 ms**.

<!-- GENERATED REPORT -->

| Image | Labels | Detections | Diff% | Time ms |
|---|---|---|---|---|
| 001_15_PNG_jpg.rf.7ae3c04130de9c0e178fa2c1feb8eca9.jpg | 1 | 0 | 100.00 | 1095 |
| 00101001_png_jpg.rf.27db3f0cbf1a1ef078dcca2fdc2874af.jpg | 1 | 1 | 98.71 | 697 |
| 00101027_png_jpg.rf.a92770147b74d58b15829954bbba6ac6.jpg | 1 | 5 | 94.14 | 654 |
| 00101029_png_jpg.rf.14639faea024ffc684cd71be406650dc.jpg | 1 | 1 | 91.44 | 586 |
| 00104001_png_jpg.rf.bfafcce0144b089dc34bc63f05c4ea12.jpg | 1 | 1 | 98.44 | 618 |
| ... | ... | ... | ... | ... |
| 02205002_png_jpg.rf.c491e313d0f62c95e2990f664fe44c8b.jpg | 1 | 84 | 41.44 | 288 |
| 02302023_png_jpg.rf.7b59991fc80b082bb1925a5071c22464.jpg | 1 | 65 | 49.00 | 244 |
| 02302070_png_jpg.rf.5db163a7de9ae621c56c1a86c3de2d84.jpg | 1 | 67 | 71.82 | 259 |
| 02305070_png_jpg.rf.d7ecd0ef0984bbe479271ab32d0888af.jpg | 1 | 66 | 33.45 | 225 |
| 02403024_png_jpg.rf.234c51b41d237cc3246c71e4fae0e0e0.jpg | 1 | 66 | 57.94 | 224 |
| 02601026_png_jpg.rf.2e55a766ff4dc7a77260ab10c910bca5.jpg | 1 | 76 | 43.46 | 249 |
| 02701027_png_jpg.rf.cbb3219446cb316a4c42533c35249aef.jpg | 1 | 68 | 38.28 | 233 |
| 02702027_png_jpg.rf.819a7dc18c7f3c8ce22710a3ed5abc08.jpg | 1 | 84 | 36.60 | 229 |
| 02703027_png_jpg.rf.b76da2e8d9524be6951e25848b1add1a.jpg | 1 | 79 | 41.97 | 246 |
| 03_069_png_jpg.rf.5f1e837207ef6358f292d292daf5e71e.jpg | 1 | 76 | 63.16 | 221 |
| 03205085_png_jpg.rf.93ec2f5b272bef47099a1b0f3c158976.jpg | 1 | 75 | 37.43 | 226 |
| 08501085_png_jpg.rf.9221f3b87796f82891e256a75a7696ee.jpg | 1 | 79 | 46.33 | 255 |
| 08504085_png_jpg.rf.f5c2b292988cf055946a56494933cdea.jpg | 1 | 75 | 41.82 | 236 |
| 08602086_png_jpg.rf.48160dbed60bba75fa1b4edd0a8c945d.jpg | 1 | 70 | 31.95 | 248 |
| 09304093_png_jpg.rf.9494e76e10a326b85b826c8ec40f5152.jpg | 1 | 71 | 34.71 | 235 |
| 1_png.rf.2b1f3375a14adfb8dc7ba771d0c3ebe3.jpg | 1 | 80 | 59.14 | 266 |
| 1_png.rf.44276085919656a54a022d02083f4e0e.jpg | 1 | 80 | 42.52 | 235 |
| 1_png.rf.4851c565455ea0bf15090a87112764bc.jpg | 1 | 80 | 56.32 | 227 |
| 1_png.rf.953d2eb49ec06cbdceded4af8f4cbc44.jpg | 1 | 80 | 41.70 | 255 |
| 1_png.rf.cc7899c34d1d8241b3026a8e0fa74a47.jpg | 1 | 80 | 56.32 | 246 |
| 1_png.rf.cd81f8e3ba409c128c01fe3f2f221fb6.jpg | 1 | 80 | 42.33 | 253 |
| 10_png.rf.2d8da220f77b18e7de44f4756cc687b3.jpg | 1 | 78 | 40.48 | 324 |
| 10_png.rf.6ff4cc9343039deacf1e1dec59ef0821.jpg | 1 | 78 | 53.94 | 248 |
| 10_png.rf.7f8156a37f83d2155f9b1591e61b19da.jpg | 1 | 78 | 44.20 | 226 |
| 10_png.rf.a88fd2178d8b8e93944d3a6f02aa409c.jpg | 1 | 78 | 45.31 | 227 |
| 14_png.rf.04d6e155fea10001a2bb6b9fd5fb0d63.jpg | 1 | 66 | 50.46 | 245 |
| 14_png.rf.7e3c63e0d06e0949759dfe46795fba76.jpg | 1 | 66 | 51.92 | 235 |
| 14_png.rf.8df113443a6ffc3a6271af52c8706b52.jpg | 1 | 66 | 48.55 | 232 |
| 14_png.rf.cbc1767876a4d6119ee164f00927918d.jpg | 1 | 66 | 54.75 | 249 |
| 14_png.rf.fcc5241fe17a3726f416dccb679f3a30.jpg | 1 | 66 | 59.00 | 231 |
| 14_png.rf.ffe93fbb3b9f8030b794f3d9673289ac.jpg | 1 | 66 | 51.59 | 259 |
| 18_png.rf.03cc6781acb7697210a2854f5de12c24.jpg | 1 | 77 | 59.92 | 231 |
| 18_png.rf.16c61b94c3d61bdb668e179ca9dc3212.jpg | 1 | 77 | 58.77 | 222 |
| 18_png.rf.65d9c8997b92e7eeeab4c03087fb791f.jpg | 1 | 77 | 62.47 | 250 |
| 18_png.rf.6f1ba185b3bff2da1d753d0fdae324d5.jpg | 1 | 77 | 66.74 | 223 |
| 18_png.rf.9980efa13f873328e9d4aceaaa0cd730.jpg | 1 | 77 | 58.77 | 229 |
| 18_png.rf.a04237916cf8afa8c755aada5f331ae1.jpg | 1 | 77 | 64.32 | 248 |
| 2_png.rf.43fa3c3f3dfe467f37395f4ab16405dc.jpg | 0 | 76 | 100.00 | 219 |
| 2_png.rf.5f247cf3a8537bfe796799662d8efd66.jpg | 0 | 76 | 100.00 | 224 |
| 2_png.rf.f25b6ecd6ca8f4124cebd9a7bb1d4a15.jpg | 0 | 76 | 100.00 | 257 |
| 21_png.rf.1ff01628f0cab0e53ee7aa75e8e87f1f.jpg | 1 | 75 | 54.71 | 230 |
| 21_png.rf.3338010398ec8ccf5acf084f9d65de66.jpg | 1 | 75 | 56.69 | 264 |
| 21_png.rf.3968b3a75b79489b6ef0cf8439d9fde4.jpg | 1 | 75 | 54.71 | 232 |
| 21_png.rf.43f466219daac754f9d2ab234c9f6897.jpg | 1 | 75 | 45.73 | 261 |
| 21_png.rf.9220576a908f8f52b532894da9695635.jpg | 1 | 75 | 56.54 | 221 |
| 3_png.rf.50b8175ab72a33adbff972beb671c512.jpg | 0 | 68 | 100.00 | 227 |
| 3_png.rf.a098c14de07b8ec3b96c80ec6df66e44.jpg | 0 | 68 | 100.00 | 254 |
| 3_png.rf.a7a511fb6136459523901c3fa456480b.jpg | 0 | 68 | 100.00 | 230 |
| 4_png.rf.d4374a737da88e15f923b657ce573dfd.jpg | 0 | 79 | 100.00 | 226 |
| 4_png.rf.e065ff219e031b0ec80e2b3b1c822245.jpg | 0 | 79 | 100.00 | 286 |
| MicrosoftTeams-image-1-Copy-Copy_png.rf.0411222c4cdd7a87b26e1499094c8abe.jpg | 1 | 78 | 58.09 | 271 |
| MicrosoftTeams-image-1-Copy-Copy_png.rf.5f715cdf9b6077827ba941958c6471cf.jpg | 1 | 78 | 59.51 | 240 |
| MicrosoftTeams-image-1-Copy-Copy_png.rf.e7156c9aa338e6d3e11b01cac2a1b6da.jpg | 1 | 78 | 46.63 | 242 |
| MicrosoftTeams-image-10-Copy_png.rf.70058f1172783de55228f382c6926a01.jpg | 1 | 75 | 55.41 | 244 |
| MicrosoftTeams-image-10-Copy_png.rf.b3641b1a85725bb36af117ed22d8465c.jpg | 1 | 75 | 47.96 | 223 |
| MicrosoftTeams-image-10-Copy-Copy_png.rf.559d2e85435cc13a9a59c591e458fb41.jpg | 1 | 66 | 51.45 | 247 |
| MicrosoftTeams-image-17-Copy-Copy_png.rf.52eb98eceab8212bdada65e4fb576d8a.jpg | 1 | 77 | 65.45 | 230 |
| MicrosoftTeams-image-2-Copy_png.rf.9c934cec6018d1aedc65599007c88dbe.jpg | 1 | 66 | 54.44 | 219 |
| MicrosoftTeams-image-4-Copy_png.rf.c503ef98258345642cdba6623441a9a2.jpg | 1 | 77 | 65.60 | 253 |
| MicrosoftTeams-image-5-Copy_png.rf.c5f05eb16773b7cb282c4e0beccfa555.jpg | 1 | 75 | 51.66 | 243 |
| MicrosoftTeams-image-Copy-2-_png.rf.4243e55eb9f3c805f6c0d54f2d8a9341.jpg | 1 | 80 | 54.53 | 327 |
| MicrosoftTeams-image-Copy-2-_png.rf.792670557ee74e5a96bffabc81e40af0.jpg | 1 | 80 | 52.10 | 317 |
| NFI-00101014_png_jpg.rf.cbeb968701eaf9f15d7bbf0529df4b7e.jpg | 1 | 73 | 48.61 | 284 |
| NFI-00103027_PNG_jpg.rf.c763a0557185133b505ad43670e78239.jpg | 1 | 80 | 56.51 | 235 |
| NFI-00305002_png_jpg.rf.0797ea7b2ca8d9f6d6c652d496aebd16.jpg | 1 | 71 | 61.46 | 237 |
| NFI-00401024_png_jpg.rf.4a8cce7a421e30602dd15aa063e130b6.jpg | 1 | 83 | 32.43 | 264 |
| NFI-00803008_png_jpg.rf.5ba69debb10f322c26a3429f68231637.jpg | 1 | 63 | 57.39 | 227 |
| NFI-00804008_png_jpg.rf.cb233aebe9c9b3152feac7c1d260d2bb.jpg | 1 | 75 | 46.09 | 245 |
| NFI-00902009_png_jpg.rf.9e72b1cbffa61913b29d580495fc1ace.jpg | 1 | 74 | 46.10 | 222 |
| NFI-00903009_png_jpg.rf.cc4355c6cb338363a8cbc819b366c457.jpg | 1 | 81 | 51.88 | 228 |
| NFI-01102011_png_jpg.rf.160e293d7a0a1a0e836212c5ceff789c.jpg | 1 | 61 | 46.31 | 245 |
| NFI-01302013_png_jpg.rf.28ca47c98d2bb16234f7d08d7e9d9df4.jpg | 1 | 77 | 49.81 | 234 |
| NFI-02401024_png_jpg.rf.67eacdbd4b4661a5435f7682673944a3.jpg | 1 | 78 | 42.68 | 226 |
| NFI-02401024_png_jpg.rf.fdfc8f98be1e92c8684e97613bae6e7f.jpg | 1 | 75 | 44.57 | 246 |
| NFI-02902029_PNG_jpg.rf.7160649ae532f53ff6baad3728b288b3.jpg | 1 | 78 | 45.33 | 230 |

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

## Sample images

The `samples/detr` and `samples/yolov8s` directories contain 20 sample images each. Every image is annotated with the ground truth box (red) and the prediction from the respective model. Generate them with:

```bash
dotnet run --project tools/DrawBoundingBoxes/DrawBoundingBoxes.csproj
```

## YOLOv8s comparison

Download `yolov8s.onnx` from the
[tech4humans/yolov8s-signature-detector](https://huggingface.co/tech4humans/yolov8s-signature-detector)
repository (an authenticated HuggingFace account may be required) and place it in the project root. When the file is present the tool will also run the YOLOv8s model and write its annotated images to `samples/yolov8s/`.

## YOLOv8s Python inference

The repository also includes `yolov8_inference.py` which performs inference using the `ultralytics` package and draws both the predicted boxes and the ground truth labels. The script expects `yolov8s.onnx` in the `model/` directory and reads images from `dataset/`. Annotated results are written to `samples/yolov8s_py/`.

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
