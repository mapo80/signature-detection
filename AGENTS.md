# Development Notes

To run the .NET SDK and the tests you need .NET 9 (version 303).
Install it with the provided `dotnet-install.sh` script:

```bash
./dotnet-install.sh --version 9.0.303 --install-dir "$HOME/.dotnet"
export PATH="$HOME/.dotnet:$PATH"
```

The ONNX model used by the SDK is split into two parts in the repository.
Before building or testing, recombine them:

```bash
cat conditional_detr_signature.onnx_parte_{1,2} > conditional_detr_signature.onnx
```

Run the unit tests with:

```bash
dotnet test
```

Build the evaluation tool before running tests:

```bash
dotnet build tools/EvaluateMetrics/EvaluateMetrics.csproj -c Release
```

The dataset used in the tests is stored in `dataset/` and follows the
YOLOv8 format.

To generate a full report of detections for every image run:

```bash
dotnet run --project tools/DatasetReport/DatasetReport.csproj
```
The results will be written to `dataset_report.csv`.

## Vincoli di commit

- Tutti i file binari generati devono essere serializzati in Base64 prima del commit.
- Il README.md deve contenere il report dettagliato e aggiornato automaticamente.
