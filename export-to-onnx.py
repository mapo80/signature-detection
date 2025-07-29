from transformers import AutoImageProcessor, ConditionalDetrForObjectDetection
from PIL import Image
import torch

model_name = "tech4humans/conditional-detr-50-signature-detector"

processor = AutoImageProcessor.from_pretrained(model_name)
model = ConditionalDetrForObjectDetection.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    from_tf=False,
    ignore_mismatched_sizes=True
)

dummy_image = Image.new("RGB", (640, 640), color=0)
dummy_input = processor(images=[dummy_image], return_tensors="pt")["pixel_values"]

onnx_path = "conditional_detr_signature.onnx"
torch.onnx.export(
    model,
    (dummy_input,),
    onnx_path,
    input_names=["pixel_values"],
    output_names=["logits", "boxes"],
    dynamic_axes={"pixel_values": {0: "batch_size"}, "boxes": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=14,
)
print(f"Exported model to {onnx_path}")
