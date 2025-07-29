import os
import cv2
import supervision as sv
from ultralytics import YOLO

from dataset import download_dataset, DATASET_DIR

REPO_ID = "tech4humans/yolov8s-signature-detector"
FILENAME = "yolov8s.onnx"
LOCAL_MODEL_DIR = "model"
OUTPUT_DIR = os.path.join("samples", "yolov8s_py")


def download_model():
    """Download the model using Hugging Face Hub"""
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    try:
        print(f"Downloading model from {REPO_ID}...")
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=LOCAL_MODEL_DIR,
            cache_dir=None,
        )
        print("Model downloaded successfully!")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise


def load_ground_truth(image_name):
    label_path = os.path.join(DATASET_DIR, "labels", os.path.splitext(image_name)[0] + ".txt")
    boxes = []
    if os.path.exists(label_path):
        with open(label_path) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 5:
                    _, cx, cy, w, h = map(float, parts[:5])
                    boxes.append((cx, cy, w, h))
    return boxes


if __name__ == "__main__":
    model_path = os.path.join(LOCAL_MODEL_DIR, FILENAME)
    if not os.path.exists(model_path):
        model_path = download_model()

    if not os.path.exists(DATASET_DIR):
        download_dataset()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO(model_path)

    images = [f for f in os.listdir(os.path.join(DATASET_DIR, "images")) if f.endswith(".jpg")]
    if not images:
        print("No images found in dataset")
        exit()

    for name in images:
        img_path = os.path.join(DATASET_DIR, "images", name)
        results = model(img_path)
        image = cv2.imread(img_path)
        detections = sv.Detections.from_ultralytics(results[0])
        box_annotator = sv.BoxAnnotator(color=sv.Color.green())
        annotated = box_annotator.annotate(scene=image.copy(), detections=detections)

        for cx, cy, w, h in load_ground_truth(name):
            x1 = int((cx - w / 2) * image.shape[1])
            y1 = int((cy - h / 2) * image.shape[0])
            x2 = int((cx + w / 2) * image.shape[1])
            y2 = int((cy + h / 2) * image.shape[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

        out_path = os.path.join(OUTPUT_DIR, name)
        cv2.imwrite(out_path, annotated)
        print(f"Saved {out_path}")
