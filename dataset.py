import os

DATASET1_DIR = os.path.join(os.path.dirname(__file__), "dataset", "dataset1")
DATASET2_DIR = os.path.join(os.path.dirname(__file__), "dataset", "dataset2")

# Keep the old name for backward compatibility, pointing to the first dataset
DATASET_DIR = DATASET1_DIR

def download_dataset():
    """Placeholder to mirror python sample script."""
    print(f"Dataset available at {DATASET_DIR}")
