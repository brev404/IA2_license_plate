# src/detection/train_detector.py

from ultralytics import YOLO
import torch
import pathlib
import os

# --- Configuration ---

# Use pathlib to define paths relative to this script file
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve() # Assumes script is in ProjectRoot/src/detection/

# --- Select Dataset ---
# Choose which dataset YAML to use (uncomment the one you want)
DATASET_YAML = PROJECT_ROOT / 'yolo_romanian_lp.yaml'
# DATASET_YAML = PROJECT_ROOT / 'yolo_spanish_lp.yaml'

# --- End Dataset Selection ---

# --- Training Parameters ---
# MODEL_CONFIG_OR_WEIGHTS = 'yolov8n.yaml' # For training from scratch
# MODEL_TO_TRAIN = str(PROJECT_ROOT / 'results/LicensePlateDet/yolov8s_yolo_spanish_lp_e50/weights/best.pt')
#MODEL_TO_TRAIN = 'yolov8n.yaml'   # Start with pretrained weights (recommended)
#MODEL_TO_TRAIN = 'yolov8n.pt'   # Start with pretrained weights (recommended) # yolov8n.pt first batch then yolov8s.pt second batch
MODEL_TO_TRAIN = 'yolov8s.pt'   # Start with pretrained weights (recommended) # yolov8n.pt first batch then yolov8s.pt second batch
EPOCHS = 50#50                     # Number of epochs to train for
IMG_SIZE = 640                  # Image size (must match preprocessing)
BATCH_SIZE = 4                 # Adjust based on your GPU memory (start lower if needed)
PROJECT_NAME = 'results/LicensePlateDet' # Results saved under runs/detect/<PROJECT_NAME>_*
# --- End Training Parameters ---

# Automatically select device
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
if DEVICE == 0:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")


def train():
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Using Dataset YAML: {DATASET_YAML}")
    if not DATASET_YAML.exists():
        print(f"ERROR: Dataset YAML file not found at {DATASET_YAML}")
        return

    print(f"Using Model: {MODEL_TO_TRAIN}")
    print(f"Training for {EPOCHS} epochs.")

    # Load the model
    # '.pt' loads pretrained weights, '.yaml' builds from scratch
    model = YOLO(MODEL_TO_TRAIN)

    # Define a specific name for this training run
    run_name = f'{pathlib.Path(MODEL_TO_TRAIN).stem}_{DATASET_YAML.stem}_e{EPOCHS}'

    # Start training
    try:
        results = model.train(
            data=str(DATASET_YAML), # Path to dataset config file
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            device=DEVICE,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,   # Groups runs under this project name
            name=run_name,          # Specific name for this run's folder
            exist_ok=False,          # Prevents overwriting existing runs with the same name
            box=15, # Box loss gain 
            lr0 = 0.001 #1/10 of the initial learning rate
            # Add more hyperparameters if needed:
            # patience=50, # Early stopping
            # optimizer='AdamW',
            # lr0=0.01, # Initial learning rate
            # workers=8 # Number of dataloader workers (adjust based on CPU cores)
        )
        print("Training finished successfully.")
        print(f"Results saved to: {results.save_dir}")

    except Exception as e:
        print(f"An error occurred during training: {e}")


if __name__ == "__main__":
    # This ensures the training runs only when the script is executed directly
    train()