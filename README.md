# IA2 - License Plate Detection and Recognition

## Project Goal

The objective of this project is to develop and evaluate machine learning models for detecting license plates in images (bounding box detection) and recognizing the characters on those plates (OCR). The focus is on European/Romanian license plates, with comparison to Spanish plates.

https://universe.roboflow.com/e-hh49k/european-license-plates-tjviy #european mix ~1455

#Implementation example
https://www.armyacademy.ro/cercetare/SAR/masarotunda/molder.pdf

## Project-Specific Requirements

* Experiment on at least 2 datasets.
* Compare at least 6 methods (>= 3 trained yourself, either from scratch or through transfer learning). Direct use of results reported in papers is not permitted.
* Use at least 3 distinct loss functions (implicitly or explicitly compared via different model architectures).
* Study the effect of transfer learning from one dataset to another.

## Datasets Used

1.  **Romanian License Plates:** (Approx. 534 images, XML annotations)
    * Source: `https://github.com/RobertLucian/license-plate-dataset`
    * Location: `data/raw/romanian_dataset/`
2.  **Spanish License Plates (UC3M-LP):** (Approx. 1975 images / 2547 plates, JSON annotations with polygons & characters)
    * Source: `https://github.com/ramajoballester/UC3M-LP`
    * Location: `data/raw/spanish_dataset/`

## Setup Instructions

1.  **Clone Repository:** `git clone [your-repo-url]`
2.  **Create Environment:** Requires Python >= 3.8. Use the provided `.venv` or create your own:
    ```bash
    python -m venv .venv
    # Activate (Windows):
    .\.venv\Scripts\activate
    # Activate (Linux/macOS):
    # source .venv/bin/activate
    ```
3.  **Install Dependencies:** Ensure PyTorch with CUDA support is installed correctly (see previous steps/PyTorch website) then install project requirements:
    ```bash
    pip install -r requirements.txt
    # Potentially install PyTorch+CUDA first if needed
    # pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    pip install ultralytics opencv-python numpy pandas matplotlib tqdm Pillow # Add more as needed
    ```
    *(Note: You should update `requirements.txt`)*
4.  **Data:** Download the raw datasets into the `data/raw/` subdirectories as specified above. Data is not tracked by Git.

## Current Status (As of May 5, 2025)

1.  **Project Setup:**
    * Git repository initialized.
    * Folder structure established.
    * Python environment (`.venv`) configured with PyTorch 2.5.1+cu121 and GPU (RTX 4060 Laptop) verified.
    * `.gitignore` is set up.
2.  **Data Exploration (EDA):**
    * Completed initial analysis and visualization of both datasets (`notebooks/1_Data_Exploration.ipynb`).
    * Confirmed ability to parse XML (Romanian) and JSON (Spanish) annotations.
    * Identified need for image resizing (especially for large Spanish images) and handling polygon coordinates.
    * Observed potential for multiple plates per image.
3.  **Data Preprocessing (`src/data_utils/`):**
    * Implemented core preprocessing script (`preprocess.py`) using `utils.py`.
    * Script resizes images to 640x640 (with padding) and saves them to `data/processed/images/`.
    * Script parses annotations, scales coordinates (handling polygons), and calculates absolute bounding boxes `[xmin, ymin, xmax, ymax]` relative to resized images.
    * Implemented formatter to save labels in **YOLO `.txt` format** to `data/processed/labels/{dataset}/{split}/`.
    * Successfully ran preprocessing for both datasets.
4.  **Detection Model Training (`src/detection/`):**
    * Implemented training script `train_detector.py` using Ultralytics Python API.
    * Successfully trained **YOLOv8n** on **Romanian** dataset (`romanian_lp.yaml`) for 50 epochs on GPU.
        * Achieved strong validation results (mAP50-95: 0.859).
        * Results currently saved in `runs/detect/LicensePlateDet/yolov8n_romanian_lp_e50X/` (Needs adjustment to save under `results/`).
    * Training **YOLOv8n** on **Spanish** dataset (`spanish_lp.yaml`) currently in progress/just started.

## File Structure Overview
IA2/
├── .venv/              # Python Virtual Environment (ignored by git)
├── data/
│   ├── raw/            # Original downloaded datasets (ignored by git)
│   │   ├── romanian_dataset/
│   │   └── spanish_dataset/
│   └── processed/      # Processed data (images, labels)
│       ├── images/     # Resized 640x640 images
│       └── labels/     # Formatted labels (currently YOLO)
├── notebooks/          # Jupyter notebooks for exploration
├── results/            # Target directory for saved models, logs, plots
├── src/
│   ├── data_utils/     # Preprocessing scripts (utils.py, preprocess.py)
│   ├── detection/      # Detection model training/evaluation scripts
│   └── ocr/            # OCR model training/evaluation scripts (TODO)
├── .gitignore
├── requirements.txt    # Project dependencies (TODO: Update)
├── romanian_lp.yaml    # Dataset config for YOLOv8
├── spanish_lp.yaml     # Dataset config for YOLOv8
└── README.md           # This file

## Next Steps

### Detection Pipeline

1.  **Monitor & Evaluate Spanish Run:** Analyze results for YOLOv8n on the Spanish dataset.
2.  **Adjust Save Directory:** Modify `train_detector.py` (`PROJECT_NAME` variable) to save future runs under the `results/` directory.
3.  **Compare YOLOv8 Models:** Train YOLOv8s (and optionally YOLOv8m) on both datasets. Compare performance (mAP, speed) vs YOLOv8n.
4.  **Train from Scratch:** Train YOLOv8n using `.yaml` config (no pre-trained weights) on one dataset to evaluate impact of pretraining.
5.  **Prepare COCO Labels:** Add a COCO JSON formatter function to `preprocess.py` and regenerate labels in `data/processed/labels/coco/`.
6.  **Implement Faster R-CNN (or SSD/EfficientDet):**
    * Choose & install a framework (e.g., MMDetection).
    * Configure Faster R-CNN for license plates using COCO JSON labels.
    * Train and evaluate Faster R-CNN on both datasets.
7.  **Transfer Learning:** Evaluate Romanian-trained models on Spanish validation set and vice-versa.
8.  **Summarize & Compare:** Document results for all models (>=6 methods), discuss loss functions used (>=3 types across models), and analyze transfer learning performance.

### OCR Pipeline

1.  **Prepare OCR Data (Spanish First):**
    * Write script (`src/ocr/preprocess_ocr.py`?) to read Spanish JSON annotations and processed images (`data/processed/images/spanish/...`).
    * Extract ground truth text (`lp_id`).
    * Crop license plate regions using scaled coordinates (derived from `poly_coord`).
    * Save cropped images and corresponding text labels (e.g., `ocr_crops/spanish/...` + `labels.txt`).
2.  **Baseline OCR Model:** Implement and train a baseline OCR model (e.g., CRNN) on the prepared Spanish data. Evaluate using CRR and Word Accuracy.
3.  **Address Romanian Data:** Develop strategy (transfer learning, pseudo-labeling) and prepare Romanian OCR data. Train/fine-tune OCR model.
4.  **Compare OCR Methods:** Experiment with different OCR architectures or techniques if time permits to meet comparison requirements (shared across detection/OCR).
5.  **Integrate with Detection:** Test OCR performance using bounding boxes predicted by the best detection model(s).
6.  **Summarize:** Document OCR results and methods.

### Final Report

* Combine Detection and OCR results.
* Ensure all project requirements are addressed and clearly presented with tables, plots, and discussion.
