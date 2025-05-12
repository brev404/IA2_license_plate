# IA2 - License Plate Detection and Recognition

## Project Goal

The objective of this project is to develop and evaluate machine learning models for detecting license plates in images (bounding box detection) and recognizing the characters on those plates (OCR). The focus is on European/Romanian license plates, with a comparative study involving Spanish license plates.

## Project-Specific Requirements Status

- [x] Experiment on at least 2 datasets.
- [x] Compare at least 6 methods (>= 3 trained yourself). (YOLOv8n-scratch, YOLOv8n-PT, YOLOv8s-PT, YOLOv8-Transfer, Faster R-CNN, RetinaNet, Tesseract OCR)
- [x] Use at least 3 distinct loss functions (YOLOv8 suite, Faster R-CNN suite, RetinaNet suite).
- [x] Study the effect of transfer learning from one dataset to another.

## Datasets Used

1.  **Romanian License Plates:**
    * Source: `https://github.com/RobertLucian/license-plate-dataset` (Approx. 534 images, XML annotations)
    * Location: `data/raw/romanian_dataset/`
2.  **Spanish License Plates (UC3M-LP):**
    * Source: `https://github.com/ramajoballester/UC3M-LP` (Approx. 1975 images / 2547 plates, JSON annotations)
    * Location: `data/raw/spanish_dataset/`

## Directory Structure

A high-level overview of the project structure:

-   `data/`: Contains raw, processed, and intermediate data, including images and labels.
    -   `raw/`: Original datasets.
    -   `processed/`: Resized images and generated labels (YOLO, COCO).
    -   `ocr_input_data/`: Cropped plates for OCR.
-   `src/`: Contains all source code.
    -   `data_utils/`: Scripts for data preprocessing and label generation.
    _   `detection/`: Scripts for training and evaluating detection models.
    _   `character_recognition/`: Scripts for OCR.
-   `results/`: Stores model checkpoints, training logs, and evaluation metrics.
    -   `LicensePlateDet/`: YOLOv8 training runs and results (CSV logs).
    -   `FasterRCNN_checkpoints/`, `FasterRCNN_evaluation/`: Faster R-CNN models and evaluations (TXT metrics).
    -   `RetinaNet_checkpoints/`, `RetinaNet_evaluation/`: RetinaNet models and evaluations (TXT metrics).
    -   `OCR_Experiments/`: OCR results (JSON results, TXT analysis).
-   `notebooks/`: Jupyter notebooks (e.g., for data exploration).
-   `IA 2 SotA/`: Literature review (State-of-the-Art papers).

(For a detailed file tree, see `project_structure.txt`)

## Methodology

### 1. Data Preprocessing

-   Raw images from both datasets are resized to a consistent input size (e.g., 640x640) while maintaining aspect ratio via padding.
-   Annotations are parsed (XML for Romanian, JSON for Spanish) and converted to scaled bounding boxes relative to the processed images.
-   This processed image and annotation data is stored in an intermediate format (`data/processed/intermediate/processed_annotations.pkl`).
-   **Script:** `src/data_utils/preprocess.py`
-   From the intermediate data, labels are generated in:
    -   **COCO format:** For training Faster R-CNN and RetinaNet.
        -   **Script:** `src/data_utils/create_coco_labels.py`
        -   **Output:** `data/processed/labels/coco/`
    -   **YOLO format:** For training YOLOv8 models.
        -   **Script:** `src/data_utils/create_yolo_labels.py`
        -   **Output:** `data/processed/labels/yolo/` (organized by dataset/split)

### 2. License Plate Detection

Multiple object detection architectures were trained and evaluated:

**a) YOLOv8 (Ultralytics)**
    * **Description:** State-of-the-art single-stage detector known for its speed and accuracy. Variants YOLOv8n (nano) and YOLOv8s (small) were used.
    * **Training Script:** `src/detection/train_detector.py`
    * **Dataset Configs:** `yolo_romanian_lp.yaml`, `yolo_spanish_lp.yaml` (in project root)
    * **Loss Function Suite:** Comprises Bounding Box Loss (e.g., CIoU), Classification Loss (e.g., BCEWithLogitsLoss or Varifocal Loss), and Distribution Focal Loss (DFL).
    * **Experiments:**
        * Training from scratch (using `.yaml` model definition).
        * Fine-tuning from COCO pre-trained weights (using `.pt` model weights).
        * Transfer learning from Spanish to Romanian dataset.

**b) Faster R-CNN (TorchVision)**
    * **Description:** A popular two-stage detector using a ResNet50-FPN backbone. Employs a Region Proposal Network (RPN) and an RoI Head.
    * **Training Script:** `src/detection/train_faster_rcnn.py`
    * **Evaluation Script:** `src/detection/evaluate_faster_rcnn.py`
    * **Loss Function Suite:**
        * RPN: Objectness Loss (Binary Cross-Entropy) & BBox Regression Loss (Smooth L1).
        * RoI Head: Classification Loss (Cross-Entropy) & BBox Regression Loss (Smooth L1).

**c) RetinaNet (TorchVision)**
    * **Description:** A single-stage detector with a ResNet50-FPN backbone, designed to address class imbalance in dense object detection using Focal Loss.
    * **Training Script:** `src/detection/train_retinanet.py`
    * **Evaluation Script:** `src/detection/evaluate_retinanet.py`
    * **Loss Function Suite:**
        * Classification Loss: **Focal Loss**.
        * BBox Regression Loss: Smooth L1 Loss.

### 3. License Plate Character Recognition (OCR)

-   **Tool:** Tesseract OCR engine.
-   **Input Crops:** Can use either:
    1.  Ground truth bounding box crops (generated by `src/character_recognition/preprocess_ocr_data.py`).
    2.  Crops from the output of trained detection models.
-   **Preprocessing for OCR:** Grayscaling, contrast enhancement (CLAHE), adaptive thresholding (implemented in `src/character_recognition/utils_ocr.py`).
-   **Experiment Runner:** `src/character_recognition/run_ocr_experiment.py`
-   **Results Analysis:** `src/character_recognition/analyze_ocr_results.py` (calculates format validation rates, etc.). For Spanish GT crops, `run_ocr_experiment.py` also records exact match if GT text is available from `ground_truth_texts.tsv`.

## Experiments and Results

### Detection Model Performance

**Key Metrics:**
* `AP@\[.5:.95]`: Primary COCO metric (mAP@IoU=0.50:0.95, area=all, maxDets=100).
* `AP@.50`: mAP@IoU=0.50 (PASCAL VOC metric, area=all, maxDets=100).
* `AP@.75`: mAP@IoU=0.75 (stricter localization, area=all, maxDets=100).

**1. Romanian Dataset (Validation Split - `romanian_valid`)**

| Model Architecture | Training Details (Epochs, Pretrain Type)              | Loss Function Principle(s)                     | AP@\[.5:.95] | AP@.50 | AP@.75 | Result File/Path                                                                                           |
| :----------------- | :---------------------------------------------------- | :--------------------------------------------- | :---------- | :----- | :----- | :----------------------------------------------------------------------------------------------------------------- |
| Faster R-CNN       | ResNet50-FPN (10 epochs, from COCO)                   | RPN (BCE, SmoothL1), RoI (CE, SmoothL1)        | **0.7665** | 0.9930 | 0.9121 | `results/FasterRCNN_evaluation/fasterrcnn_resnet50_fpn_epoch_10/eval_on_romanian_valid_metrics.txt`              |
| RetinaNet          | ResNet50-FPN (e.g., 15 epochs, from COCO)             | Classification (Focal Loss), BBox (Smooth L1)  | *TBD* | *TBD* | *TBD* | *(Run `evaluate_retinanet.py` on Romanian valid set)* |
| YOLOv8s            | Pre-trained on COCO, 50 epochs                        | YOLOv8 (CIoU, BCE/VFL, DFL)                    | **0.8618** | 0.9925 | N/A    | `results_yolov8s_yolo_romanian_lp_e50.csv` (epoch 49)                                                              |
| YOLOv8n            | Pre-trained on COCO, 50 epochs                        | YOLOv8 (CIoU, BCE/VFL, DFL)                    | **0.8578** | 0.9926 | N/A    | `results_yolov8n_yolo_romanian_lp_e50.csv` (epoch 50)                                                              |
| YOLOv8n            | From Scratch, 50 epochs                               | YOLOv8 (CIoU, BCE/VFL, DFL)                    | **0.7466** | 0.9861 | N/A    | `results_yolov8n_yolo_romanian_lp_e50_train_0.csv` (epoch 47)                                                      |
| YOLOv8s (Transfer) | Transfer ES->RO (`yolov8s.pt` base), 30 epochs        | YOLOv8 (CIoU, BCE/VFL, DFL)                    | **0.8530** | 0.9941 | N/A    | `results_best_yolo_romanian_lp_e30_transfer.csv` (epoch 30)                                                        |

**2. Spanish Dataset (Evaluation on `spanish_test` for RetinaNet, `spanish_valid` assumed for YOLO from CSV names)**
*(Note: YOLO CSVs usually report on a validation split. If `spanish_test` was used for YOLO, please adjust the table or note this.)*

| Model Architecture | Training Details (Epochs, Pretrain Type)              | Loss Function Principle(s)                     | AP@\[.5:.95] | AP@.50 | AP@.75 | Result File/Path                                                                                           |
| :----------------- | :---------------------------------------------------- | :--------------------------------------------- | :---------- | :----- | :----- | :----------------------------------------------------------------------------------------------------------------- |
| RetinaNet          | ResNet50-FPN (15 epochs, from COCO)                   | Classification (Focal Loss), BBox (Smooth L1)  | **0.6807** | 0.9478 | 0.8275 | `results/RetinaNet_evaluation/retinanet_resnet50_fpn_epoch_15/eval_on_spanish_test_metrics.txt`               |
| Faster R-CNN       | ResNet50-FPN (e.g., 10 epochs, from COCO)             | RPN (BCE, SmoothL1), RoI (CE, SmoothL1)        | *TBD* | *TBD* | *TBD* | *(Run `evaluate_faster_rcnn.py` on Spanish test/valid set)* |
| YOLOv8s            | Pre-trained on COCO, 50 epochs                        | YOLOv8 (CIoU, BCE/VFL, DFL)                    | **0.8929** | 0.9935 | N/A    | `results_yolov8s_yolo_spanish_lp_e50.csv` (epoch 50)                                                               |
| YOLOv8n            | Pre-trained on COCO, 50 epochs                        | YOLOv8 (CIoU, BCE/VFL, DFL)                    | **0.8720** | 0.9928 | N/A    | `results_yolov8n_yolo_spanish_lp_e50.csv` (epoch 49)                                                               |
| YOLOv8n            | From Scratch, 50 epochs                               | YOLOv8 (CIoU, BCE/VFL, DFL)                    | **0.7762** | 0.9768 | N/A    | `results_yolov8n_yolo_spanish_lp_e50_train_0.csv` (epoch 50)                                                       |

*Note on YOLO Result File Paths: The table lists the original CSV filenames you provided. The actual Ultralytics output path would be within `results/LicensePlateDet/{run_name}/results.csv` where `{run_name}` corresponds to the experiment (e.g., `yolov8s_pt_yolo_romanian_lp_e50`).*

### Transfer Learning Study (YOLOv8)

-   **Experiment:** A YOLOv8s model was used as a base for transfer learning. One path involved fine-tuning from COCO pre-trained weights directly on Romanian data. Another path involved taking a model (presumably the COCO pre-trained YOLOv8s) and fine-tuning it on the Spanish dataset, then subsequently fine-tuning those weights on the Romanian dataset for 30 epochs.
-   **Transfer Model (COCO -> Spanish -> Romanian, 30 epochs on RO):**
    * AP@\[.5:.95]: **0.8530**
    * AP@.50: **0.9941**
    * (Source: `results_best_yolo_romanian_lp_e30_transfer.csv`, epoch 30)
-   **Comparison with Romanian baselines (on Romanian `valid` set):**
    * YOLOv8s (COCO Pre-trained -> Romanian, 50 epochs): AP@\[.5:.95] = **0.8618**, AP@.50 = **0.9925**
    * YOLOv8n (From Scratch -> Romanian, 50 epochs): AP@\[.5:.95] = **0.7466**, AP@.50 = **0.9861**
-   **Observation:** Fine-tuning from a model already exposed to license plates (Spanish) yields strong performance on the Romanian dataset, very close to fine-tuning directly from general COCO pre-trained weights for more epochs, and significantly better than training from scratch.

### OCR Performance (Tesseract)

-   OCR experiments are run using `src/character_recognition/run_ocr_experiment.py`.
-   Analysis summaries are saved in `results/OCR_Experiments/{experiment_name}/{experiment_name}_analysis.txt`.
-   The primary metric currently reported in these summaries is the **Format Validation Rate**.
-   For Spanish GT crops, exact string match (`is_exact_match`) is also recorded in the `ocr_results.json` if GT text is available.

**Summary of OCR Format Validation Rates:**

| Experiment Description                                         | Dataset       | Split   | Crop Source        | Detector Model (if applicable)       | Total Plates | Valid Format (%) | Analysis File                                                                             |
| :------------------------------------------------------------- | :------------ | :------ | :----------------- | :----------------------------------- | :----------- | :--------------- | :---------------------------------------------------------------------------------------- |
| Tesseract on GT Crops                                          | Romanian      | valid   | Ground Truth       | N/A                                  | 131          | 3.82%            | `exp001_tesseract_romanian_gt_crops_valid_ocr_results_analysis.txt`                       |
| Tesseract on GT Crops                                          | Spanish       | test    | Ground Truth       | N/A                                  | 491          | 2.04%            | `exp002_tesseract_spanish_gt_crops_test_ocr_results_analysis.txt`                       |
| Tesseract on GT Crops                                          | Spanish       | train   | Ground Truth       | N/A                                  | 2056         | 1.17%            | `exp003_tesseract_spanish_gt_crops_train_ocr_results_analysis.txt`                      |
| Tesseract on Detector Crops (YOLOv8s-PT)                     | Spanish       | test    | YOLOv8s-PT         | `yolov8s_yolo_spanish_lp_e50`        | 502          | 2.59%            | `ES_YOLOv8s_Tesseract_Test_DetectorCrops_ocr_results_analysis.txt`                        |
| Tesseract on Detector Crops (YOLOv8n-Scratch)                | Romanian      | valid   | YOLOv8n-Scratch    | `yolov8n_yolo_romanian_lp_e50_train_0` | 144          | 1.39%            | `RO_YOLOv8nScratch_DetectorCrops_Valid_Tesseract_ocr_results_analysis.txt`              |
| Tesseract on Detector Crops (YOLOv8s-PT)                     | Romanian      | valid   | YOLOv8s-PT         | `yolov8s_yolo_romanian_lp_e50`       | 146          | 3.42%            | `RO_YOLOv8s_Tesseract_Valid_DetectorCrops_ocr_results_analysis.txt`                     |
| Tesseract on Detector Crops (YOLOv8s-Transfer ES->RO)        | Romanian      | valid   | YOLOv8s-Transfer   | `best_yolo_romanian_lp_e30_transfer` | 145          | 3.45%            | `RO_YOLOv8s_Transfer_ES_to_RO_DetectorCrops_Valid_Tesseract_ocr_results_analysis.txt` |

**Observations on OCR:**
- The current Tesseract OCR pipeline yields a low percentage of plates whose recognized text strictly matches the defined Romanian or Spanish formats.
- Further analysis of the `ocr_results.json` files, especially for Spanish ground truth crops where `is_exact_match` is available, is needed to determine Character Error Rate (CER) and Word Error Rate (WER) for a more nuanced understanding of OCR accuracy beyond just format validation.
- The choice of detector appears to have a slight influence on the downstream OCR format validation rate, but all rates are currently low.

*(This section should be expanded with more detailed OCR accuracy metrics (CER, WER, exact match rates) after further analysis of the `ocr_results.json` files.)*

## How to Run

### 1. Setup
- Clone the repository.
- Create a Python virtual environment (e.g., using `venv` or `conda`).
  ```bash
  python -m venv .venv
  source .venv/bin/activate # Linux/macOS
  # .venv\Scripts\activate # Windows
  Install dependencies:
Bash

pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cuXXX](https://download.pytorch.org/whl/cuXXX) # Replace cuXXX with your CUDA version e.g., cu118, cu121, or cpu
pip install ultralytics opencv-python pytesseract pycocotools tqdm PyYAML pandas matplotlib scikit-learn Pillow
# For Windows, pycocotools might need: pip install pycocotools-windows
Ensure Tesseract OCR is installed on your system and added to your PATH. Alternatively, set TESSERACT_CMD_PATH in src/character_recognition/run_ocr_experiment.py.
2. Data Preprocessing
Run the main preprocessing script to prepare images and intermediate annotations:

Bash

python src/data_utils/preprocess.py
Generate labels in COCO and YOLO formats:

Bash

python src/data_utils/create_coco_labels.py
python src/data_utils/create_yolo_labels.py
Generate ground truth crops for OCR baseline:

Bash

python src/character_recognition/preprocess_ocr_data.py
3. Training Detection Models
a) YOLOv8

Modify src/detection/train_detector.py to select:
DATASET_YAML (e.g., PROJECT_ROOT / 'yolo_romanian_lp.yaml')
MODEL_TO_TRAIN (e.g., 'yolov8s.pt' for COCO pre-trained, or 'yolov8n.yaml' for scratch)
EPOCHS, BATCH_SIZE, etc.
Run: python src/detection/train_detector.py
Results are saved in results/LicensePlateDet/.
b) Faster R-CNN

Modify parameters at the top of src/detection/train_faster_rcnn.py (dataset names, splits, NUM_EPOCHS, LEARNING_RATE).
Run: python src/detection/train_faster_rcnn.py
Checkpoints are saved in results/FasterRCNN_checkpoints/.
c) RetinaNet

Modify parameters at the top of src/detection/train_retinanet.py (dataset names, splits, NUM_EPOCHS, LEARNING_RATE).
Run: python src/detection/train_retinanet.py
Checkpoints are saved in results/RetinaNet_checkpoints/.
4. Evaluating Detection Models
a) Faster R-CNN

To run with specific arguments:
Bash

python src/detection/evaluate_faster_rcnn.py --model_checkpoint path/to/faster_rcnn_checkpoint.pth --dataset_name romanian --split valid
Or, modify the hardcoded config dictionary in src/detection/evaluate_faster_rcnn.py and run:
Bash

python src/detection/evaluate_faster_rcnn.py
Metrics are printed and saved in results/FasterRCNN_evaluation/{checkpoint_stem}/.
b) RetinaNet

To run with specific arguments:
Bash

python src/detection/evaluate_retinanet.py --model_checkpoint path/to/retinanet_checkpoint.pth --dataset_name spanish --split test
Or, modify the hardcoded config dictionary in src/detection/evaluate_retinanet.py and run:
Bash

python src/detection/evaluate_retinanet.py
Metrics are printed and saved in results/RetinaNet_evaluation/{checkpoint_stem}/.
(YOLOv8 evaluation metrics are available in the CSV files in results/LicensePlateDet/{run_name}/results.csv generated during training.)

5. Running OCR Experiments
Edit the experiments dictionary and selected_config_name in src/character_recognition/run_ocr_experiment.py.
Ensure detection_model_path points to the correct trained detector weights (.pt for YOLO, .pth for TorchVision models) if using crop_source: 'detector_output'.
Run: python src/character_recognition/run_ocr_experiment.py
JSON results and analysis text files are saved in results/OCR_Experiments/{experiment_name}/.
Dependencies (Example)
Python 3.9+
PyTorch (torch, torchvision)
Ultralytics
OpenCV-Python (opencv-python)
Pytesseract
pycocotools (or pycocotools-windows)
tqdm
NumPy
PyYAML
Pandas
Matplotlib
scikit-learn
Pillow