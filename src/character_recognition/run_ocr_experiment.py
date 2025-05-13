# src/character_recognition/run_ocr_experiment.py
import os
import pathlib
import json
import cv2
import yaml # PyYAML
from tqdm import tqdm
import sys
import re # Ensure re is imported
import torch

# Assuming utils_ocr.py and analyze_ocr_results.py are in the same directory
try:
    from utils_ocr import preprocess_plate_for_ocr, validate_plate_text, recognize_text_tesseract
    from analyze_ocr_results import analyze_ocr_results
except ImportError:
    print("ERROR: Could not import from utils_ocr.py or analyze_ocr_results.py. Make sure they are correctly placed.")
    sys.exit(1)

# Import YOLO only if we might use it
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    # This warning will be handled if 'detector_output' is chosen in config
    # print("Warning: 'ultralytics' package not found. 'detector_output' crop_source will not work if selected.")

# --- Tesseract Path (Global Setting - can be overridden by config) ---
import pytesseract
TESSERACT_CMD_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # EXAMPLE FOR WINDOWS
# TESSERACT_CMD_PATH = r'/usr/bin/tesseract' # EXAMPLE FOR LINUX
# TESSERACT_CMD_PATH = r'/opt/homebrew/bin/tesseract' # EXAMPLE FOR MACOS (ARM)

# Set it if the path exists, otherwise assume it's in PATH
if os.path.exists(TESSERACT_CMD_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH
# --- End Tesseract Path ---

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()


def load_config_from_dict(config_dict):
    """Validates and prepares a config dictionary."""
    if 'project_root' not in config_dict:
        config_dict['project_root'] = str(PROJECT_ROOT)
    # Ensure project_root is a Path object for internal use
    config_dict['project_root'] = pathlib.Path(config_dict['project_root'])
    return config_dict

def save_experiment_results(results_data, output_dir, json_filename="ocr_results.json"):
    """Saves the OCR results to a JSON file in the experiment directory."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / json_filename
    try:
        with open(file_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"\nOCR results saved to: {file_path}")
        return file_path
    except IOError as e:
        print(f"Error saving results to {file_path}: {e}")
        return None

def get_gt_text_map_for_spanish_split(gt_crop_dir_for_split):
    """Loads the ground_truth_texts.tsv into a dictionary."""
    gt_text_map = {}
    gt_text_file = gt_crop_dir_for_split / "ground_truth_texts.tsv"
    if gt_text_file.exists():
        with open(gt_text_file, 'r', encoding='utf-8') as f_gt:
            for line in f_gt:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    gt_text_map[parts[0]] = parts[1] # crop_filename -> text
    return gt_text_map

def ocr_and_validate_single_cropped_image(cropped_image_cv, # The actual image data of the crop
                                          config, # The experiment config dictionary
                                          original_full_image_name_stem, # Stem of the original full image
                                          crop_identifier, # e.g., "plate_0", "gt_crop_1", or just crop filename
                                          ground_truth_text=None): # Optional GT text
    """
    Takes an already cropped plate image (OpenCV format), preprocesses, OCRs, and validates.
    """
    if cropped_image_cv is None or cropped_image_cv.size == 0:
        return {
            "image_file_original": original_full_image_name_stem,
            "cropped_plate_identifier": crop_identifier,
            "error": "Empty crop received"
        }

    preprocessed_for_ocr = preprocess_plate_for_ocr(cropped_image_cv)
    if preprocessed_for_ocr is None:
        return {
            "image_file_original": original_full_image_name_stem,
            "cropped_plate_identifier": crop_identifier,
            "error": "Preprocessing failed"
        }

    recognized_text_raw = ""
    ocr_engine_type = config.get('ocr_engine') # Use .get for safety

    if ocr_engine_type == 'tesseract':
        recognized_text_raw = recognize_text_tesseract(
            preprocessed_for_ocr,
            config.get('tesseract_config', "--oem 3 --psm 7"),
            config.get('tesseract_whitelist')
        )
    # Add elif blocks here for other OCR engines (e.g., 'easyocr', 'custom_model')
    else:
        print(f"Warning: OCR engine '{ocr_engine_type}' not handled or config is malformed.")
        return {
            "image_file_original": original_full_image_name_stem,
            "cropped_plate_identifier": crop_identifier,
            "error": f"Unknown or unhandled OCR engine: {ocr_engine_type}"
        }

    cleaned_text_for_validation = re.sub(r'[\s\-]', '', recognized_text_raw).upper()
    is_valid_format, validation_message = validate_plate_text(
        cleaned_text_for_validation,
        config.get('plate_format_validation', "RO")
    )
    
    entry = {
        "image_file_original": original_full_image_name_stem,
        "cropped_plate_identifier": crop_identifier,
        "recognized_text_raw": recognized_text_raw,
        "recognized_text_cleaned": cleaned_text_for_validation,
        "is_valid_format": is_valid_format,
        "validation_message": validation_message,
        "validation_schema": config.get('plate_format_validation', "RO")
    }

    if ground_truth_text is not None:
        entry["ground_truth_text"] = ground_truth_text
        entry["is_exact_match"] = (cleaned_text_for_validation == re.sub(r'[\s\-]', '', ground_truth_text).upper())
    
    return entry

def run_ocr_core(config): # This is the main experiment runner
    """Runs a single OCR experiment based on the provided configuration dictionary."""
    print(f"\n--- Starting OCR Experiment: {config['experiment_name']} ---")

    project_root = config['project_root'] # This is now a pathlib.Path object
    output_dir_base_str = config.get("results_base_dir", "results/OCR_Experiments")
    output_dir = project_root / output_dir_base_str / config['experiment_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    all_ocr_results = []

    if config['crop_source'] == 'ground_truth':
        gt_crop_base_dir = project_root / 'data' / 'ocr_input_data' / 'ground_truth_crops'
        input_crop_dir = gt_crop_base_dir / config['dataset_name'] / config['split']
        print(f"Using Ground Truth crops from: {input_crop_dir}")

        if not input_crop_dir.exists() or not input_crop_dir.is_dir():
            print(f"ERROR: Ground truth crop directory not found: {input_crop_dir}")
            print("Please run 'src/character_recognition/preprocess_ocr_data.py' first.")
            return None

        gt_text_map_for_split = {}
        if config['dataset_name'] == 'spanish':
            gt_text_map_for_split = get_gt_text_map_for_spanish_split(input_crop_dir)

        image_extensions = ["*.jpg", "*.jpeg", "*.png"]
        cropped_image_files = [p for ext in image_extensions for p in input_crop_dir.glob(ext)]

        if not cropped_image_files:
            print(f"No cropped images found in {input_crop_dir}")
            return None
        
        print(f"Found {len(cropped_image_files)} ground truth cropped images to process...")
        for crop_path in tqdm(cropped_image_files, desc="Processing GT Crops"):
            crop_cv = cv2.imread(str(crop_path))
            # original_image_name_stem is the part of the crop filename before "_plate_X"
            original_image_name_stem = crop_path.stem.split('_plate_')[0]
            
            gt_text = None
            if config['dataset_name'] == 'spanish':
                 gt_text = gt_text_map_for_split.get(crop_path.name)
            
            result_entry = ocr_and_validate_single_cropped_image(
                crop_cv, config, original_image_name_stem, crop_path.name, gt_text
            )
            if result_entry:
                result_entry["source_type"] = "ground_truth_crop"
                all_ocr_results.append(result_entry)

    elif config['crop_source'] == 'detector_output':
        if not YOLO_AVAILABLE:
            print("ERROR: 'ultralytics' (YOLO) package is not installed, but config requires 'detector_output'.")
            return None
            
        detector_model_path_str = config.get('detection_model_path')
        if not detector_model_path_str:
            print("ERROR: 'detection_model_path' not specified in config for 'detector_output' mode.")
            return None
        
        detector_model_path = project_root / detector_model_path_str
        if not detector_model_path.exists():
            print(f"ERROR: Detection model not found at {detector_model_path}")
            return None
            
        print(f"Loading detection model: {detector_model_path}")
        detector_model = YOLO(str(detector_model_path))
        detector_conf = float(config.get('detector_confidence_threshold', 0.25))

        full_image_dir = project_root / 'data' / 'processed' / 'images' / config['dataset_name'] / config['split']
        print(f"Running detector on full images from: {full_image_dir}")

        if not full_image_dir.exists():
            print(f"ERROR: Full image directory for detection not found: {full_image_dir}")
            return None

        image_extensions = ["*.jpg", "*.jpeg", "*.png"]
        full_image_paths = [p for ext in image_extensions for p in full_image_dir.glob(ext)]
        
        if not full_image_paths:
            print(f"No images found in {full_image_dir} to run detection on.")
            return None

        print(f"Found {len(full_image_paths)} full images for detection and OCR...")
        for full_img_path in tqdm(full_image_paths, desc="Detecting and OCRing"):
            detection_results_list = detector_model.predict(
                source=str(full_img_path), conf=detector_conf, 
                device=0 if torch.cuda.is_available() else 'cpu', verbose=False
            )
            
            if not detection_results_list or not detection_results_list[0].boxes.xyxy.numel() > 0:
                all_ocr_results.append({
                    "image_file_original": full_img_path.name, # Use .name for consistency
                    "source_type": "detector_output",
                    "error": "No detection"
                })
                continue

            detection_results = detection_results_list[0]
            original_image_cv = detection_results.orig_img # This is the 640x640 processed image
            boxes_xyxy_tensor = detection_results.boxes.xyxy

            if boxes_xyxy_tensor.numel() == 0:
                all_ocr_results.append({
                    "image_file_original": full_img_path.name,
                    "source_type": "detector_output",
                    "error": "No valid boxes in detection"
                })
                continue
            
            boxes_xyxy_list = boxes_xyxy_tensor.cpu().numpy().tolist() # Iterate over list of boxes

            for i, box in enumerate(boxes_xyxy_list):
                x1, y1, x2, y2 = map(int, box)
                img_h, img_w = original_image_cv.shape[:2]
                x1_c, y1_c, x2_c, y2_c = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

                if x1_c >= x2_c or y1_c >= y2_c: continue
                cropped_plate_cv = original_image_cv[y1_c:y2_c, x1_c:x2_c]
                
                crop_id_suffix = f"det_plate_{i}" # Identifier for this specific detected crop
                result_entry = ocr_and_validate_single_cropped_image(
                    cropped_plate_cv, config, full_img_path.stem, crop_id_suffix
                ) # GT text will be None here

                if result_entry:
                    result_entry["source_type"] = "detector_output"
                    result_entry["detector_model_used"] = detector_model_path.name
                    result_entry["box_coordinates_on_640x640_image"] = [x1,y1,x2,y2] # Store the detected box
                    all_ocr_results.append(result_entry)
    else:
        print(f"Error: Invalid 'crop_source': {config['crop_source']}")
        return None

    results_json_filename = config.get("results_json_filename", "ocr_results.json")
    saved_results_path = save_experiment_results(all_ocr_results, output_dir, results_json_filename)
    
    print(f"--- Experiment {config['experiment_name']} finished ---")
    
    if saved_results_path and os.path.exists(project_root / "src/character_recognition/analyze_ocr_results.py"):
        print("\nRunning analysis on the results...")
        try:
            analyze_ocr_results(saved_results_path) # Call directly
        except Exception as e_analyze:
            print(f"Could not run analysis script automatically: {e_analyze}")
            print(f"You can run it manually: python src/character_recognition/analyze_ocr_results.py --results_file \"{saved_results_path}\"")

    return saved_results_path


if __name__ == "__main__":
    # --- Define Your Experiments Here ---
    experiments = {
        "exp_ro_gt_valid_tesseract": { # Process ground truth crops from Romanian validation set
            "project_root": str(PROJECT_ROOT),
            "experiment_name": "RO_GT_Tesseract_Valid",
            "dataset_name": "romanian", "split": "valid",
            "crop_source": "ground_truth",
            "ocr_engine": "tesseract",
            "tesseract_whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "plate_format_validation": "RO",
        },
        "exp_es_gt_test_tesseract": { # Process ground truth crops from Spanish test set
            "project_root": str(PROJECT_ROOT),
            "experiment_name": "ES_GT_Tesseract_Test",
            "dataset_name": "spanish", "split": "test",
            "crop_source": "ground_truth",
            "ocr_engine": "tesseract",
            "tesseract_whitelist": "BCDFGHJKLMNPQRSTUVWXYZ0123456789",
            "plate_format_validation": "ES"
        },
        "exp_ro_yolov8s_det_valid_tesseract": { # Run detector on Romanian validation set & OCR crops
            "project_root": str(PROJECT_ROOT),
            "experiment_name": "RO_YOLOv8s_Tesseract_Valid_DetectorCrops",
            "dataset_name": "romanian", "split": "valid",
            "crop_source": "detector_output",
            "detection_model_path": "results/LicensePlateDet/yolov8s_yolo_romanian_lp_e50/weights/best.pt", # Make sure this path is correct!
            "detector_confidence_threshold": 0.4,
            "ocr_engine": "tesseract",
            "tesseract_whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "plate_format_validation": "RO"
        },
        "exp_es_yolov8s_det_test_tesseract": { # Run detector on Spanish test set & OCR crops
            "project_root": str(PROJECT_ROOT),
            "experiment_name": "ES_YOLOv8s_Tesseract_Test_DetectorCrops",
            "dataset_name": "spanish", "split": "test",
            "crop_source": "detector_output",
            "detection_model_path": "results/LicensePlateDet/yolov8s_yolo_spanish_lp_e50/weights/best.pt", # UPDATE 'X' to your Spanish run ID
            "detector_confidence_threshold": 0.4,
            "ocr_engine": "tesseract",
            "tesseract_whitelist": "BCDFGHJKLMNPQRSTUVWXYZ0123456789", # Adjust if needed for Spanish
            "plate_format_validation": "ES"
        },
        "exp_es_yolov8s_det_train_tesseract": { # Run detector on Spanish train set & OCR crops
            "project_root": str(PROJECT_ROOT),
            "experiment_name": "ES_YOLOv8s_Tesseract_train_DetectorCrops",
            "dataset_name": "spanish", "split": "train",
            "crop_source": "detector_output",
            "detection_model_path": "results/LicensePlateDet/yolov8s_yolo_spanish_lp_e50/weights/best.pt", # UPDATE 'X' to your Spanish run ID
            "detector_confidence_threshold": 0.4,
            "ocr_engine": "tesseract",
            "tesseract_whitelist": "BCDFGHJKLMNPQRSTUVWXYZ0123456789", # Adjust if needed for Spanish
            "plate_format_validation": "ES"
        },
            # Add this to your 'experiments' dictionary
        "exp_ro_yolov8n_scratch_det_valid_tesseract": {
            "project_root": str(PROJECT_ROOT),
            "experiment_name": "RO_YOLOv8nScratch_DetectorCrops_Valid_Tesseract",
            "dataset_name": "romanian",  # Full images for detection are from Romanian
            "split": "valid",           # Use the validation split of Romanian full images
            "crop_source": "detector_output",
            # !!! UPDATE THIS PATH to your actual 'from scratch' Romanian model !!!
            "detection_model_path": "results/LicensePlateDet/yolov8n_yolo_romanian_lp_e50_train_0/weights/best.pt", 
            "detector_confidence_threshold": 0.25, # You can adjust this
            "ocr_engine": "tesseract",
            "tesseract_whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "plate_format_validation": "RO"
        },
        # Add this to your 'experiments' dictionary
        "exp_ro_yolov8s_transfer_ES_to_RO_det_valid_tesseract": {
            "project_root": str(PROJECT_ROOT),
            "experiment_name": "RO_YOLOv8s_Transfer_ES_to_RO_DetectorCrops_Valid_Tesseract",
            "dataset_name": "romanian", # Full images for detection are from Romanian
            "split": "valid",          # Use the validation split of Romanian full images
            "crop_source": "detector_output",
            # !!! UPDATE THIS PATH to your actual ES->RO transfer model !!!
            # Example: "results/LicensePlateDet/yolov8s_ES_to_RO_transfer_e30/weights/best.pt"
            "detection_model_path": "results/LicensePlateDet/best_yolo_romanian_lp_e30_transfer/weights/best.pt", 
            "detector_confidence_threshold": 0.25,
            "ocr_engine": "tesseract",
            "tesseract_whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "plate_format_validation": "RO"
        }
    }

    # --- Select and Run Experiment ---
    # Change this key to run a different experiment
    # selected_config_name = "exp_ro_gt_valid_tesseract" # Start with GT crops
    # selected_config_name = "exp_es_gt_test_tesseract"
    # selected_config_name = "exp_ro_yolov8s_det_valid_tesseract" # Test this after RO YOLOv8s training
    selected_config_name = "exp_es_yolov8s_det_train_tesseract" # Test this after ES YOLOv8s training

    if selected_config_name in experiments:
        config_dict_selected = experiments[selected_config_name]
        
        # --- Tesseract Path Check ---
        if config_dict_selected.get('ocr_engine') == 'tesseract':
            try:
                tesseract_cmd_path_from_config = config_dict_selected.get('tesseract_cmd_path')
                if tesseract_cmd_path_from_config and os.path.exists(tesseract_cmd_path_from_config):
                     pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path_from_config
                     print(f"Using Tesseract CMD from experiment config: {tesseract_cmd_path_from_config}")
                elif 'TESSERACT_CMD_PATH' in globals() and TESSERACT_CMD_PATH and os.path.exists(TESSERACT_CMD_PATH):
                     pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH
                     print(f"Using Tesseract CMD from global script setting: {TESSERACT_CMD_PATH}")
                
                version = pytesseract.get_tesseract_version()
                print(f"Tesseract version {version} found and accessible.")
            except pytesseract.TesseractNotFoundError:
                print("Tesseract is not installed or not found in your PATH, and not specified in config or globally.")
                sys.exit(1)
            except Exception as e:
                print(f"An error occurred with Tesseract setup: {e}")
                sys.exit(1)
        # --- End Tesseract Path Check ---

        final_config = load_config_from_dict(config_dict_selected)
        run_ocr_core(final_config)
    else:
        print(f"Error: Experiment '{selected_config_name}' not defined.")
        print("Available experiments:", list(experiments.keys()))