# src/data_utils/preprocess.py
import os
import cv2
import json
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm # Progress bar
import shutil
import pathlib # Import pathlib
import sys

print(f"Python executable: {sys.executable}") # Debug Print

# Import utility functions from the same directory
try:
    from utils import (
        resize_and_pad,
        scale_coordinates,
        scale_polygon_coordinates,
        polygon_to_bbox,
        convert_to_yolo_format
    )
except ImportError:
    print("ERROR: Could not import from utils.py. Make sure it's in the same directory (src/data_utils/).")
    sys.exit(1)


# --- Configuration ---
TARGET_SIZE = (640, 640) # Width, Height
CLASS_MAP = {'license_plate': 0}

# --- Robust Path Calculation using pathlib ---
# Get the directory where this script (preprocess.py) is located
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
# Get the project root directory (assuming this script is in ProjectRoot/src/data_utils/)
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()
# Define base paths relative to the project root
RAW_DATA_BASE = PROJECT_ROOT / 'data' / 'raw'
OUTPUT_IMAGE_BASE = PROJECT_ROOT / 'data' / 'processed' / 'images'
OUTPUT_LABEL_BASE = PROJECT_ROOT / 'data' / 'processed' / 'labels'
# --- End Path Calculation ---


# --- Helper Functions for Parsing (Keep as before) ---
def parse_romanian_annotation(xml_path):
    """ Parses Romanian VOC XML file. """
    boxes = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            if bndbox is not None:
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
    except ET.ParseError: print(f"Warning: Skipping corrupted XML file: {xml_path}")
    except Exception as e: print(f"Warning: Error parsing XML {xml_path}: {e}")
    return boxes

def parse_spanish_annotation_polygons(json_path):
    """ Parses Spanish JSON file and returns list of original polygons. """
    polygons = []
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if 'lps' in data and isinstance(data['lps'], list):
            for lp_annotation in data['lps']:
                if 'poly_coord' in lp_annotation and isinstance(lp_annotation['poly_coord'], list):
                    polygon_points = lp_annotation['poly_coord']
                    if polygon_points:
                        polygons.append(polygon_points)
    except json.JSONDecodeError: print(f"Warning: Skipping corrupted JSON file: {json_path}")
    except Exception as e: print(f"Warning: Error parsing JSON {json_path}: {e}")
    return polygons


# --- Core Processing Function ---
def process_split(dataset_name, split_name, raw_img_dir, raw_annot_info, output_img_dir):
    """
    Processes images and annotations for a split, saving processed images
    and returning structured annotation data relative to processed images.
    Args are expected to be pathlib.Path objects or lists.
    """
    # --- Debug Prints ---
    print(f"\n[Debug] process_split called for: {dataset_name} - {split_name}")
    print(f"[Debug] Checking raw_img_dir: {raw_img_dir}")
    print(f"[Debug] raw_annot_info type: {type(raw_annot_info)}")
    if isinstance(raw_annot_info, pathlib.Path):
         print(f"[Debug] Checking raw_annot_info path: {raw_annot_info}")
    else:
         print(f"[Debug] raw_annot_info list length: {len(raw_annot_info) if isinstance(raw_annot_info, list) else 'N/A'}")
    print(f"[Debug] Expecting output_img_dir: {output_img_dir}")
    # --- End Debug Prints ---

    os.makedirs(output_img_dir, exist_ok=True) # pathlib paths work here
    processed_data = []

    print(f"\nCore Processing {dataset_name} - {split_name} split...")

    items_to_process = []
    # --- Determine items to process ---
    if dataset_name == 'romanian':
        # --- Debug Prints ---
        print(f"[Debug] Listing files in: {raw_img_dir}")
        try:
            # Use pathlib's iterdir() for listing
            img_files_debug = list(raw_img_dir.iterdir())
            print(f"[Debug] Found {len(img_files_debug)} items in raw_img_dir (first 5): {[f.name for f in img_files_debug[:5]]}")
        except Exception as e:
            print(f"[Debug] ERROR listing directory {raw_img_dir}: {e}")
        # --- End Debug Prints ---
        try:
            img_files = [f for f in raw_img_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
            annot_dir = raw_annot_info # This is the path to annots dir (pathlib object)
            print(f"[Debug] Found {len(img_files)} potential image files.")
            for img_path in img_files: # img_path is now a Path object
                 base_name = img_path.stem # Get filename without extension
                 annot_path = annot_dir / f"{base_name}.xml" # Use pathlib / operator
                 # --- Debug Print ---
                 if img_files.index(img_path) < 5:
                      print(f"[Debug] Checking for annot: {annot_path} -> Exists: {annot_path.exists()}")
                 # --- End Debug Print ---
                 if annot_path.exists(): items_to_process.append({'id': base_name, 'img_path': img_path, 'annot_path': annot_path})

        except FileNotFoundError: print(f"[Debug] ERROR: os.listdir failed for raw_img_dir (FileNotFound): {raw_img_dir}")
        except Exception as e: print(f"[Debug] ERROR during Romanian item determination: {e}")

    elif dataset_name == 'spanish':
        split_ids = raw_annot_info
        for base_id in split_ids:
            img_name = f"{base_id}.jpg"
            annot_name = f"{base_id}.json"
            # raw_img_dir is train/ or test/ folder path (pathlib object)
            img_path = raw_img_dir / img_name
            annot_path = raw_img_dir / annot_name
            if img_path.exists() and annot_path.exists(): items_to_process.append({'id': base_id, 'img_path': img_path, 'annot_path': annot_path})
    # --- End determine items ---

    print(f"[Debug] Number of items to process: {len(items_to_process)}")

    for item in tqdm(items_to_process, desc=f"Processing {split_name}"):
        base_name = item['id']
        raw_img_path = item['img_path'] # pathlib object
        annot_path = item['annot_path'] # pathlib object

        # 1. Read Image (cv2 handles pathlib)
        image = cv2.imread(str(raw_img_path)) # Convert to string just to be safe
        if image is None: continue

        # 2. Resize and Pad Image
        image_processed, ratio, padding = resize_and_pad(image, TARGET_SIZE)
        processed_img_h, processed_img_w = image_processed.shape[:2]

        # 3. Save Processed Image to common location
        output_image_path = output_img_dir / f"{base_name}.jpg" # Use pathlib /
        cv2.imwrite(str(output_image_path), image_processed) # Convert to string for cv2

        # 4. Parse Annotations & Get Scaled Absolute BBoxes relative to processed image
        scaled_abs_bboxes = []
        if dataset_name == 'romanian':
            boxes_orig = parse_romanian_annotation(annot_path)
            for box_orig in boxes_orig:
                scaled_box = scale_coordinates(box_orig, ratio, padding)
                scaled_abs_bboxes.append(scaled_box)
        elif dataset_name == 'spanish':
            polygons_orig = parse_spanish_annotation_polygons(annot_path)
            for poly in polygons_orig:
                 scaled_poly = scale_polygon_coordinates(poly, ratio, padding)
                 bbox = polygon_to_bbox(scaled_poly)
                 if bbox:
                     scaled_abs_bboxes.append(bbox)

        # 5. Store processed info for this image
        image_data = {
            'img_path': str(output_image_path), # Store path as string
            'img_shape': (processed_img_w, processed_img_h),
            'annotations': []
        }
        for bbox in scaled_abs_bboxes:
             xmin, ymin, xmax, ymax = bbox
             xmin = max(0, min(processed_img_w, xmin))
             ymin = max(0, min(processed_img_h, ymin))
             xmax = max(0, min(processed_img_w, xmax))
             ymax = max(0, min(processed_img_h, ymax))
             if xmax > xmin and ymax > ymin:
                 image_data['annotations'].append({
                     'class_id': CLASS_MAP['license_plate'],
                     'bbox': [xmin, ymin, xmax, ymax]
                 })
        processed_data.append(image_data)

    print(f"Finished core processing for {len(processed_data)} images in {split_name}.")
    return processed_data


# --- Formatting Function ---
def save_yolo_labels(processed_data, output_label_dir_base): # output_label_dir_base is now ../../data/processed/labels
    """
    Generates YOLO label files from the processed data structure.
    Saves labels mirroring the image structure directly under output_label_dir_base.
    e.g., if image is images/romanian/train/im1.jpg, label is labels/romanian/train/im1.txt
    """
    print(f"\nGenerating YOLO labels...")
    count = 0
    if not processed_data:
        print("Warning: No processed data received to generate labels.")
        return

    for image_info in tqdm(processed_data, desc="Saving YOLO Labels"):
        processed_img_path_str = image_info['img_path']
        img_shape = image_info['img_shape']
        annotations = image_info['annotations']

        try:
            # Get image path relative to OUTPUT_IMAGE_BASE
            # e.g., romanian/train/img_001.jpg
            relative_img_path = pathlib.Path(processed_img_path_str).relative_to(OUTPUT_IMAGE_BASE)

            # Construct label path directly under output_label_dir_base
            # e.g., data/processed/labels / romanian/train / img_001.txt  <- NO '/yolo/' here
            label_path = output_label_dir_base / relative_img_path.with_suffix('.txt') # REMOVED /'yolo'

        except ValueError as e:
             print(f"\nError constructing label path for {processed_img_path_str}. Base dirs might be incompatible? Error: {e}")
             print(f"OUTPUT_IMAGE_BASE: {OUTPUT_IMAGE_BASE}")
             print(f"OUTPUT_LABEL_BASE: {output_label_dir_base}")
             continue

        # Create parent directory for the label file if it doesn't exist
        label_path.parent.mkdir(parents=True, exist_ok=True)

        yolo_lines = []
        for annot in annotations:
            yolo_str = convert_to_yolo_format(annot['bbox'], img_shape, annot['class_id'])
            if yolo_str:
                yolo_lines.append(yolo_str)

        if yolo_lines:
            with open(label_path, 'w') as f:
                f.write("\n".join(yolo_lines))
            count += 1

    print(f"Saved {count} YOLO label files in base directory: {output_label_dir_base.resolve()}")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Debug Prints ---
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Project root calculated: {PROJECT_ROOT}")
    print(f"Current Working Directory: {os.getcwd()}") # See where VS Code runs it from
    print(f"Raw data base calculated: {RAW_DATA_BASE} (Exists: {RAW_DATA_BASE.exists()})")
    print(f"Output image base calculated: {OUTPUT_IMAGE_BASE}")
    print(f"Output label base calculated: {OUTPUT_LABEL_BASE}")
    # --- End Debug Prints ---

    print("Starting data preprocessing...")
    all_processed_data = {}

    # --- Process Romanian Dataset ---
    print("\n=== Processing Romanian Dataset ===")
    ds_name_folder = 'romanian_dataset'
    ds_name_output = 'romanian'
    all_processed_data[ds_name_output] = {}
    romanian_raw_path = RAW_DATA_BASE / ds_name_folder
    romanian_output_img_path = OUTPUT_IMAGE_BASE / ds_name_output
    print(f"[Debug] Romanian raw base path: {romanian_raw_path}")

    # Process train split
    split = 'train'
    img_dir_in_ro_train = romanian_raw_path / split / 'images'
    annot_info_in_ro_train = romanian_raw_path / split / 'annots'
    img_dir_out_ro_train = romanian_output_img_path / split
    if img_dir_in_ro_train.exists() and annot_info_in_ro_train.exists():
        all_processed_data[ds_name_output][split] = process_split(
            dataset_name='romanian', split_name=split,
            raw_img_dir=img_dir_in_ro_train, # Pass pathlib object
            raw_annot_info=annot_info_in_ro_train, # Pass pathlib object
            output_img_dir=img_dir_out_ro_train # Pass pathlib object
        )
    else: print(f"ERROR: Could not find input folders for Romanian {split}: {img_dir_in_ro_train} or {annot_info_in_ro_train}")

    # Process validation split
    split = 'valid'
    img_dir_in_ro_valid = romanian_raw_path / split / 'images'
    annot_info_in_ro_valid = romanian_raw_path / split / 'annots'
    img_dir_out_ro_valid = romanian_output_img_path / split
    if img_dir_in_ro_valid.exists() and annot_info_in_ro_valid.exists():
        all_processed_data[ds_name_output][split] = process_split(
            dataset_name='romanian', split_name=split,
            raw_img_dir=img_dir_in_ro_valid,
            raw_annot_info=annot_info_in_ro_valid,
            output_img_dir=img_dir_out_ro_valid
        )
    else: print(f"ERROR: Could not find input folders for Romanian {split}: {img_dir_in_ro_valid} or {annot_info_in_ro_valid}")

    # --- Process Spanish Dataset ---
    print("\n=== Processing Spanish Dataset ===")
    ds_name_folder = 'spanish_dataset'
    ds_name_output = 'spanish'
    all_processed_data[ds_name_output] = {}
    spanish_raw_path = RAW_DATA_BASE / ds_name_folder
    spanish_output_img_path = OUTPUT_IMAGE_BASE / ds_name_output
    print(f"[Debug] Spanish raw base path: {spanish_raw_path}")

    # Read IDs
    train_txt_path = spanish_raw_path / 'train.txt'
    test_txt_path = spanish_raw_path / 'test.txt'
    print(f"[Debug] Checking for Spanish train.txt at: {train_txt_path} (Exists: {train_txt_path.exists()})") # Debug Print
    print(f"[Debug] Checking for Spanish test.txt at: {test_txt_path} (Exists: {test_txt_path.exists()})") # Debug Print
    spanish_train_ids = []
    spanish_test_ids = []
    if train_txt_path.exists():
         with open(train_txt_path, 'r') as f: spanish_train_ids = [line.strip() for line in f if line.strip()]
    else: print(f"ERROR: {train_txt_path} not found.")
    if test_txt_path.exists():
         with open(test_txt_path, 'r') as f: spanish_test_ids = [line.strip() for line in f if line.strip()]
    else: print(f"ERROR: {test_txt_path} not found.")

    # Process train split
    split = 'train'
    img_dir_in_es_train = spanish_raw_path / split
    img_dir_out_es_train = spanish_output_img_path / split
    if spanish_train_ids and img_dir_in_es_train.exists():
        all_processed_data[ds_name_output][split] = process_split(
            dataset_name='spanish', split_name=split,
            raw_img_dir=img_dir_in_es_train,
            raw_annot_info=spanish_train_ids,
            output_img_dir=img_dir_out_es_train
        )
    elif not img_dir_in_es_train.exists(): print(f"ERROR: Could not find input folder for Spanish {split}: {img_dir_in_es_train}")

    # Process test split
    split = 'test'
    img_dir_in_es_test = spanish_raw_path / split
    img_dir_out_es_test = spanish_output_img_path / split
    if spanish_test_ids and img_dir_in_es_test.exists():
        all_processed_data[ds_name_output][split] = process_split(
            dataset_name='spanish', split_name=split,
            raw_img_dir=img_dir_in_es_test,
            raw_annot_info=spanish_test_ids,
            output_img_dir=img_dir_out_es_test
        )
    elif not img_dir_in_es_test.exists(): print(f"ERROR: Could not find input folder for Spanish {split}: {img_dir_in_es_test}")

    # --- Generate YOLO Labels ---
    # Define the base output directory for labels (WITHOUT '/yolo')
    labels_output_dir = OUTPUT_LABEL_BASE # e.g., D:/.../IA2/data/processed/labels

    # Flatten the processed data (same as before)
    all_data_flat = []
    for ds_key in all_processed_data:
        if ds_key in all_processed_data:
             for split_key in all_processed_data[ds_key]:
                 if split_key in all_processed_data[ds_key] and isinstance(all_processed_data[ds_key][split_key], list):
                      all_data_flat.extend(all_processed_data[ds_key][split_key])

    # Call the save function with the corrected base path
    if all_data_flat:
         save_yolo_labels(all_data_flat, labels_output_dir) # Pass the base label dir
    else:
         print("\nNo processed data generated, skipping label saving.")


    print("\nPreprocessing finished.")
    print(f"Processed images saved in: {OUTPUT_IMAGE_BASE.resolve()}")
    # Update the final print message to reflect the correct label location
    print(f"YOLO labels saved in subdirs under: {labels_output_dir.resolve()}")