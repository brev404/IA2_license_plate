# src/data_utils/preprocess.py
import os
import cv2
import json
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm # Progress bar
import shutil
import pathlib # Use pathlib for robust path handling
import sys
import pickle # To save intermediate data

print(f"Python executable: {sys.executable}")

# Import utility functions from utils.py (make sure utils.py exists in this directory)
try:
    from utils import (
        resize_and_pad,
        scale_coordinates,
        scale_polygon_coordinates,
        polygon_to_bbox
        # NOTE: convert_to_yolo_format is NOT needed in this script anymore
    )
except ImportError:
    print("ERROR: Could not import from utils.py. Make sure it's in the same directory (src/data_utils/).")
    sys.exit(1)


# --- Configuration ---
TARGET_SIZE = (640, 640) # Width, Height
CLASS_MAP = {'license_plate': 0} # Define your class map (index 0 for the single class)

# Construct paths relative to this script file using pathlib
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve() # Assumes script is in ProjectRoot/src/data_utils/
RAW_DATA_BASE = PROJECT_ROOT / 'data' / 'raw'
OUTPUT_IMAGE_BASE = PROJECT_ROOT / 'data' / 'processed' / 'images'
INTERMEDIATE_DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'intermediate'
# --- End Path Calculation ---


# --- Helper Functions for Parsing ---
def parse_romanian_annotation(xml_path):
    """ Parses Romanian VOC XML file. Returns list of [xmin, ymin, xmax, ymax]. """
    boxes = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for member in root.findall('object'):
            # Assuming class is always license plate for this dataset
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
    """ Parses Spanish JSON file and returns list of original polygons [[x,y],...]. """
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
    print(f"\n[Debug] process_split called for: {dataset_name} - {split_name}")
    print(f"[Debug] Checking raw_img_dir: {raw_img_dir}")
    print(f"[Debug] raw_annot_info type: {type(raw_annot_info)}")
    if isinstance(raw_annot_info, pathlib.Path):
         print(f"[Debug] Checking raw_annot_info path: {raw_annot_info}")
    else:
         print(f"[Debug] raw_annot_info list length: {len(raw_annot_info) if isinstance(raw_annot_info, list) else 'N/A'}")
    print(f"[Debug] Expecting output_img_dir: {output_img_dir}")

    os.makedirs(output_img_dir, exist_ok=True)
    processed_data_for_split = [] # Store results for this split

    print(f"\nCore Processing {dataset_name} - {split_name} split...")

    items_to_process = []
    # --- Determine items to process ---
    if dataset_name == 'romanian':
        try:
            img_files = [f for f in raw_img_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
            annot_dir = raw_annot_info # Path object
            for img_path in img_files:
                 base_name = img_path.stem
                 annot_path = annot_dir / f"{base_name}.xml"
                 if annot_path.exists(): items_to_process.append({'id': base_name, 'img_path': img_path, 'annot_path': annot_path})
        except Exception as e: print(f"[Debug] ERROR during Romanian item determination: {e}")

    elif dataset_name == 'spanish':
        split_ids = raw_annot_info
        for base_id in split_ids:
            img_name = f"{base_id}.jpg"
            annot_name = f"{base_id}.json"
            img_path = raw_img_dir / img_name # raw_img_dir is train/ or test/
            annot_path = raw_img_dir / annot_name
            if img_path.exists() and annot_path.exists(): items_to_process.append({'id': base_id, 'img_path': img_path, 'annot_path': annot_path})
    # --- End determine items ---

    print(f"[Debug] Number of items to process: {len(items_to_process)}")

    for item in tqdm(items_to_process, desc=f"Processing {split_name}"):
        base_name = item['id']
        raw_img_path = item['img_path']
        annot_path = item['annot_path']

        # 1. Read Image
        image = cv2.imread(str(raw_img_path))
        if image is None:
            print(f"Warning: Failed to load image {raw_img_path}, skipping.")
            continue

        # 2. Resize and Pad Image
        image_processed, ratio, padding = resize_and_pad(image, TARGET_SIZE)
        processed_img_h, processed_img_w = image_processed.shape[:2]

        # 3. Save Processed Image to common location
        output_image_path = output_img_dir / f"{base_name}.jpg"
        try:
             cv2.imwrite(str(output_image_path), image_processed)
        except Exception as e_write:
             print(f"Warning: Failed to write processed image {output_image_path}: {e_write}")
             continue # Skip if image can't be saved

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
        # Use relative path from OUTPUT_IMAGE_BASE for portability if possible
        try:
             relative_output_img_path = output_image_path.relative_to(OUTPUT_IMAGE_BASE)
             img_path_to_store = str(relative_output_img_path).replace('\\', '/') # Store POSIX style
        except ValueError:
             img_path_to_store = output_image_path.name # Fallback to just the name

        image_data = {
            # Store relative path if possible, makes intermediate data more portable
            'img_path': img_path_to_store,
            'img_shape': (processed_img_w, processed_img_h), # Store as W, H
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
                     'class_id': CLASS_MAP['license_plate'], # Use defined class map
                     'bbox': [xmin, ymin, xmax, ymax] # Store absolute coords
                 })
        processed_data_for_split.append(image_data)

    print(f"Finished core processing for {len(processed_data_for_split)} images in {split_name}.")
    return processed_data_for_split


# --- Main Execution ---
if __name__ == "__main__":
    # --- Debug Prints ---
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Project root calculated: {PROJECT_ROOT}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Raw data base calculated: {RAW_DATA_BASE} (Exists: {RAW_DATA_BASE.exists()})")
    print(f"Output image base calculated: {OUTPUT_IMAGE_BASE}")
    # --- End Debug Prints ---

    print("\nStarting data preprocessing (Images & Intermediate Annotations)...")
    all_processed_data = {} # Store processed data for all splits/datasets

    # --- Process Romanian Dataset ---
    print("\n=== Processing Romanian Dataset ===")
    ds_name_folder_ro = 'romanian_dataset'; ds_name_output_ro = 'romanian'
    all_processed_data[ds_name_output_ro] = {}
    romanian_raw_path = RAW_DATA_BASE / ds_name_folder_ro
    romanian_output_img_path = OUTPUT_IMAGE_BASE / ds_name_output_ro

    # Process train split
    split = 'train'
    img_dir_in_ro_train = romanian_raw_path / split / 'images'; annot_info_in_ro_train = romanian_raw_path / split / 'annots'
    img_dir_out_ro_train = romanian_output_img_path / split
    if img_dir_in_ro_train.exists() and annot_info_in_ro_train.exists():
        all_processed_data[ds_name_output_ro][split] = process_split(
            'romanian', split, img_dir_in_ro_train, annot_info_in_ro_train, img_dir_out_ro_train)
    else: print(f"ERROR: Could not find input folders for Romanian {split}")

    # Process validation split
    split = 'valid'
    img_dir_in_ro_valid = romanian_raw_path / split / 'images'; annot_info_in_ro_valid = romanian_raw_path / split / 'annots'
    img_dir_out_ro_valid = romanian_output_img_path / split
    if img_dir_in_ro_valid.exists() and annot_info_in_ro_valid.exists():
        all_processed_data[ds_name_output_ro][split] = process_split(
            'romanian', split, img_dir_in_ro_valid, annot_info_in_ro_valid, img_dir_out_ro_valid)
    else: print(f"ERROR: Could not find input folders for Romanian {split}")


    # --- Process Spanish Dataset ---
    print("\n=== Processing Spanish Dataset ===")
    ds_name_folder_es = 'spanish_dataset'; ds_name_output_es = 'spanish'
    all_processed_data[ds_name_output_es] = {}
    spanish_raw_path = RAW_DATA_BASE / ds_name_folder_es
    spanish_output_img_path = OUTPUT_IMAGE_BASE / ds_name_output_es

    # Read IDs
    train_txt_path = spanish_raw_path / 'train.txt'; test_txt_path = spanish_raw_path / 'test.txt'
    spanish_train_ids = []; spanish_test_ids = []
    if train_txt_path.exists():
         with open(train_txt_path, 'r') as f: spanish_train_ids = [line.strip() for line in f if line.strip()]
    else: print(f"ERROR: {train_txt_path} not found.")
    if test_txt_path.exists():
         with open(test_txt_path, 'r') as f: spanish_test_ids = [line.strip() for line in f if line.strip()]
    else: print(f"ERROR: {test_txt_path} not found.")

    # Process train split
    split = 'train'
    img_dir_in_es_train = spanish_raw_path / split; img_dir_out_es_train = spanish_output_img_path / split
    if spanish_train_ids and img_dir_in_es_train.exists():
        all_processed_data[ds_name_output_es][split] = process_split(
            'spanish', split, img_dir_in_es_train, spanish_train_ids, img_dir_out_es_train)
    elif not img_dir_in_es_train.exists(): print(f"ERROR: Could not find input folder for Spanish {split}: {img_dir_in_es_train}")

    # Process test split
    split = 'test'
    img_dir_in_es_test = spanish_raw_path / split; img_dir_out_es_test = spanish_output_img_path / split
    if spanish_test_ids and img_dir_in_es_test.exists():
        all_processed_data[ds_name_output_es][split] = process_split(
            'spanish', split, img_dir_in_es_test, spanish_test_ids, img_dir_out_es_test)
    elif not img_dir_in_es_test.exists(): print(f"ERROR: Could not find input folder for Spanish {split}: {img_dir_in_es_test}")


    # --- Save Intermediate Processed Data ---
    INTERMEDIATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    intermediate_data_path = INTERMEDIATE_DATA_DIR / 'processed_annotations.pkl'

    print(f"\nSaving intermediate processed annotation data to: {intermediate_data_path}")
    try:
        with open(intermediate_data_path, 'wb') as f:
            pickle.dump(all_processed_data, f)
        print("Intermediate data saved successfully.")
    except Exception as e:
        print(f"ERROR saving intermediate data: {e}")
    # --- End Saving Intermediate Data ---

    print("\nPreprocessing finished.")
    print(f"Processed images saved in: {OUTPUT_IMAGE_BASE.resolve()}")
    print(f"Intermediate annotation data saved in: {INTERMEDIATE_DATA_DIR.resolve()}")