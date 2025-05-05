# src/data_utils/create_coco_labels.py
import os
import pathlib
import pickle
import json
import datetime
from tqdm import tqdm
import sys

# --- Configuration ---
# Define paths relative to this script file using pathlib
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve() # Assumes script is in ProjectRoot/src/data_utils/
# Path to the intermediate data saved by preprocess.py
INTERMEDIATE_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'intermediate' / 'processed_annotations.pkl'
# Base path where processed images were saved (needed to construct relative file_name)
OUTPUT_IMAGE_BASE = PROJECT_ROOT / 'data' / 'processed' / 'images'
# Base path where COCO JSON label files will be saved
OUTPUT_COCO_LABEL_DIR = PROJECT_ROOT / 'data' / 'processed' / 'labels' / 'coco'

# Define COCO categories
# IMPORTANT: COCO category IDs typically start from 1 (unlike YOLO which starts from 0)
# We map our internal class_id 0 ('license_plate') to COCO category_id 1
COCO_CATEGORIES = [{"id": 1, "name": "license_plate", "supercategory": "vehicle_part"}]
INTERNAL_TO_COCO_CAT_ID = {0: 1} # Mapping from CLASS_MAP index used in intermediate file to COCO ID

# --- Formatting Function ---
def save_coco_labels(processed_data_split, output_json_path, categories):
    """
    Generates a COCO JSON annotation file from the processed data for a specific split.

    Args:
        processed_data_split (list): List of dicts for ONE specific split, loaded from intermediate data.
        output_json_path (pathlib.Path): The full path where the output JSON file should be saved.
        categories (list): List of COCO category dictionaries.
    """
    print(f"\nGenerating COCO JSON for: {output_json_path.name}...")
    if not processed_data_split:
        print(f"Warning: No processed data provided for {output_json_path.name}, skipping.")
        return

    # Basic COCO structure
    coco_output = {
        "info": {
            "description": f"License Plate Dataset - {output_json_path.stem}",
            "url": "", "version": "1.0", "year": datetime.datetime.utcnow().year,
            "contributor": "IA2 Project", "date_created": datetime.datetime.utcnow().isoformat(' ')
        },
        "licenses": [{"id": 1, "name": "Placeholder License", "url": ""}],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    img_id_counter = 1
    annot_id_counter = 1

    print(f"Processing {len(processed_data_split)} images for {output_json_path.name}...")

    for image_info in tqdm(processed_data_split, desc=f"Converting {output_json_path.stem} to COCO"):
        # Ensure expected keys exist
        if 'img_path' not in image_info or 'img_shape' not in image_info or 'annotations' not in image_info:
            print(f"Warning: Skipping malformed image_info entry: {image_info.get('img_path', 'Path missing')}")
            continue

        img_w, img_h = image_info['img_shape']
        relative_img_path_str = image_info['img_path'] # Should be relative like 'romanian/train/img.jpg'

        # Add image entry
        coco_output["images"].append({
            "id": img_id_counter,
            "width": img_w,
            "height": img_h,
            "file_name": relative_img_path_str, # Store the relative path directly
            "license": 1,
            "date_captured": ""
        })

        # Add annotation entries for this image
        for annot in image_info['annotations']:
            if 'bbox' not in annot or 'class_id' not in annot:
                 print(f"Warning: Skipping malformed annotation in {relative_img_path_str}: {annot}")
                 continue

            xmin, ymin, xmax, ymax = annot['bbox']
            width = xmax - xmin
            height = ymax - ymin

            # Skip if width or height are non-positive
            if width <= 0 or height <= 0:
                 print(f"Warning: Skipping annotation with non-positive width/height in {relative_img_path_str}: bbox={annot['bbox']}")
                 continue

            area = float(width * height)
            internal_class_id = annot['class_id']
            category_id = INTERNAL_TO_COCO_CAT_ID.get(internal_class_id) # Map internal ID (0) to COCO ID (1)

            if category_id is None:
                print(f"Warning: Skipping annotation with unknown internal class_id {internal_class_id} in {relative_img_path_str}")
                continue

            coco_output["annotations"].append({
                "id": annot_id_counter,
                "image_id": img_id_counter, # Corresponds to the ID in the 'images' list
                "category_id": category_id, # COCO category ID (e.g., 1)
                "bbox": [float(xmin), float(ymin), float(width), float(height)], # COCO format: [xmin, ymin, width, height]
                "area": area,
                "segmentation": [], # No segmentation masks
                "iscrowd": 0
            })
            annot_id_counter += 1

        img_id_counter += 1

    # Ensure output directory exists
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the COCO JSON file
    print(f"Attempting to save {len(coco_output['annotations'])} annotations for {len(coco_output['images'])} images...")
    try:
        with open(output_json_path, 'w') as f:
            json.dump(coco_output, f, indent=4)
        print(f"Saved COCO JSON to: {output_json_path}")
    except Exception as e:
        print(f"ERROR saving COCO JSON to {output_json_path}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Attempting to load intermediate data from: {INTERMEDIATE_DATA_PATH}")
    if not INTERMEDIATE_DATA_PATH.exists():
        print(f"ERROR: Intermediate data file not found at {INTERMEDIATE_DATA_PATH}.")
        print(f"Please run preprocess.py first to generate it.")
        sys.exit(1)

    try:
        with open(INTERMEDIATE_DATA_PATH, 'rb') as f:
            # Load the nested dictionary structure {dataset: {split: [data]}}
            all_processed_data = pickle.load(f)
        print("Intermediate data loaded successfully.")
    except Exception as e:
        print(f"ERROR loading intermediate data: {e}")
        sys.exit(1)

    # Ensure the base COCO label directory exists
    OUTPUT_COCO_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ensured COCO output directory exists: {OUTPUT_COCO_LABEL_DIR}")

    # Generate one JSON file per original split
    for ds_key in all_processed_data: # e.g., 'romanian', 'spanish'
        if ds_key in all_processed_data:
            for split_key in all_processed_data[ds_key]: # e.g., 'train', 'valid', 'test'
                split_data = all_processed_data[ds_key][split_key]
                if split_data: # Check if list is not empty
                    json_filename = f"{ds_key}_{split_key}.json" # e.g., romanian_train.json
                    output_json_path = OUTPUT_COCO_LABEL_DIR / json_filename
                    # Call the save function for this specific split's data
                    save_coco_labels(split_data, output_json_path, COCO_CATEGORIES)
                else:
                     print(f"Skipping COCO generation for empty split: {ds_key} / {split_key}")

    print("\nCOCO label generation finished.")