# src/character_recognition/preprocess_ocr_data.py
import os
import pathlib
import pickle
import cv2
from tqdm import tqdm
import shutil # shutil might not be needed here unless copying files
import sys
import json # For loading Spanish raw JSON

# Ensure src/data_utils is in path to import from utils.py there if needed for paths
SCRIPT_DIR_OCR_PREPROCESS = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT_OCR_PREPROCESS = SCRIPT_DIR_OCR_PREPROCESS.parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT_OCR_PREPROCESS / 'src' / 'data_utils'))

try:
    # These are from the main detection preprocessing utils.py
    # We don't directly need them if we use the scaled bboxes from the pickle file
    # from utils import scale_polygon_coordinates, polygon_to_bbox
    pass # No specific imports needed from data_utils.utils for this script currently
except ImportError:
    print("Warning: Could not import from src/data_utils/utils.py. Not strictly needed if using pickled bboxes.")
    # sys.exit(1) # Not critical for this script's core logic

# --- Configuration ---
INTERMEDIATE_DATA_PATH = PROJECT_ROOT_OCR_PREPROCESS / 'data' / 'processed' / 'intermediate' / 'processed_annotations.pkl'
# Base path where processed (640x640) images were saved by the main preprocess.py
PROCESSED_IMAGE_BASE = PROJECT_ROOT_OCR_PREPROCESS / 'data' / 'processed' / 'images' # <<< THIS WAS MISSING
GT_CROPS_OUTPUT_BASE = PROJECT_ROOT_OCR_PREPROCESS / 'data' / 'ocr_input_data' / 'ground_truth_crops'

# For Spanish dataset, to read original JSON and get ground truth text ('lp_id')
SPANISH_RAW_DATA_BASE = PROJECT_ROOT_OCR_PREPROCESS / 'data' / 'raw' / 'spanish_dataset'

def create_ground_truth_crops():
    print(f"Loading intermediate annotation data from: {INTERMEDIATE_DATA_PATH}")
    if not INTERMEDIATE_DATA_PATH.exists():
        print(f"ERROR: Intermediate data file not found at {INTERMEDIATE_DATA_PATH}.")
        print(f"Please run the main preprocess.py (in src/data_utils) first.")
        return

    try:
        with open(INTERMEDIATE_DATA_PATH, 'rb') as f:
            all_processed_data = pickle.load(f) # This is {dataset_name: {split_name: [image_infos]}}
        print("Intermediate data loaded successfully.")
    except Exception as e:
        print(f"ERROR loading intermediate data: {e}")
        return

    # --- Helper to get Spanish GT Text ---
    def get_spanish_gt_text_from_raw(base_id, split_name):
        # base_id is like '00001', split_name is 'train' or 'test'
        json_path = SPANISH_RAW_DATA_BASE / split_name / f"{base_id}.json"
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f_json: # Added encoding
                    data = json.load(f_json)
                # The intermediate data from preprocess.py creates one entry per image,
                # but one image might have multiple plates (lps).
                # We need to decide how to map crops back to the correct lp_id if there are multiple.
                # For now, let's assume the order of annotations in the pickle file's
                # 'annotations' list for an image matches the order in the 'lps' list of the raw JSON.
                # This might need refinement if the order isn't guaranteed or if 'infer_and_crop' changes it.
                if 'lps' in data and data['lps']:
                    # This simple return will only work if we assume one plate per image in the pickle,
                    # or if the crop index matches the lp_id index.
                    # Let's return a list of all lp_ids in the image.
                    return [lp.get('lp_id', '') for lp in data['lps']]
            except Exception as e:
                print(f"Error reading/parsing Spanish raw JSON {json_path}: {e}")
        return [] # Return empty list if not found or error

    print("\nStarting ground truth crop generation...")

    for dataset_name, splits_data in all_processed_data.items(): # e.g. romanian, spanish
        print(f"\nProcessing dataset: {dataset_name}")

        for split_name, image_data_list_for_split in splits_data.items(): # e.g. train, valid, test
            if not image_data_list_for_split:
                print(f"No data for {dataset_name} - {split_name}, skipping.")
                continue

            output_crop_dir_for_split = GT_CROPS_OUTPUT_BASE / dataset_name / split_name
            output_crop_dir_for_split.mkdir(parents=True, exist_ok=True)
            
            # For Spanish GT texts, prepare a list to save to a file for this split
            spanish_gt_texts_for_this_split = []

            print(f"  Processing split: {split_name} -> saving crops to {output_crop_dir_for_split}")

            for image_info in tqdm(image_data_list_for_split, desc=f"  Cropping {dataset_name} {split_name}"):
                # 'img_path' in image_info is already relative to OUTPUT_IMAGE_BASE
                # e.g., 'romanian/train/some_image.jpg'
                full_processed_img_path = PROCESSED_IMAGE_BASE / image_info['img_path'] # <<< USE THE DEFINED BASE
                
                if not full_processed_img_path.exists():
                    print(f"Warning: Processed image not found: {full_processed_img_path}, skipping crops for this entry.")
                    continue

                image_cv = cv2.imread(str(full_processed_img_path))
                if image_cv is None:
                    print(f"Warning: Could not load processed image: {full_processed_img_path}, skipping.")
                    continue

                # The base_img_filename for the crop should come from the processed image path
                base_img_filename_for_crop = pathlib.Path(image_info['img_path']).stem

                # For Spanish dataset, try to get list of ground truth texts for this image
                spanish_lp_ids_for_this_image = []
                if dataset_name == "spanish":
                    # The image_info['img_path'] is relative, like 'spanish/train/00001.jpg'
                    # We need the base_id ('00001') from it
                    raw_base_id = pathlib.Path(image_info['img_path']).stem
                    spanish_lp_ids_for_this_image = get_spanish_gt_text_from_raw(raw_base_id, split_name)

                for i, annot in enumerate(image_info['annotations']):
                    xmin, ymin, xmax, ymax = map(int, annot['bbox']) # Bbox is relative to processed image

                    if xmin >= xmax or ymin >= ymax:
                        # print(f"Warning: Invalid bbox {annot['bbox']} for {base_img_filename_for_crop}, skipping crop.")
                        continue
                    
                    cropped_plate = image_cv[ymin:ymax, xmin:xmax]

                    if cropped_plate.size == 0:
                        # print(f"Warning: Cropped plate from {base_img_filename_for_crop} (box {i}) is empty, skipping.")
                        continue

                    crop_filename = f"{base_img_filename_for_crop}_plate_{i}.jpg"
                    crop_save_path = output_crop_dir_for_split / crop_filename
                    try:
                        cv2.imwrite(str(crop_save_path), cropped_plate)
                    except Exception as e_write:
                        print(f"Error writing crop {crop_save_path}: {e_write}")
                        continue
                    
                    # If it's Spanish and we have a corresponding lp_id
                    if dataset_name == "spanish" and i < len(spanish_lp_ids_for_this_image):
                        gt_text = spanish_lp_ids_for_this_image[i]
                        if gt_text: # Only add if not empty
                            spanish_gt_texts_for_this_split.append(f"{crop_filename}\t{gt_text}")
            
            # Save the collected Spanish GT texts for the current split
            if dataset_name == "spanish" and spanish_gt_texts_for_this_split:
                gt_text_file_path = output_crop_dir_for_split / "ground_truth_texts.tsv"
                try:
                    with open(gt_text_file_path, 'w', encoding='utf-8') as f_gt:
                        f_gt.write("\n".join(spanish_gt_texts_for_this_split))
                    print(f"  Saved Spanish ground truth texts for {split_name} to {gt_text_file_path}")
                except Exception as e_gt_write:
                    print(f"  Error writing Spanish GT text file for {split_name}: {e_gt_write}")

    print("\nGround truth crop generation finished.")
    print(f"Crops saved under: {GT_CROPS_OUTPUT_BASE.resolve()}")

if __name__ == "__main__":
    create_ground_truth_crops()