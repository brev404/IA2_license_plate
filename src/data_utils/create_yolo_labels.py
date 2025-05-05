# src/data_utils/create_yolo_labels.py
import os
import pathlib
import pickle
from tqdm import tqdm
import sys

# Import only the necessary utility function
try:
    from utils import convert_to_yolo_format
except ImportError:
    print("ERROR: Could not import from utils.py. Make sure it's in the same directory (src/data_utils/).")
    sys.exit(1)


# --- Configuration ---
# Define paths relative to this script file using pathlib
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve() # Assumes script is in ProjectRoot/src/data_utils/
# Path to the intermediate data saved by preprocess.py
INTERMEDIATE_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'intermediate' / 'processed_annotations.pkl'
# Base path where processed images were saved (needed to reconstruct relative paths)
OUTPUT_IMAGE_BASE = PROJECT_ROOT / 'data' / 'processed' / 'images'
# Base path where YOLO label files will be saved
OUTPUT_YOLO_LABEL_DIR = PROJECT_ROOT / 'data' / 'processed' / 'labels' / 'yolo'

# --- Formatting Function (Corrected Version) ---
def save_yolo_labels_from_intermediate(processed_data, output_label_dir_base):
    """
    Generates YOLO label files from the processed data structure loaded from intermediate file.
    """
    print(f"\nGenerating YOLO labels...")
    count = 0
    if not processed_data:
        print("Warning: No processed data found to generate labels.")
        return

    # Flatten the data if it's in the nested dict structure {dataset: {split: [data]}}
    all_data_flat = []
    if isinstance(processed_data, dict):
        for ds_key in processed_data:
             # Check if the key exists before accessing
             if ds_key in processed_data:
                 for split_key in processed_data[ds_key]:
                     # Check if split data exists and is a list
                     if split_key in processed_data[ds_key] and isinstance(processed_data[ds_key][split_key], list):
                          all_data_flat.extend(processed_data[ds_key][split_key])
    elif isinstance(processed_data, list): # If already flat (less likely with current preprocess.py)
         all_data_flat = processed_data
    else:
         print("Error: Unexpected format for processed_data. Expected dictionary.")
         return

    if not all_data_flat:
         print("Warning: No data items found after flattening.")
         return

    print(f"Processing {len(all_data_flat)} total image annotations...")

    for image_info in tqdm(all_data_flat, desc="Saving YOLO Labels"):
        # 'img_path' from the pickle file should be the relative path string
        # e.g., 'romanian/train/img001.jpg' or 'spanish/test/01234.jpg'
        relative_img_path_str = image_info['img_path']
        img_shape = image_info['img_shape'] # (width, height)
        annotations = image_info['annotations']

        # Double check if annotations list is empty
        if not annotations:
            # print(f"Debug: No annotations found for image {relative_img_path_str}, skipping label file creation.")
            continue

        try:
            # Convert the relative string path to a Path object
            relative_img_path_obj = pathlib.Path(relative_img_path_str)

            # Change the suffix from image extension (.jpg/.png) to .txt
            label_relative_path = relative_img_path_obj.with_suffix('.txt')

            # Join the base output directory for labels with this relative label path
            # e.g., D:/.../IA2/data/processed/labels/yolo / romanian/train/img001.txt
            label_path = output_label_dir_base / label_relative_path

        except Exception as e: # Catch any path construction error
             print(f"\nError constructing label path for relative image path '{relative_img_path_str}'. Error: {e}")
             continue # Skip this file

        # Ensure the parent directory for the label file exists
        label_path.parent.mkdir(parents=True, exist_ok=True)

        yolo_lines = []
        for annot in annotations:
            # Convert the absolute bbox from image_info['annotations'] to YOLO format
            # Assumes 'bbox' is [xmin, ymin, xmax, ymax] and 'class_id' exists
            yolo_str = convert_to_yolo_format(annot['bbox'], img_shape, annot['class_id'])
            if yolo_str:
                yolo_lines.append(yolo_str)

        # Only write file if there are valid annotations for this image
        if yolo_lines:
            try:
                with open(label_path, 'w') as f:
                    f.write("\n".join(yolo_lines))
                count += 1
            except Exception as e_write:
                 print(f"\nError writing label file {label_path}. Error: {e_write}")
        # Optional: Log if an image had annotations but no valid YOLO lines generated
        # elif annotations:
        #     print(f"Warning: No valid YOLO lines generated for {relative_img_path_str}, although annotations existed.")

    print(f"\nSaved {count} YOLO label files in base directory: {output_label_dir_base.resolve()}")

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Attempting to load intermediate data from: {INTERMEDIATE_DATA_PATH}")
    if not INTERMEDIATE_DATA_PATH.exists():
        print(f"ERROR: Intermediate data file not found at {INTERMEDIATE_DATA_PATH}.")
        print(f"Please run preprocess.py first to generate it.")
    else:
        try:
            with open(INTERMEDIATE_DATA_PATH, 'rb') as f:
                # Load the nested dictionary structure {dataset: {split: [data]}}
                all_processed_data = pickle.load(f)
            print("Intermediate data loaded successfully.")

            # Ensure the base YOLO label directory exists
            OUTPUT_YOLO_LABEL_DIR.mkdir(parents=True, exist_ok=True)

            # Call the save function, passing the loaded data structure
            save_yolo_labels_from_intermediate(all_processed_data, OUTPUT_YOLO_LABEL_DIR)

        except FileNotFoundError:
             print(f"ERROR: Could not find intermediate data file during load: {INTERMEDIATE_DATA_PATH}")
        except Exception as e:
            print(f"ERROR loading or processing intermediate data: {e}")

    print("\nYOLO label generation finished.")