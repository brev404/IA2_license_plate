from ultralytics import YOLO
import cv2
import pathlib
import os
import numpy as np # Good practice to import numpy

# --- Configuration ---
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()  # Adjust if directory changes
MODEL_PATH = PROJECT_ROOT / 'results/LicensePlateDet/yolov8s_yolo_spanish_lp_e50/weights/best.pt'  # Path to trained YOLO model
OUTPUT_DIR = PROJECT_ROOT / 'data/cropped_plates_sph'  # Cropped license plates will be saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Function to Perform Detection and Crop License Plates ---
def detect_and_crop(source_path_str):
    """
    Performs detection on images from a given source (file or directory)
    and crops the detected license plates.

    Args:
        source_path_str (str): Path to an image file or a directory of images.
    """
    # Load the trained YOLO model
    model = YOLO(str(MODEL_PATH))
    
    # Perform inference. 
    # The model call handles both single image files and directories.
    # It returns a list of Results objects (one for each image).
    all_results = model(source_path_str)
    
    # Process results for each image
    for single_image_results in all_results:
        # Get the original image from the Results object (YOLO already loaded it)
        original_image = single_image_results.orig_img 
        
        # Get the path of the current image being processed
        current_image_path_obj = pathlib.Path(single_image_results.path)
        
        # Get bounding boxes [xmin, ymin, xmax, ymax]
        boxes = single_image_results.boxes.xyxy 
        
        if boxes.nelement() == 0: # Check if no bounding boxes were detected
            print(f"No license plates detected in {current_image_path_obj.name}")
            continue
            
        # Process each detected box for the current image
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Basic validation for box coordinates
            if x1 >= x2 or y1 >= y2:
                print(f"Warning: Invalid box coordinates {box} for {current_image_path_obj.name}. Skipping crop.")
                continue

            # Crop the license plate region
            # Ensure coordinates are within image bounds before cropping
            img_h, img_w = original_image.shape[:2]
            x1_c = max(0, x1)
            y1_c = max(0, y1)
            x2_c = min(img_w, x2)
            y2_c = min(img_h, y2)

            if x1_c >= x2_c or y1_c >= y2_c: # Re-check after clamping
                print(f"Warning: Box coordinates [{x1_c},{y1_c},{x2_c},{y2_c}] became invalid after clamping for {current_image_path_obj.name}. Original box: {box}. Skipping crop.")
                continue
                
            cropped_plate = original_image[y1_c:y2_c, x1_c:x2_c]
            
            if cropped_plate.size == 0:
                print(f"Warning: Cropped plate from {current_image_path_obj.name} with box {box} is empty. Skipping save.")
                continue
            
            # Save the cropped plate
            output_filename = f"{current_image_path_obj.stem}_plate_{i}.jpg"
            output_path = OUTPUT_DIR / output_filename
            
            try:
                cv2.imwrite(str(output_path), cropped_plate)
                print(f"Cropped plate saved to: {output_path}")
            except Exception as e:
                print(f"Error saving cropped plate {output_path}: {e}. Original box: {box}, Cropped shape: {cropped_plate.shape}")

# --- Main Function ---
if __name__ == "__main__":
    # Path to an input image or folder of images
    # This should be the path to your directory of images
    IMAGE_FOLDER_PATH = PROJECT_ROOT / 'data/processed/images/spanish/train' 

    # Check if the image folder path exists
    if not IMAGE_FOLDER_PATH.exists():
        print(f"Error: Image folder not found at {IMAGE_FOLDER_PATH}")
    elif not IMAGE_FOLDER_PATH.is_dir():
        print(f"Error: The path {IMAGE_FOLDER_PATH} is not a directory.")
    else:
        # Run the detection and cropping
        detect_and_crop(str(IMAGE_FOLDER_PATH))