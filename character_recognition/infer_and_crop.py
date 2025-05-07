from ultralytics import YOLO
import cv2
import pathlib
import os

# --- Configuration ---
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()  # Adjust if directory changes
MODEL_PATH = PROJECT_ROOT / 'runs/detect/LicensePlateDet/yolov8s_romanian_lp_e50/best.pt'  # Path to trained YOLO model
OUTPUT_DIR = PROJECT_ROOT / 'data/cropped_plates'  # Cropped license plates will be saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Function to Perform Detection and Crop License Plates ---
def detect_and_crop(image_path):
    # Load the trained YOLO model
    model = YOLO(str(MODEL_PATH))
    
    # Perform inference
    results = model(image_path)
    
    # Read the original image
    original_image = cv2.imread(image_path)
    
    # Process each detection
    for result in results:
        boxes = result.boxes.xyxy  # Bounding boxes [xmin, ymin, xmax, ymax]
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Crop the license plate region
            cropped_plate = original_image[y1:y2, x1:x2]
            
            # Save the cropped plate
            output_path = OUTPUT_DIR / f"{pathlib.Path(image_path).stem}_plate_{i}.jpg"
            cv2.imwrite(str(output_path), cropped_plate)
            print(f"Cropped plate saved to: {output_path}")

# --- Main Function ---
if __name__ == "__main__":
    # Path to an input image or folder of images
    IMAGE_PATH = PROJECT_ROOT / 'data/raw/test_image.jpg'  # Replace with your test image path

    # Run the detection and cropping
    detect_and_crop(str(IMAGE_PATH))
