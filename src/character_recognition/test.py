import cv2
import pytesseract # For OCR
import pathlib
import os
import json
import re

# --- Configuration ---
# Assuming this script is in a subdirectory of your project root (e.g., project_root/scripts/)
# Adjust these paths if your script is located elsewhere or your directory structure differs.
try:
    SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent.resolve()
except NameError:
    # Fallback if __file__ is not defined (e.g., in an interactive environment)
    # You might need to set PROJECT_ROOT manually in such cases.
    PROJECT_ROOT = pathlib.Path('.').resolve() # Assumes script is run from project root
    print(f"Warning: __file__ not defined. Assuming PROJECT_ROOT is current directory: {PROJECT_ROOT}")


# Directory containing the already cropped license plates
# This should be the 'OUTPUT_DIR' from your previous script.
CROPPED_PLATES_DIR = PROJECT_ROOT / 'data' / 'cropped_plates'

# Path to save the OCR results
RESULTS_FILE_PATH = PROJECT_ROOT / 'data' / 'ocr_results.json'

# --- Preprocessing Function ---
def preprocess_plate(image):
    """Converts image to grayscale, enhances contrast using CLAHE, and applies binary thresholding."""
    if image is None:
        print("Error: Input image to preprocess_plate is None.")
        return None
    
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2: # Already grayscale
        gray = image
    else:
        print(f"Error: Unsupported image format with shape {image.shape} for preprocessing.")
        return None

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary_image

# --- OCR Function ---
def recognize_text_from_plate_image(image_path_str):
    """Reads a cropped plate image, preprocesses it, and performs OCR."""
    image = cv2.imread(image_path_str)
    if image is None:
        print(f"Error: Could not read image at {image_path_str}")
        return "" # Return empty string on error

    preprocessed_image = preprocess_plate(image)
    if preprocessed_image is None:
        print(f"Error: Preprocessing failed for image {image_path_str}")
        return ""

    # Configure Tesseract OCR
    # --psm 8: Treat the image as a single word.
    # --psm 7: Treat the image as a single text line. (Often good for license plates)
    # --oem 3: Default OCR Engine Mode (uses LSTM if available)
    # -c tessedit_char_whitelist: Restricts OCR to a specific set of characters.
    # Adjust whitelist based on expected characters in your license plates.
    char_whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    custom_config = f'--oem 3 --psm 7 -c tessedit_char_whitelist={char_whitelist}'
    
    try:
        text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
    except pytesseract.TesseractError as e:
        print(f"Tesseract OCR error for {image_path_str}: {e}")
        return ""
    return text.strip().upper() # Strip whitespace and convert to uppercase

import re

# List of official Romanian county codes (județe) including Bucharest ('B')
VALID_ROMANIAN_COUNTY_CODES = {
    "AB", "AR", "AG", "BC", "BH", "BN", "BT", "BV", "BR", "B", "BZ",
    "CS", "CL", "CJ", "CT", "CV", "DB", "DJ", "GL", "GR", "GJ", "HR",
    "HD", "IL", "IS", "IF", "MM", "MH", "MS", "NT", "OT", "PH", "SM",
    "SJ", "SB", "SV", "TR", "TM", "TL", "VS", "VL", "VN"
}

def validate_plate_text(text, country_format="RO"):
    """
    Validates the recognized text against a regex pattern for license plates.
    For Romanian plates, it also checks if the county code is valid.
    Cleans text (removes spaces, ensures uppercase) before validation.

    Args:
        text (str): The OCR'd text from the license plate.
        country_format (str): The country format to validate against.
                              Currently supports "RO" and "USER_PROVIDED".

    Returns:
        bool: True if the text is considered a valid plate number for the format, False otherwise.
    """
    if not text:  # Handle empty text early
        return False
        
    # Remove all internal spaces and ensure uppercase for consistent processing
    cleaned_text = re.sub(r'\s+', '', text).upper()

    if country_format == "RO":
        # Romanian standard plate pattern:
        # 1 or 2 letters (County Code)
        # 2 or 3 numbers
        # 3 letters (Suffix)
        # Example: B12ABC, B123ABC, CJ01XYZ, IF999WWW
        ro_strict_pattern = r'^([A-Z]{1,2})([0-9]{2,3})([A-Z]{3})$'
        
        match = re.fullmatch(ro_strict_pattern, cleaned_text)
        
        if match:
            county_code = match.group(1)
            numbers_part = match.group(2)
            # letters_part = match.group(3) # Available if needed for further rules

            # 1. Check if the extracted county code is a valid Romanian county code
            if county_code not in VALID_ROMANIAN_COUNTY_CODES:
                # print(f"Debug: Invalid county code '{county_code}' in '{cleaned_text}'") # Optional debug
                return False  # County code is not in the official list

            # 2. Apply specific rules for number of digits (optional refinement, regex already handles 2 or 3)
            #    - Bucharest ('B') should have 2 or 3 digits.
            #    - Other 2-letter county codes typically have 2 digits for standard issues,
            #      though 3 digits can appear in some cases (e.g. temporary plates, older re-registrations).
            #      The regex [0-9]{2,3} already allows this flexibility.
            
            # Example of stricter digit rule (currently, the regex [0-9]{2,3} is more general):
            # if county_code == 'B':
            #     if not (2 <= len(numbers_part) <= 3): return False # Should be covered by regex
            # elif len(county_code) == 2: # For other counties
            #     if len(numbers_part) != 2:
            #          # This would make it stricter, current regex allows 2 or 3.
            #          # Consider if you want to flag 3-digit numbers for 2-letter counties as invalid
            #          # or as a different category. For now, we accept what the regex [0-9]{2,3} allows.
            #          pass

            return True  # Format is correct, and county code is valid
        
        # print(f"Debug: Text '{cleaned_text}' did not match RO pattern.") # Optional debug
        return False  # Regex pattern did not match

    elif country_format == "USER_PROVIDED":
        # User's originally provided pattern from a previous request
        user_provided_pattern = r'^[A-Z]{1,2}[0-9]{2,3}[A-Z]{1,2}$'
        return re.fullmatch(user_provided_pattern, cleaned_text) is not None
        
    else:  # Generic fallback (any alphanumeric string of reasonable length)
        generic_pattern = r'^[A-Z0-9]{3,10}$' # Example: 3 to 10 alphanumeric chars
        return re.fullmatch(generic_pattern, cleaned_text) is not None

# --- Function to Save Results ---
def save_ocr_results(results_data, file_path_obj):
    """Saves the OCR results to a JSON file."""
    try:
        file_path_obj.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with open(file_path_obj, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"OCR results saved to: {file_path_obj}")
    except IOError as e:
        print(f"Error saving results to {file_path_obj}: {e}")

# --- Main Processing Function ---
def process_cropped_plates_in_directory(directory_path_obj):
    """
    Processes all cropped license plate images in the specified directory,
    performs OCR, validates the text, and collects the results.
    """
    if not directory_path_obj.exists() or not directory_path_obj.is_dir():
        print(f"Error: Directory not found or is not a directory: {directory_path_obj}")
        return []

    ocr_results = []
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(directory_path_obj.glob(ext)))

    if not image_files:
        print(f"No image files found with extensions {image_extensions} in {directory_path_obj}")
        return []

    print(f"Found {len(image_files)} images in {directory_path_obj}. Processing...")

    for image_file_path in image_files:
        print(f"\nProcessing image: {image_file_path.name}")
        
        recognized_text = recognize_text_from_plate_image(str(image_file_path))
        print(f"  Raw OCR Text: '{recognized_text}'")
        
        # Validate using the Romanian pattern by default.
        # To use your original pattern, change to: country_format="USER_PROVIDED"
        is_valid = validate_plate_text(recognized_text, country_format="RO") 
        
        print(f"  Cleaned for Validation: '{re.sub(r'[^\w]', '', recognized_text).upper()}' -> Valid (RO Format): {is_valid}")
        
        ocr_results.append({
            "image_file": image_file_path.name,
            "recognized_text_raw": recognized_text, # Store raw OCR text before final cleaning for validation
            "recognized_text_cleaned": re.sub(r'\s+', '', recognized_text).upper(),
            "is_valid_ro_format": is_valid
        })
        
    return ocr_results


# --- Main Execution Block ---
if __name__ == "__main__":
    
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # IMPORTANT: Set the Tesseract CMD path if Tesseract is not in your system's PATH.
    # Example for Windows (uncomment and adjust the path if needed):
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Example for Linux (if installed in a non-standard location):
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

    # Check if Tesseract is accessible
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version {version} found.")
    except pytesseract.TesseractNotFoundError:
        print("Tesseract is not installed or not found in your PATH.")
        print("Please install Tesseract OCR and make sure it's added to your system's PATH,")
        print("or set the tesseract_cmd path explicitly at the beginning of the "
              "'if __name__ == \"__main__\":' block.")
        exit() # Exit if Tesseract is not found
    except Exception as e:
        print(f"An error occurred while checking Tesseract version: {e}")
        print("Please ensure Tesseract OCR is correctly installed and configured.")
        exit()

    print(f"Processing images from: {CROPPED_PLATES_DIR}")
    print(f"Saving results to: {RESULTS_FILE_PATH}")

    all_ocr_data = process_cropped_plates_in_directory(CROPPED_PLATES_DIR)
    
    if all_ocr_data:
        save_ocr_results(all_ocr_data, RESULTS_FILE_PATH)
    else:
        print("No OCR results were generated to save.")

    print("\nScript finished.")