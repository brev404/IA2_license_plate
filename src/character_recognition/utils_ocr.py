# src/character_recognition/utils_ocr.py
import cv2
import re # Ensure re is imported
import pytesseract
import pathlib

# --- Preprocessing Function ---
def preprocess_plate_for_ocr(image):
    """Converts image to grayscale, enhances contrast, and applies thresholding."""
    if image is None:
        return None
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        gray = image
    else:
        return None

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    binary_image = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    return binary_image

# --- Detailed Spanish Validation Function (Internal Helper) ---
def _validate_spanish_license_plate_detailed(plate_string_cleaned_upper):
    """
    Validates a pre-cleaned, uppercase string against Spanish license plate formats.
    Args:
        plate_string_cleaned_upper (str): The license plate string (uppercase, no separators).
    Returns:
        tuple: (bool: True if valid, str: validation message)
    """
    # 1. Current National Format (since September 2000): NNNN LLL
    current_format_strict_regex = r"^\d{4}[BCDFGHJKLMNPRSTVWXYZ]{3}$"
    if re.fullmatch(current_format_strict_regex, plate_string_cleaned_upper):
        return True, "Valid (Current National Format)"

    # 2. Provincial System (October 1971 - September 2000): P NNNN LL / P NNNN L / PP NNNN L / PP NNNN LL
    # Already cleaned of separators, so we match the content directly.
    provincial_no_sep_regex = r"^[A-Z]{1,2}\d{4}[A-Z]{1,2}$"
    if re.fullmatch(provincial_no_sep_regex, plate_string_cleaned_upper):
        # Further check: ensure the numeric part is exactly 4 digits
        # This is implicitly handled by the regex \d{4}
        return True, "Valid (Provincial Format 1971-2000)"

    # 3. Old Provincial System (1900 - October 1971): P NNNNNN
    old_provincial_no_sep_regex = r"^[A-Z]{1,3}\d{1,6}$"
    if re.fullmatch(old_provincial_no_sep_regex, plate_string_cleaned_upper):
        # This format can be short (e.g., B1).
        # Avoid conflict with current format if it's all digits after letters.
        if not (len(plate_string_cleaned_upper) == 7 and plate_string_cleaned_upper[:4].isdigit()): # Avoid NNNNLLL
             return True, "Valid (Old Provincial Format 1900-1971)"

    return False, "Invalid or unrecognized Spanish format"

# --- Main Validation Function ---
VALID_ROMANIAN_COUNTY_CODES = {
    "AB", "AR", "AG", "BC", "BH", "BN", "BT", "BV", "BR", "B", "BZ", "CS", "CL", "CJ", "CT", "CV",
    "DB", "DJ", "GL", "GR", "GJ", "HR", "HD", "IL", "IS", "IF", "MM", "MH", "MS", "NT", "OT", "PH",
    "SM", "SJ", "SB", "SV", "TR", "TM", "TL", "VS", "VL", "VN"
}

def validate_plate_text(text, plate_format="RO"):
    """
    Validates the recognized text against specified license plate formats.
    Cleans text (removes common separators, ensures uppercase) before validation.

    Args:
        text (str): The OCR'd text from the license plate.
        plate_format (str): The country/region format to validate against.
                              Supports "RO", "ES", "GENERIC_ALPHANUM", "NONE".
    Returns:
        tuple: (bool, str)
               bool: True if the text is considered a valid plate number for the format.
               str: Message describing validation result.
    """
    if not text or not isinstance(text, str):
        return False, f"Invalid (Input not a string or empty): '{text}'"

    # Standard cleaning: remove common separators (space, hyphen, period) and uppercase
    cleaned_text_for_regex = re.sub(r"[\s\-\.]", "", text).upper()

    if not cleaned_text_for_regex:
        return False, f"Invalid (Empty after cleaning): Original '{text}'"

    original_text_for_msg = f"Original '{text}', Cleaned '{cleaned_text_for_regex}'"

    if plate_format == "RO":
        ro_strict_pattern = r'^([A-Z]{1,2})(\d{2,3})([A-Z]{3})$'
        match = re.fullmatch(ro_strict_pattern, cleaned_text_for_regex)
        if match:
            county_code = match.group(1)
            if county_code in VALID_ROMANIAN_COUNTY_CODES:
                return True, f"Valid (Romanian Format) - {original_text_for_msg}"
            else:
                return False, f"Invalid (Romanian - Bad County '{county_code}') - {original_text_for_msg}"
        return False, f"Invalid (Romanian - Pattern Mismatch) - {original_text_for_msg}"

    elif plate_format == "ES":
        is_valid_es, es_detail_message = _validate_spanish_license_plate_detailed(cleaned_text_for_regex)
        return is_valid_es, f"{es_detail_message} - {original_text_for_msg}"

    elif plate_format == "GENERIC_ALPHANUM":
        if re.fullmatch(r'^[A-Z0-9]{5,9}$', cleaned_text_for_regex): # Example: 5-9 alphanumeric
            return True, f"Valid (Generic Alphanum) - {original_text_for_msg}"
        return False, f"Invalid (Generic Alphanum) - {original_text_for_msg}"

    elif plate_format == "NONE":
        return True, f"Validation Skipped - {original_text_for_msg}"

    else:
        # print(f"Warning: Unknown plate_format '{plate_format}' for validation.") # Console print if needed
        return False, f"Unknown Format '{plate_format}' - {original_text_for_msg}"


# --- OCR Function ---
def recognize_text_tesseract(image_cv2, tesseract_config_str="--oem 3 --psm 7", whitelist=None):
    """Performs OCR using Tesseract on a preprocessed OpenCV image."""
    if image_cv2 is None: return ""

    custom_config = tesseract_config_str
    if whitelist:
        custom_config += f" -c tessedit_char_whitelist={whitelist}"
    
    try:
        text = pytesseract.image_to_string(image_cv2, config=custom_config)
    except pytesseract.TesseractError as e:
        print(f"Tesseract OCR error: {e}") # Keep this for important errors
        return ""
    return text.strip() # Return raw strip, uppercasing is handled in validate or main script