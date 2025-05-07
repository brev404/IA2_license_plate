import pytesseract

# Optional: Set Tesseract path (if not in system PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def recognize_text(cropped_plate_path):
    # Read the cropped plate image
    image = cv2.imread(cropped_plate_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to improve OCR accuracy
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Perform OCR using Tesseract
    text = pytesseract.image_to_string(thresh, config='--psm 8')  # PSM 8: Treat image as a single word
    return text.strip()

# Integrate OCR into the pipeline
def detect_and_recognize(image_path):
    detect_and_crop(image_path)  # Run detection and crop plates
    
    # Process each cropped license plate
    for cropped_path in OUTPUT_DIR.glob("*.jpg"):
        text = recognize_text(str(cropped_path))
        print(f"Recognized Text for {cropped_path.name}: {text}")
