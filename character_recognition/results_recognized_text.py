import json

RESULTS_PATH = PROJECT_ROOT / 'results/detections.json'  # Save results in a JSON file

def save_results(results):
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {RESULTS_PATH}")

def detect_and_recognize_with_logging(image_path):
    detect_and_crop(image_path)  # Run detection and crop plates
    
    results = []
    for cropped_path in OUTPUT_DIR.glob("*.jpg"):
        text = recognize_text(str(cropped_path))
        results.append({
            "image": str(cropped_path.name),
            "text": text
        })
        print(f"Recognized Text: {text}")
    
    save_results(results)
