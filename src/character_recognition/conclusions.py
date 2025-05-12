import json
import pathlib
from collections import Counter

# --- Configuration (if running this function standalone or in a new script) ---
# This assumes the function might be in a script within a project structure.
# Adjust if necessary, or ensure RESULTS_FILE_PATH is passed correctly.
try:
    SCRIPT_DIR_ANALYZER = pathlib.Path(__file__).parent.resolve()
    PROJECT_ROOT_ANALYZER = SCRIPT_DIR_ANALYZER.parent.resolve()
except NameError:
    PROJECT_ROOT_ANALYZER = pathlib.Path('.').resolve() # Assumes script is run from project root

# Default path for the OCR results JSON file
DEFAULT_OCR_RESULTS_PATH = PROJECT_ROOT_ANALYZER / 'data' / 'ocr_results.json'

def analyze_ocr_results(json_file_path_str=str(DEFAULT_OCR_RESULTS_PATH)):
    """
    Processes an OCR results JSON file and prints a summary of conclusions.

    Args:
        json_file_path_str (str): The path to the ocr_results.json file.

    Returns:
        dict: A dictionary containing the summarized statistics, or None if an error occurs.
    """
    json_file_path = pathlib.Path(json_file_path_str)

    if not json_file_path.is_file():
        print(f"Error: Results file not found at {json_file_path}")
        return None

    try:
        with open(json_file_path, 'r') as f:
            results_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {json_file_path}: {e}")
        return None

    if not isinstance(results_data, list):
        print("Error: JSON data is not in the expected list format.")
        return None

    # Initialize accumulators
    total_processed = len(results_data)
    valid_plate_count = 0
    unvalidated_entries = []
    
    all_cleaned_texts = []
    chars_in_valid_plates = []
    lengths_of_valid_plates = []

    for entry in results_data:
        if not isinstance(entry, dict):
            print(f"Warning: Skipping non-dictionary entry: {entry}")
            continue

        is_valid = entry.get("is_valid_ro_format", False) # Default to False if key missing
        cleaned_text = entry.get("recognized_text_cleaned", "")
        raw_text = entry.get("recognized_text_raw", "")
        image_file = entry.get("image_file", "N/A")

        all_cleaned_texts.append(cleaned_text)

        if is_valid:
            valid_plate_count += 1
            if cleaned_text: # Only process if there's actual text
                chars_in_valid_plates.extend(list(cleaned_text))
                lengths_of_valid_plates.append(len(cleaned_text))
        else:
            unvalidated_entries.append({
                "image_file": image_file,
                "raw_text": raw_text,
                "cleaned_text": cleaned_text
            })

    # --- Calculate Statistics ---
    invalid_plate_count = total_processed - valid_plate_count
    percent_valid = (valid_plate_count / total_processed * 100) if total_processed > 0 else 0
    percent_invalid = (invalid_plate_count / total_processed * 100) if total_processed > 0 else 0

    unique_cleaned_text_counts = Counter(all_cleaned_texts)
    char_frequency_valid = Counter(chars_in_valid_plates)
    length_distribution_valid = Counter(lengths_of_valid_plates)

    # --- Prepare Summary Output ---
    summary = {
        "total_plates_processed": total_processed,
        "valid_plates": {
            "count": valid_plate_count,
            "percentage": round(percent_valid, 2)
        },
        "invalid_plates": {
            "count": invalid_plate_count,
            "percentage": round(percent_invalid, 2)
        },
        "unique_cleaned_plate_texts_count": len(unique_cleaned_text_counts),
        "most_frequent_cleaned_texts": unique_cleaned_text_counts.most_common(10), # Top 10
        "character_frequency_in_valid_plates": dict(char_frequency_valid.most_common()),
        "length_distribution_of_valid_plates": dict(length_distribution_valid.most_common()),
        "unvalidated_entries_details": unvalidated_entries[:20] # Show details for up to 20
    }

    # --- Print Summary ---
    print("\n--- OCR Results Analysis Summary ---")
    print(f"Processed results from: {json_file_path.name}")
    print("--------------------------------------------------")
    print(f"Total Plates Processed: {summary['total_plates_processed']}")
    print("--------------------------------------------------")
    print("Plate Validation (RO Format):")
    print(f"  Valid Plates: {summary['valid_plates']['count']} ({summary['valid_plates']['percentage']}%)")
    print(f"  Invalid Plates: {summary['invalid_plates']['count']} ({summary['invalid_plates']['percentage']}%)")
    print("--------------------------------------------------")
    print(f"Unique Cleaned Plate Texts Recognized: {summary['unique_cleaned_plate_texts_count']}")
    print("Most Frequent Cleaned Texts (Top 10):")
    if summary['most_frequent_cleaned_texts']:
        for text, count in summary['most_frequent_cleaned_texts']:
            print(f"  - '{text}': {count} time(s)")
    else:
        print("  No cleaned texts found.")
    print("--------------------------------------------------")
    print("Character Frequency in Valid & Cleaned Plates:")
    if summary['character_frequency_in_valid_plates']:
        for char, count in summary['character_frequency_in_valid_plates'].items():
            print(f"  - '{char}': {count}")
    else:
        print("  No characters found in valid plates (or no valid plates).")
    print("--------------------------------------------------")
    print("Length Distribution of Valid & Cleaned Plates:")
    if summary['length_distribution_of_valid_plates']:
        for length, count in sorted(summary['length_distribution_of_valid_plates'].items()):
            print(f"  - Length {length}: {count} plate(s)")
    else:
        print("  No length data for valid plates (or no valid plates).")
    print("--------------------------------------------------")
    print(f"Details for Unvalidated Plates (showing up to {len(summary['unvalidated_entries_details'])}):")
    if summary['unvalidated_entries_details']:
        for entry in summary['unvalidated_entries_details']:
            print(f"  - Image: {entry['image_file']}, Raw OCR: '{entry['raw_text']}', Cleaned: '{entry['cleaned_text']}'")
    else:
        print("  No unvalidated plates found, or all plates were valid.")
    print("--- End of Summary ---")

    return summary

# --- Example of how to use the function ---
if __name__ == "__main__":
    print(f"Attempting to analyze OCR results. Default file path: {DEFAULT_OCR_RESULTS_PATH}")
    
    # Create a dummy ocr_results.json for testing if it doesn't exist
    # In a real scenario, this file would be generated by your OCR processing script
    dummy_data_for_testing = False # Set to True to generate a dummy file for a quick test run
    if not DEFAULT_OCR_RESULTS_PATH.exists() and dummy_data_for_testing:
        print(f"Creating a dummy '{DEFAULT_OCR_RESULTS_PATH.name}' for testing purposes.")
        DEFAULT_OCR_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        sample_results = [
            {"image_file": "plate1_crop0.jpg", "recognized_text_raw": "B 123 ABC", "recognized_text_cleaned": "B123ABC", "is_valid_ro_format": True},
            {"image_file": "plate2_crop0.jpg", "recognized_text_raw": "CJ 01 XYZ", "recognized_text_cleaned": "CJ01XYZ", "is_valid_ro_format": True},
            {"image_file": "plate3_crop0.jpg", "recognized_text_raw": "B 22 DEF ", "recognized_text_cleaned": "B22DEF", "is_valid_ro_format": True},
            {"image_file": "plate4_crop0.jpg", "recognized_text_raw": "IS 345 KLM", "recognized_text_cleaned": "IS345KLM", "is_valid_ro_format": True}, # Example of 3 numbers for non-B
            {"image_file": "plate5_crop0.jpg", "recognized_text_raw": "X Y Z 123", "recognized_text_cleaned": "XYZ123", "is_valid_ro_format": False},
            {"image_file": "plate6_crop0.jpg", "recognized_text_raw": "AB01CDE", "recognized_text_cleaned": "AB01CDE", "is_valid_ro_format": True}, # Correct format AB01CDE
            {"image_file": "plate7_crop0.jpg", "recognized_text_raw": "B123ABC", "recognized_text_cleaned": "B123ABC", "is_valid_ro_format": True}, # Duplicate to test frequency
            {"image_file": "plate8_crop0.jpg", "recognized_text_raw": "", "recognized_text_cleaned": "", "is_valid_ro_format": False}, # Empty OCR
            {"image_file": "plate9_crop0.jpg", "recognized_text_raw": "B001TOOLONG", "recognized_text_cleaned": "B001TOOLONG", "is_valid_ro_format": False},
        ]
        try:
            with open(DEFAULT_OCR_RESULTS_PATH, 'w') as f:
                json.dump(sample_results, f, indent=4)
            print(f"Dummy file '{DEFAULT_OCR_RESULTS_PATH.name}' created successfully.")
        except Exception as e:
            print(f"Could not create dummy file: {e}")
            
    analysis_summary = analyze_ocr_results() # Uses the default path

    # If you want to use the returned dictionary for something else:
    # if analysis_summary:
    #     print("\n(Programmatic access to summary data example):")
    #     print(f"Total valid plates from summary dict: {analysis_summary['valid_plates']['count']}")