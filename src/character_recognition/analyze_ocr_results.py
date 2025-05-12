# src/character_recognition/analyze_ocr_results.py
import json
import pathlib
import argparse
from collections import Counter
import sys

def analyze_ocr_results(json_file_path_obj): # Takes pathlib.Path object
    """
    Processes an OCR results JSON file and prints a summary of conclusions.
    Also saves the summary to a text file.
    """
    if not json_file_path_obj.is_file():
        print(f"Error: Results file not found at {json_file_path_obj}")
        return None

    try:
        with open(json_file_path_obj, 'r') as f:
            results_data = json.load(f)
    except Exception as e:
        print(f"Error reading/decoding JSON from {json_file_path_obj}: {e}")
        return None

    if not isinstance(results_data, list):
        print("Error: JSON data is not in the expected list format.")
        return None

    summary_lines = []
    def add_to_summary(line):
        print(line)
        summary_lines.append(line)

    # --- Initialize accumulators (Similar to your original conclusions.py) ---
    total_processed = len(results_data)
    format_valid_count = 0 # Changed from is_valid_ro_format
    
    all_cleaned_texts = []
    chars_in_valid_plates = []
    lengths_of_valid_plates = []
    validation_schema_used = "N/A"

    for entry in results_data:
        if not isinstance(entry, dict):
            print(f"Warning: Skipping non-dictionary entry: {entry}")
            continue

        is_valid = entry.get("is_valid_format", False) # Use the new key
        cleaned_text = entry.get("recognized_text_cleaned", "")
        # raw_text = entry.get("recognized_text_raw", "")
        # image_file = entry.get("image_file", "N/A")
        if "validation_schema" in entry: # Get the schema used
             validation_schema_used = entry["validation_schema"]


        all_cleaned_texts.append(cleaned_text)
        if is_valid and cleaned_text: # Only count if text is present and format is valid
            format_valid_count += 1
            chars_in_valid_plates.extend(list(cleaned_text))
            lengths_of_valid_plates.append(len(cleaned_text))

    # --- Calculate Statistics (Similar to your original) ---
    format_invalid_count = total_processed - format_valid_count
    percent_valid = (format_valid_count / total_processed * 100) if total_processed > 0 else 0
    percent_invalid = (format_invalid_count / total_processed * 100) if total_processed > 0 else 0
    unique_cleaned_text_counts = Counter(all_cleaned_texts)
    char_frequency_valid = Counter(chars_in_valid_plates)
    length_distribution_valid = Counter(lengths_of_valid_plates)
    
    # --- Print Summary ---
    add_to_summary("\n--- OCR Results Analysis Summary ---")
    add_to_summary(f"Processed results from: {json_file_path_obj.name}")
    add_to_summary(f"Validation Schema Used: {validation_schema_used}")
    add_to_summary("--------------------------------------------------")
    add_to_summary(f"Total Plates Processed: {total_processed}")
    add_to_summary("--------------------------------------------------")
    add_to_summary(f"Plate Validation (Format: {validation_schema_used}):")
    add_to_summary(f"  Valid Format: {format_valid_count} ({percent_valid:.2f}%)")
    add_to_summary(f"  Invalid Format: {format_invalid_count} ({percent_invalid:.2f}%)")
    # ... (add more detailed printouts for frequencies, lengths etc. as in your original conclusions.py)

    # --- Save summary to file ---
    summary_file_path = json_file_path_obj.parent / f"{json_file_path_obj.stem}_analysis.txt"
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as f_sum:
            f_sum.write("\n".join(summary_lines))
        print(f"\nAnalysis summary saved to: {summary_file_path}")
    except Exception as e_write_sum:
        print(f"Error saving analysis summary: {e_write_sum}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze OCR results JSON file.")
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the ocr_results.json file to analyze."
    )
    args = parser.parse_args()

    results_path = pathlib.Path(args.results_file)
    analyze_ocr_results(results_path)