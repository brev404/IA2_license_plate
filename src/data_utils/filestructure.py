# src/test_structure.py (or wherever you saved it)
import os
import pathlib

def generate_project_structure(root_dir_path, indent_str='|   ', ignore_dirs=None, max_depth=None):
    """
    Generates a list of strings representing the directory structure.

    Args:
        root_dir_path (str or pathlib.Path): The root directory to start from.
        indent_str (str): String to use for indentation.
        ignore_dirs (list, optional): A list of directory names to ignore.
                                      Defaults to common temporary/large dirs.
        max_depth (int, optional): Maximum depth to traverse. None means no limit.

    Returns:
        list: A list of strings, each string is a line in the structure tree.
    """
    root_dir = pathlib.Path(root_dir_path).resolve()
    output_lines = []

    if ignore_dirs is None:
        ignore_dirs = ['.git', '.venv', '__pycache__', 'runs', 'results', 'build', 'dist', '*.egg-info']

    output_lines.append(f"{root_dir.name}/")

    _generate_directory_lines(root_dir, 0, indent_str, ignore_dirs, max_depth, output_lines)
    return output_lines

def _generate_directory_lines(current_path, level, indent_str, ignore_dirs, max_depth, output_lines):
    """Helper recursive function to generate directory content lines."""
    if max_depth is not None and level >= max_depth:
        return

    items = sorted(list(current_path.iterdir()), key=lambda p: (p.is_file(), p.name.lower()))

    for i, item_path in enumerate(items):
        is_last_item = (i == len(items) - 1)
        prefix = indent_str * level
        connector = '└── ' if is_last_item else '├── '

        if item_path.is_dir():
            # Check against simple names and also glob patterns in ignore_dirs
            should_ignore = item_path.name in ignore_dirs or \
                            item_path.name.startswith('.') or \
                            any(item_path.match(pattern) for pattern in ignore_dirs if '*' in pattern)
            
            if should_ignore:
                continue
                
            output_lines.append(f"{prefix}{connector}{item_path.name}/")
            _generate_directory_lines(item_path, level + 1, indent_str, ignore_dirs, max_depth, output_lines)
        else:
            output_lines.append(f"{prefix}{connector}{item_path.name}")


if __name__ == "__main__":
    # --- How to use it ---

    # Get the project root directory
    SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
    # Adjust PROJECT_ROOT based on where this script is located
    # If src/test_structure.py or src/data_utils/test_structure.py:
    PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()
    # If src/test_structure.py (and src is directly under project root):
    # PROJECT_ROOT = SCRIPT_DIR.parent.resolve()

    # Define the output file path (in the project root)
    output_file_name = "project_structure.txt"
    output_file_path = PROJECT_ROOT / output_file_name

    print(f"Generating project structure for: {PROJECT_ROOT}")
    
    # Generate the structure lines
    structure_lines = generate_project_structure(PROJECT_ROOT)

    # Write the lines to the output file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for line in structure_lines:
                f.write(line + '\n')
        print(f"Project structure saved to: {output_file_path}")
    except Exception as e:
        print(f"Error writing to file {output_file_path}: {e}")

    # Example: Generating structure for a subfolder with max_depth
    # data_folder = PROJECT_ROOT / 'data'
    # data_structure_output_path = PROJECT_ROOT / 'data_structure.txt'
    # if data_folder.exists():
    #     print(f"\nGenerating data folder structure (max_depth=2) for: {data_folder}")
    #     data_structure_lines = generate_project_structure(data_folder, max_depth=2, ignore_dirs=['.git', '.venv', '__pycache__'])
    #     try:
    #         with open(data_structure_output_path, 'w', encoding='utf-8') as f:
    #             for line in data_structure_lines:
    #                 f.write(line + '\n')
    #         print(f"Data folder structure saved to: {data_structure_output_path}")
    #     except Exception as e:
    #         print(f"Error writing to file {data_structure_output_path}: {e}")
    # else:
    #     print(f"Data folder not found at: {data_folder}")