# src/test_structure.py (or wherever you save it)
import os
import pathlib

def generate_project_structure(root_dir_path, indent_str='|   ', ignore_dirs=None, max_depth=None, max_files_per_dir=100):
    """
    Generates a list of strings representing the directory structure.

    Args:
        root_dir_path (str or pathlib.Path): The root directory to start from.
        indent_str (str): String to use for indentation.
        ignore_dirs (list, optional): A list of directory names or glob patterns
                                      to ignore. Defaults to common temporary/large dirs.
        max_depth (int, optional): Maximum depth to traverse. None means no limit.
        max_files_per_dir (int, optional): Maximum number of files a directory can
                                           contain directly for its contents to be
                                           listed. Defaults to 100. If exceeded,
                                           the directory is listed but its contents are not.

    Returns:
        list: A list of strings, each string is a line in the structure tree.
    """
    root_dir = pathlib.Path(root_dir_path).resolve()
    output_lines = []

    if ignore_dirs is None:
        # Added '.mypy_cache', 'node_modules' as common large/cache dirs
        ignore_dirs = [
            '.git', '.venv', 'venv', '__pycache__', 'runs', 'results',
            'build', 'dist', '*.egg-info', '.mypy_cache', 'node_modules'
        ]

    output_lines.append(f"{root_dir.name}/")

    _generate_directory_lines(
        root_dir, 0, indent_str, ignore_dirs, max_depth, output_lines, max_files_per_dir
    )
    return output_lines

def _generate_directory_lines(current_path, level, indent_str, ignore_dirs, max_depth, output_lines, max_files_per_dir):
    """Helper recursive function to generate directory content lines."""
    if max_depth is not None and level >= max_depth:
        return

    try:
        # List items first to handle potential permission errors early
        items_iterator = current_path.iterdir()
        items = sorted(list(items_iterator), key=lambda p: (p.is_file(), p.name.lower()))
    except (PermissionError, FileNotFoundError, OSError) as e:
        # Could not list directory contents, add a note
        indent = indent_str * level
        output_lines.append(f"{indent}└── [Error accessing contents: {e}]")
        return # Stop processing this branch

    for i, item_path in enumerate(items):
        is_last_item = (i == len(items) - 1)
        prefix = indent_str * level
        connector = '└── ' if is_last_item else '├── '

        if item_path.is_dir():
            # Check against simple names, hidden dirs, and glob patterns in ignore_dirs
            should_ignore = item_path.name in ignore_dirs or \
                            item_path.name.startswith('.') or \
                            any(item_path.match(pattern) for pattern in ignore_dirs if '*' in pattern)

            if should_ignore:
                continue

            output_lines.append(f"{prefix}{connector}{item_path.name}/")

            # --- Check file count before recursing ---
            try:
                # Count only files directly within this directory
                num_files = sum(1 for child in item_path.iterdir() if child.is_file())

                if num_files > max_files_per_dir:
                    # If too many files, add a note and *do not* recurse
                    placeholder_prefix = indent_str * (level + 1)
                    output_lines.append(f"{placeholder_prefix}└── [... content skipped (> {max_files_per_dir} files)]")
                    continue # Skip to the next item in the parent directory
                else:
                    # File count is within limit, recurse normally
                    _generate_directory_lines(
                        item_path, level + 1, indent_str, ignore_dirs,
                        max_depth, output_lines, max_files_per_dir
                    )

            except (PermissionError, FileNotFoundError, OSError) as e:
                 # Could not count files in the subdirectory, add a note and don't recurse
                 placeholder_prefix = indent_str * (level + 1)
                 output_lines.append(f"{placeholder_prefix}└── [Error counting files: {e}, content skipped]")
                 continue # Skip recursion for this directory

        else: # It's a file
             # Check if file itself matches ignore patterns (e.g., '*.log') - less common usage
             should_ignore_file = any(item_path.match(pattern) for pattern in ignore_dirs if '*' in pattern)
             if not should_ignore_file:
                output_lines.append(f"{prefix}{connector}{item_path.name}")


if __name__ == "__main__":
    # --- How to use it ---

    # Get the project root directory
    SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

    # --- IMPORTANT: Adjust PROJECT_ROOT based on where this script is ---
    # Option 1: If this script is in project_root/src/utils/test_structure.py
    PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()
    # Option 2: If this script is in project_root/src/test_structure.py
    # PROJECT_ROOT = SCRIPT_DIR.parent.resolve()
    # Option 3: If this script is directly in the project root
    # PROJECT_ROOT = SCRIPT_DIR.resolve()
    # --- Choose the correct option above for your structure ---

    # Define the output file path (in the project root)
    output_file_name = "project_structure.txt"
    output_file_path = PROJECT_ROOT / output_file_name

    print(f"Generating project structure for: {PROJECT_ROOT}")
    print(f"Ignoring directories: {generate_project_structure.__defaults__[0]}") # Print default ignores
    print(f"Skipping content of directories with more than {generate_project_structure.__defaults__[2]} files.") # Print default file limit

    # --- Generate the structure lines ---
    # You can override the defaults here, e.g.:
    # structure_lines = generate_project_structure(PROJECT_ROOT, max_files_per_dir=50, ignore_dirs=['.git', 'node_modules'])
    structure_lines = generate_project_structure(
        PROJECT_ROOT,
        max_files_per_dir=100 # Explicitly set or change the file limit here if needed
    )

    # Write the lines to the output file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for line in structure_lines:
                f.write(line + '\n')
        print(f"\nProject structure saved to: {output_file_path}")
    except Exception as e:
        print(f"\nError writing to file {output_file_path}: {e}")

    # --- Example: Generating structure for a subfolder ---
    # data_folder = PROJECT_ROOT / 'data'
    # data_structure_output_path = PROJECT_ROOT / 'data_structure_detailed.txt'
    # if data_folder.exists() and data_folder.is_dir():
    #     print(f"\nGenerating detailed data folder structure for: {data_folder}")
    #     # Example: Use a higher file limit for this specific folder if needed
    #     data_structure_lines = generate_project_structure(
    #         data_folder,
    #         max_depth=None, # No depth limit for this subfolder
    #         max_files_per_dir=500 # Allow more files here perhaps
    #         # inherit default ignore_dirs unless specified
    #     )
    #     try:
    #         with open(data_structure_output_path, 'w', encoding='utf-8') as f:
    #             for line in data_structure_lines:
    #                 f.write(line + '\n')
    #         print(f"Data folder structure saved to: {data_structure_output_path}")
    #     except Exception as e:
    #         print(f"Error writing to file {data_structure_output_path}: {e}")
    # else:
    #     print(f"\nData folder not found or is not a directory at: {data_folder}")