# src/detection/train_mmdet.py
import subprocess
import pathlib
import os
import sys

# Define paths relative to this script file
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve() # Assumes script is in ProjectRoot/src/detection/
MMDET_ROOT = PROJECT_ROOT / 'mmdetection' # Path to your cloned mmdetection repository
MMDET_TRAIN_SCRIPT = MMDET_ROOT / 'tools' / 'train.py'

if not MMDET_TRAIN_SCRIPT.exists():
    print(f"ERROR: MMDetection train script not found at {MMDET_TRAIN_SCRIPT}")
    sys.exit(1)

experiments = {
    "faster_rcnn_r50_romanian_coco": {
        "config_file": PROJECT_ROOT / "src/detection/configs/mmdetection/faster_rcnn_r50_fpn_1x_lp_romanian.py",
        "work_dir": PROJECT_ROOT / "results/MMDet_FasterRCNN/romanian_coco", # Changed work_dir name
        "gpu_ids": [0],
    },
    "faster_rcnn_r50_spanish_coco": {
        "config_file": PROJECT_ROOT / "src/detection/configs/mmdetection/faster_rcnn_r50_fpn_1x_lp_spanish.py",
        "work_dir": PROJECT_ROOT / "results/MMDet_FasterRCNN/spanish_coco", # Changed work_dir name
        "gpu_ids": [0],
    },
}

def run_mmdetection_training(experiment_name):
    # ... (rest of the function is the same as previously provided)
    if experiment_name not in experiments:
        print(f"Error: Experiment '{experiment_name}' not defined."); return
    config = experiments[experiment_name]
    config_file = config["config_file"]; work_dir = config["work_dir"]; gpu_ids = config["gpu_ids"]
    if not config_file.exists(): print(f"ERROR: Config file not found: {config_file}"); return
    work_dir.mkdir(parents=True, exist_ok=True)
    python_executable = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    command = [str(python_executable), str(MMDET_TRAIN_SCRIPT), str(config_file), "--work-dir", str(work_dir)]
    env = os.environ.copy()
    if gpu_ids: env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    print(f"\n--- Starting MMDetection Training: {experiment_name} ---")
    print(f"Command: {' '.join(command)}")
    process = subprocess.Popen(command, env=env, cwd=str(PROJECT_ROOT))
    process.wait()
    if process.returncode == 0: print(f"--- MMDetection Training '{experiment_name}' finished successfully. ---")
    else: print(f"--- MMDetection Training '{experiment_name}' failed with exit code {process.returncode}. ---")


if __name__ == "__main__":
    # Ensure COCO JSON labels are generated
    # print("Ensure COCO JSON labels are available in data/processed/labels/coco/")
    # print("Run: python src/data_utils/create_coco_labels.py if not done.")

    experiment_to_run = "faster_rcnn_r50_romanian_coco"
    # experiment_to_run = "faster_rcnn_r50_spanish_coco"

    if experiment_to_run in experiments:
        run_mmdetection_training(experiment_to_run)
    else:
        print(f"Experiment '{experiment_to_run}' not defined.")