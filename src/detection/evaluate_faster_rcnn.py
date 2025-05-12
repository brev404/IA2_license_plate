# src/detection/evaluate_faster_rcnn.py

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader, Dataset

import os
import json
from PIL import Image
import pathlib
import time
import argparse
from tqdm import tqdm # Make sure this is imported
import sys

# Attempt to import pycocotools
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("WARNING: pycocotools not found. Please install it to run COCO mAP evaluation: pip install pycocotools")


# --- Re-use Dataset and Model Definition from train_faster_rcnn.py ---
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
IMAGE_DIR_BASE = DATA_DIR / 'images'
ANNOTATION_DIR_BASE = DATA_DIR / 'labels' / 'coco'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'FasterRCNN_evaluation' # Define base results directory

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class LicensePlateDataset(Dataset):
    def __init__(self, image_dir_base, coco_annot_path, dataset_name, split_name, transforms=None):
        self.image_dir = image_dir_base / dataset_name / split_name
        self.transforms = transforms
        self.imgs_info = []
        self.annotations = {}
        self.coco_gt_for_eval = None # Will hold COCO ground truth object

        if not PYCOCOTOOLS_AVAILABLE:
            print("Cannot initialize dataset for COCO evaluation without pycocotools.")
            return

        print(f"Loading COCO annotations from: {coco_annot_path}")
        if not coco_annot_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {coco_annot_path}")

        self.coco_gt_for_eval = COCO(str(coco_annot_path))
        self.img_ids_from_coco = sorted(self.coco_gt_for_eval.getImgIds())

        img_id_to_coco_img_obj = {img_id: self.coco_gt_for_eval.loadImgs(img_id)[0] for img_id in self.img_ids_from_coco}

        for img_id in self.img_ids_from_coco:
            coco_img_obj = img_id_to_coco_img_obj[img_id]
            filename = coco_img_obj['file_name']
            full_img_path = IMAGE_DIR_BASE / filename
            if full_img_path.exists():
                self.imgs_info.append({'path': full_img_path, 'id': img_id, 'file_name': filename, 'height': coco_img_obj['height'], 'width': coco_img_obj['width']})
            else:
                print(f"Warning: Image file {full_img_path} for image ID {img_id} listed in COCO not found. Skipping.")
        
        print(f"Found {len(self.imgs_info)} images with annotations for evaluation.")
        if not self.imgs_info:
            print(f"Warning: No images loaded for evaluation for {dataset_name}/{split_name}.")

    def __getitem__(self, idx):
        img_info = self.imgs_info[idx]
        img_path = img_info['path']
        
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}.")
            return None, None 

        target = {}
        target["image_id"] = torch.tensor([img_info['id']])
        target["orig_size"] = torch.as_tensor([int(img_info['height']), int(img_info['width'])])

        if self.transforms:
            img_tensor = T.PILToTensor()(img)
            img_tensor, _ = self.transforms(img_tensor, target) 

        return img_tensor, target

    def __len__(self):
        return len(self.imgs_info)

def get_transform(train):
    transforms = []
    transforms.append(T.ToDtype(torch.float, scale=True))
    return T.Compose(transforms)

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
    return tuple(zip(*batch))

def get_model(num_classes, checkpoint_path=None):
    weights = None 
    if not checkpoint_path:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        print("Loading model with default COCO pre-trained weights.")
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if checkpoint_path:
        print(f"Loading trained weights from: {checkpoint_path}")
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location=DEVICE) # Ensure checkpoint_path is str
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}. Using model with specified architecture but random/default weights.")
            if not weights:
                 print("Re-loading model with default COCO pre-trained weights as fallback.")
                 model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
                 in_features = model.roi_heads.box_predictor.cls_score.in_features
                 model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

@torch.no_grad()
def evaluate_coco(model, data_loader, device, coco_gt_api, output_eval_file_path): # Added output_eval_file_path
    if not PYCOCOTOOLS_AVAILABLE:
        print("Cannot run COCO evaluation because pycocotools is not available.")
        return

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1) 
    cpu_device = torch.device("cpu")
    model.eval()
    coco_results = []

    print("Running inference for COCO evaluation...")
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        if images is None or targets is None: continue
        images = list(img.to(device) for img in images)
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        for i, output in enumerate(outputs):
            image_id = targets[i]['image_id'].item()
            boxes = output['boxes']
            labels = output['labels']
            scores = output['scores']

            for box_idx in range(boxes.shape[0]):
                xmin, ymin, xmax, ymax = boxes[box_idx].tolist()
                width = xmax - xmin
                height = ymax - ymin
                coco_results.append({
                    'image_id': image_id,
                    'category_id': labels[box_idx].item(),
                    'bbox': [xmin, ymin, width, height],
                    'score': scores[box_idx].item(),
                })
    
    if not coco_results:
        print("No predictions made. Cannot evaluate.")
        torch.set_num_threads(n_threads)
        return

    print(f"Total predictions formatted for COCO: {len(coco_results)}")
    
    temp_results_json_path = pathlib.Path("temp_coco_results.json")
    with open(temp_results_json_path, "w") as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt_api.loadRes(str(temp_results_json_path))
    coco_eval = COCOeval(coco_gt_api, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    print("\nCOCO Evaluation Summary (Printed to Console):")
    coco_eval.summarize()

    # Save the stats to the specified file
    stats_names = ['AP_IoU_0.50_0.95_area_all_maxDets_100', 'AP_IoU_0.50_area_all_maxDets_100', 
                   'AP_IoU_0.75_area_all_maxDets_100', 'AP_IoU_0.50_0.95_area_small_maxDets_100', 
                   'AP_IoU_0.50_0.95_area_medium_maxDets_100', 'AP_IoU_0.50_0.95_area_large_maxDets_100', 
                   'AR_IoU_0.50_0.95_area_all_maxDets_1', 'AR_IoU_0.50_0.95_area_all_maxDets_10', 
                   'AR_IoU_0.50_0.95_area_all_maxDets_100', 'AR_IoU_0.50_0.95_area_small_maxDets_100', 
                   'AR_IoU_0.50_0.95_area_medium_maxDets_100', 'AR_IoU_0.50_0.95_area_large_maxDets_100']
    
    eval_summary_str = "COCO Evaluation Metrics:\n"
    for i, name in enumerate(stats_names):
        eval_summary_str += f"{name}: {coco_eval.stats[i]:.4f}\n"
    
    try:
        output_eval_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        with open(output_eval_file_path, 'w') as f:
            f.write(eval_summary_str)
        print(f"\nCOCO Evaluation metrics saved to: {output_eval_file_path}")
    except Exception as e:
        print(f"Error saving evaluation metrics to file: {e}")


    if temp_results_json_path.exists():
        temp_results_json_path.unlink()

    torch.set_num_threads(n_threads)
    return coco_eval


def main(args):
    if not PYCOCOTOOLS_AVAILABLE:
        sys.exit("Evaluation requires pycocotools. Please install it.")

    num_classes = args.num_classes
    dataset_name_eval = args.dataset_name
    split_eval = args.split
    model_checkpoint_path = pathlib.Path(args.model_checkpoint) if args.model_checkpoint else None # Ensure it's a Path object
    batch_size_eval = args.batch_size
    
    eval_annot_file = ANNOTATION_DIR_BASE / f"{dataset_name_eval}_{split_eval}.json"

    print(f"Evaluation on: {dataset_name_eval} / {split_eval}")
    print(f"Annotation file: {eval_annot_file}")
    if model_checkpoint_path:
        print(f"Model checkpoint: {model_checkpoint_path}")
    else:
        print("No specific model checkpoint provided, using default COCO pre-trained FasterRCNN_ResNet50_FPN.")

    # --- Create output directory and file path for evaluation results ---
    model_name_stem = "default_coco_pretrained"
    if model_checkpoint_path and model_checkpoint_path.exists():
        model_name_stem = model_checkpoint_path.stem # e.g., fasterrcnn_resnet50_fpn_epoch_10

    eval_output_dir = RESULTS_DIR / model_name_stem
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    output_eval_file = eval_output_dir / f"eval_on_{dataset_name_eval}_{split_eval}_metrics.txt"
    # --------------------------------------------------------------------

    dataset_eval = LicensePlateDataset(IMAGE_DIR_BASE, eval_annot_file, dataset_name_eval, split_eval, get_transform(train=False))
    if not dataset_eval or not len(dataset_eval):
        print(f"Evaluation dataset {dataset_name_eval}/{split_eval} is empty or failed to load. Exiting.")
        return

    data_loader_eval = DataLoader(
        dataset_eval, batch_size=batch_size_eval, shuffle=False, num_workers=2,
        collate_fn=collate_fn
    )

    model = get_model(num_classes, model_checkpoint_path) # Pass Path object
    model.to(DEVICE)

    evaluate_coco(model, data_loader_eval, DEVICE, dataset_eval.coco_gt_for_eval, output_eval_file) # Pass output file path


if __name__ == "__main__":
    # If running directly (not via CLI with argparse)
    if not any(arg.startswith('--model_checkpoint') for arg in sys.argv):
        print("Running script directly with hardcoded config for evaluation...")
        # --- Define your evaluation parameters directly here ---
        config = {
            "model_checkpoint": "results/FasterRCNN_checkpoints/fasterrcnn_resnet50_fpn_epoch_10.pth", # <--- !!! SET THIS !!!
            "dataset_name": "romanian",
            "split": "valid",
            "num_classes": 2,
            "batch_size": 1
        }
        # ---------------------------------------------------------
        class ArgsNamespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        args_for_main = ArgsNamespace(**config)

        if not config["model_checkpoint"] or "YOUR_CHOSEN_MODEL.pth" in config["model_checkpoint"]:
            print("ERROR: Please set the 'model_checkpoint' path in the script's hardcoded config.")
        elif not pathlib.Path(config["model_checkpoint"]).exists():
            print(f"ERROR: Model checkpoint not found at: {config['model_checkpoint']}")
        else:
            if not PYCOCOTOOLS_AVAILABLE:
                print("Exiting because pycocotools is not available. Please install it.")
            else:
                main(args_for_main)
    else:
        # If running with CLI arguments
        parser = argparse.ArgumentParser(description="Evaluate a trained Faster R-CNN model.")
        parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained model .pth checkpoint file.")
        parser.add_argument("--dataset_name", type=str, default="romanian", help="Name of the dataset to evaluate on (e.g., 'romanian', 'spanish').")
        parser.add_argument("--split", type=str, default="valid", help="Dataset split to evaluate on (e.g., 'valid', 'test').")
        parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (object_class + background).")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
        args = parser.parse_args()
        main(args)