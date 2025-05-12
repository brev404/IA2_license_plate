# src/detection/evaluate_retinanet.py

import torch
import torchvision
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights, RetinaNetHead
# from torchvision.models.detection.anchor_utils import AnchorGenerator # Not strictly needed if using model.anchor_generator
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader, Dataset

import os
import json
from PIL import Image
import pathlib
import time
import argparse
from tqdm import tqdm
import sys

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("WARNING: pycocotools not found. Please install it to run COCO mAP evaluation: pip install pycocotools")

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
IMAGE_DIR_BASE = DATA_DIR / 'images'
ANNOTATION_DIR_BASE = DATA_DIR / 'labels' / 'coco'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'RetinaNet_evaluation'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class LicensePlateDataset(Dataset):
    def __init__(self, image_dir_base, coco_annot_path, dataset_name, split_name, transforms=None):
        self.image_dir = image_dir_base / dataset_name / split_name
        self.transforms = transforms
        self.imgs_info = []
        self.coco_gt_for_eval = None 

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

def get_transform(train=False): # train argument is not used here, but kept for consistency
    transforms = []
    transforms.append(T.ToDtype(torch.float, scale=True))
    return T.Compose(transforms)

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
    return tuple(zip(*batch))

# --- Corrected Model Definition for RetinaNet Evaluation ---
def get_retinanet_model_for_eval(num_classes_for_head, checkpoint_path=None):
    weights_arg = None 
    if not checkpoint_path:
        weights_arg = RetinaNet_ResNet50_FPN_Weights.DEFAULT
        print("Loading RetinaNet model with default COCO pre-trained weights for evaluation (no checkpoint specified).")
    
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=weights_arg)

    # Get the number of input channels FOR THE HEAD from the FPN backbone's output
    in_channels = model.backbone.out_channels  # <--- CORRECTED LINE

    # Get the number of anchors per location
    try:
        num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    except AttributeError:
        print("Warning: Could not get num_anchors from model.anchor_generator. Assuming default 9.")
        num_anchors = 9 
    
    # Replace the head if we are training for custom classes (always the case here unless just running COCO default)
    model.head = RetinaNetHead(in_channels=in_channels,
                               num_anchors=num_anchors,
                               num_classes=num_classes_for_head) # num_classes includes background

    if checkpoint_path:
        print(f"Loading trained RetinaNet weights from: {checkpoint_path}")
        try:
            # It's safer to load weights_only=True if you trust the source or handle potential risks.
            # For this project context, assuming the checkpoint is from your own training.
            checkpoint = torch.load(str(checkpoint_path), map_location=DEVICE) # weights_only=False (default) or True
            model.load_state_dict(checkpoint['model_state_dict'])
            print("RetinaNet model weights loaded successfully from checkpoint.")
        except Exception as e:
            print(f"Error loading RetinaNet model weights from checkpoint: {e}.")
            if not weights_arg: 
                 print("Model will use randomly initialized weights for the new head.")
    elif not weights_arg:
        print("Warning: No checkpoint provided and not loading COCO default weights. Model is randomly initialized.")
        
    return model

@torch.no_grad()
def evaluate_coco(model, data_loader, device, coco_gt_api, output_eval_file_path):
    if not PYCOCOTOOLS_AVAILABLE:
        print("Cannot run COCO evaluation because pycocotools is not available.")
        return

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1) 
    cpu_device = torch.device("cpu")
    model.eval()
    coco_results = []

    print("Running inference for COCO evaluation (RetinaNet)...")
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
    
    temp_results_json_path = pathlib.Path("temp_retinanet_coco_results.json")
    with open(temp_results_json_path, "w") as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt_api.loadRes(str(temp_results_json_path))
    coco_eval = COCOeval(coco_gt_api, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    print("\nRetinaNet COCO Evaluation Summary (Printed to Console):")
    coco_eval.summarize()

    stats_names = ['AP_IoU_0.50_0.95_area_all_maxDets_100', 'AP_IoU_0.50_area_all_maxDets_100', 
                   'AP_IoU_0.75_area_all_maxDets_100', 'AP_IoU_0.50_0.95_area_small_maxDets_100', 
                   'AP_IoU_0.50_0.95_area_medium_maxDets_100', 'AP_IoU_0.50_0.95_area_large_maxDets_100', 
                   'AR_IoU_0.50_0.95_area_all_maxDets_1', 'AR_IoU_0.50_0.95_area_all_maxDets_10', 
                   'AR_IoU_0.50_0.95_area_all_maxDets_100', 'AR_IoU_0.50_0.95_area_small_maxDets_100', 
                   'AR_IoU_0.50_0.95_area_medium_maxDets_100', 'AR_IoU_0.50_0.95_area_large_maxDets_100']
    
    eval_summary_str = "RetinaNet COCO Evaluation Metrics:\n"
    for i_stat, name in enumerate(stats_names): # Use a different loop variable
        eval_summary_str += f"{name}: {coco_eval.stats[i_stat]:.4f}\n"
    
    try:
        output_eval_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_eval_file_path, 'w') as f_out: # Use a different file handle variable
            f_out.write(eval_summary_str)
        print(f"\nRetinaNet COCO Evaluation metrics saved to: {output_eval_file_path}")
    except Exception as e:
        print(f"Error saving RetinaNet evaluation metrics to file: {e}")

    if temp_results_json_path.exists():
        temp_results_json_path.unlink()

    torch.set_num_threads(n_threads)
    return coco_eval

def main(args_namespace): # Renamed args to args_namespace for clarity with the class
    if not PYCOCOTOOLS_AVAILABLE:
        sys.exit("Evaluation requires pycocotools. Please install it.")

    num_classes = args_namespace.num_classes
    dataset_name_eval = args_namespace.dataset_name
    split_eval = args_namespace.split
    model_checkpoint_path = pathlib.Path(args_namespace.model_checkpoint) if args_namespace.model_checkpoint else None
    batch_size_eval = args_namespace.batch_size
    
    eval_annot_file = ANNOTATION_DIR_BASE / f"{dataset_name_eval}_{split_eval}.json"

    print(f"RetinaNet Evaluation on: {dataset_name_eval} / {split_eval}")
    print(f"Annotation file: {eval_annot_file}")
    if model_checkpoint_path and model_checkpoint_path.exists():
        print(f"Model checkpoint: {model_checkpoint_path}")
    elif model_checkpoint_path:
        print(f"Model checkpoint specified but not found: {model_checkpoint_path}. Using default COCO weights if applicable.")
    else:
        print("No specific model checkpoint provided, using default COCO pre-trained RetinaNet.")

    model_name_stem = "default_retinanet_coco_pretrained"
    if model_checkpoint_path and model_checkpoint_path.exists():
        model_name_stem = model_checkpoint_path.stem

    eval_output_dir = RESULTS_DIR / model_name_stem
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    output_eval_file = eval_output_dir / f"eval_on_{dataset_name_eval}_{split_eval}_metrics.txt"

    dataset_eval = LicensePlateDataset(IMAGE_DIR_BASE, eval_annot_file, dataset_name_eval, split_eval, get_transform(train=False))
    if not dataset_eval or not dataset_eval.imgs_info: 
        print(f"Evaluation dataset {dataset_name_eval}/{split_eval} is empty or failed to load properly. Exiting.")
        return

    data_loader_eval = DataLoader(
        dataset_eval, batch_size=batch_size_eval, shuffle=False, num_workers=2,
        collate_fn=collate_fn
    )

    model = get_retinanet_model_for_eval(num_classes, model_checkpoint_path)
    model.to(DEVICE)

    evaluate_coco(model, data_loader_eval, DEVICE, dataset_eval.coco_gt_for_eval, output_eval_file)

if __name__ == "__main__":
    # Logic to differentiate between direct run and CLI run
    is_direct_run = not any(arg.startswith('--model_checkpoint') for arg in sys.argv[1:]) and \
                    len(sys.argv) == 1 # True if only 'script_name.py' is passed

    if is_direct_run:
        print("Running RetinaNet evaluation script directly with hardcoded config...")
        # --- Define your evaluation parameters directly here ---
        config = {
            "model_checkpoint": "results/RetinaNet_checkpoints/retinanet_resnet50_fpn_epoch_15.pth", # <--- !!! EXAMPLE: SET THIS !!!
            "dataset_name": "spanish", # or "spanish"
            "split": "test",           # or "test"
            "num_classes": 2,           # Includes background
            "batch_size": 1
        }
        # ---------------------------------------------------------
        class ArgsNamespace: # Define it locally if not globally available
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        args_for_main = ArgsNamespace(**config)

        # Basic check for the hardcoded model_checkpoint
        model_path_to_check = pathlib.Path(config["model_checkpoint"])
        if not config["model_checkpoint"] or "YOUR_RETINANET_MODEL.pth" in config["model_checkpoint"]: # Placeholder check
            print(f"ERROR: Please set the 'model_checkpoint' path in the script's hardcoded config. Currently: {config['model_checkpoint']}")
        elif not model_path_to_check.exists():
            print(f"ERROR: Model checkpoint not found at: {model_path_to_check}")
        else:
            if not PYCOCOTOOLS_AVAILABLE:
                print("Exiting because pycocotools is not available. Please install it.")
            else:
                main(args_for_main)
    else:
        # If running with CLI arguments
        parser = argparse.ArgumentParser(description="Evaluate a trained RetinaNet model.")
        parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained RetinaNet model .pth checkpoint file.")
        parser.add_argument("--dataset_name", type=str, default="romanian", help="Dataset name for evaluation.")
        parser.add_argument("--split", type=str, default="valid", help="Dataset split for evaluation.")
        parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (object_class + background).")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
        args = parser.parse_args()
        main(args)