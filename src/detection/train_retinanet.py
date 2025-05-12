# src/detection/train_retinanet.py

import torch
import torchvision
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights, RetinaNetHead
# from torchvision.models.detection.anchor_utils import AnchorGenerator # Usually not needed if using existing model's anchor setup
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader, Dataset

import os
import json
from PIL import Image
import pathlib
import time
import sys # For sys.exit in case of errors
from tqdm import tqdm # Ensure tqdm is imported

# Attempt to import pycocotools - not strictly needed for training, but good for consistency
try:
    from pycocotools.coco import COCO
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("WARNING: pycocotools not found. It's not strictly needed for training RetinaNet, but is for evaluation.")


# --- Configuration ---
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
IMAGE_DIR_BASE = DATA_DIR / 'images'
ANNOTATION_DIR_BASE = DATA_DIR / 'labels' / 'coco'

# ----- PARAMETERS TO SET -----
DATASET_NAME_TRAIN = 'spanish' # or 'spanish'
SPLIT_TRAIN = 'train'
DATASET_NAME_VALID = 'spanish' # or 'spanish'
SPLIT_VALID = 'test'

NUM_CLASSES = 2  # 1 class (license_plate) + 1 background
NUM_EPOCHS = 15
BATCH_SIZE_TRAIN = 2
BATCH_SIZE_VALID = 1
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
CHECKPOINT_SAVE_DIR = PROJECT_ROOT / 'results' / 'RetinaNet_checkpoints'
CHECKPOINT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
# ----------------------------

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {DEVICE}")

# --- Custom Dataset Class ---
class LicensePlateDataset(Dataset):
    def __init__(self, image_dir_base, coco_annot_path, dataset_name, split_name, transforms=None):
        self.image_dir = image_dir_base / dataset_name / split_name
        self.transforms = transforms
        self.imgs_info = [] 
        self.annotations = {} 

        print(f"Loading COCO annotations from: {coco_annot_path}")
        if not coco_annot_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {coco_annot_path}")

        with open(coco_annot_path, 'r') as f:
            coco_data = json.load(f)

        img_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
        print(f"Categories found in COCO: {category_map}")

        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        for img_id, filename in img_id_to_filename.items():
            if img_id in self.annotations: 
                full_img_path = IMAGE_DIR_BASE / filename
                if full_img_path.exists():
                    pil_img_temp = Image.open(full_img_path)
                    width, height = pil_img_temp.size
                    pil_img_temp.close()
                    self.imgs_info.append({'path': full_img_path, 'id': img_id, 'file_name': filename, 'height': height, 'width': width})
                else:
                    print(f"Warning: Image file {full_img_path} listed in COCO not found on disk. Skipping.")

        print(f"Found {len(self.imgs_info)} images with annotations in {coco_annot_path.name}")
        if not self.imgs_info:
            print(f"Warning: No images with annotations loaded for {dataset_name}/{split_name}. Check paths and COCO file.")

    def __getitem__(self, idx):
        img_info = self.imgs_info[idx]
        img_path = img_info['path']
        image_id = img_info['id']

        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}.")
            return None, None 

        target = {}
        boxes = []
        labels = []

        if image_id in self.annotations:
            for ann in self.annotations[image_id]:
                xmin, ymin, width, height = ann['bbox']
                xmax = xmin + width
                ymax = ymin + height
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(ann['category_id']) 

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes.shape[0] > 0: # Check if there are any boxes before accessing
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # Ensure xmax > xmin and ymax > ymin
                boxes[:, 2] = torch.maximum(boxes[:, 2], boxes[:, 0] + 1e-4) # Add small epsilon
                boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1] + 1e-4) # Add small epsilon
        
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([image_id]) 
        
        if self.transforms:
            img_tensor = T.PILToTensor()(img)
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target

    def __len__(self):
        return len(self.imgs_info)

# --- Transforms ---
def get_transform(train):
    transforms = []
    transforms.append(T.ToDtype(torch.float, scale=True))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# --- Collate Function ---
def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: 
        return None, None
    return tuple(zip(*batch))

# --- Model Definition for RetinaNet (Corrected) ---
def get_retinanet_model(num_classes_for_head):
    weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.retinanet_resnet50_fpn(weights=weights)

    # Get the number of input channels FOR THE HEAD from the FPN backbone's output
    in_channels = model.backbone.out_channels  # <--- CORRECTED LINE

    # Get the number of anchors per location
    # This usually comes from the anchor_generator associated with the pre-trained model,
    # or from the default head's classification sub-head if it's inspected before replacement.
    try:
        # Try to get it from the pre-existing head's classification sub-head's num_anchors property
        num_anchors = model.head.classification_head.num_anchors
    except AttributeError:
        # Fallback if the above path isn't available (e.g. if head structure changes in torchvision versions)
        # Use the model's anchor_generator. num_anchors_per_location() returns a list,
        # one for each FPN level. They are typically the same.
        print("Falling back to model.anchor_generator.num_anchors_per_location()[0] for num_anchors")
        num_anchors = model.anchor_generator.num_anchors_per_location()[0]
    
    # Replace the pre-trained head with a new one
    # num_classes_for_head for RetinaNetHead in torchvision is num_classes including background
    model.head = RetinaNetHead(in_channels=in_channels,
                               num_anchors=num_anchors,
                               num_classes=num_classes_for_head)
    return model

# --- Main Training Function ---
def main():
    train_annot_file = ANNOTATION_DIR_BASE / f"{DATASET_NAME_TRAIN}_{SPLIT_TRAIN}.json"
    valid_annot_file = ANNOTATION_DIR_BASE / f"{DATASET_NAME_VALID}_{SPLIT_VALID}.json"

    dataset_train = LicensePlateDataset(IMAGE_DIR_BASE, train_annot_file, DATASET_NAME_TRAIN, SPLIT_TRAIN, get_transform(train=True))
    dataset_valid = LicensePlateDataset(IMAGE_DIR_BASE, valid_annot_file, DATASET_NAME_VALID, SPLIT_VALID, get_transform(train=False))

    if not dataset_train or not len(dataset_train.imgs_info): # Check if imgs_info populated
        print(f"Training dataset for {DATASET_NAME_TRAIN}/{SPLIT_TRAIN} is empty or failed to load. Exiting.")
        sys.exit(1)
    if not dataset_valid or not len(dataset_valid.imgs_info): # Check if imgs_info populated
        print(f"Validation dataset for {DATASET_NAME_VALID}/{SPLIT_VALID} is empty or failed to load. Exiting.")
        sys.exit(1)

    data_loader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=2,
        collate_fn=collate_fn
    )
    data_loader_valid = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE_VALID, shuffle=False, num_workers=2,
        collate_fn=collate_fn
    )

    model = get_retinanet_model(NUM_CLASSES) # NUM_CLASSES includes background
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print("--- Starting RetinaNet Training ---")
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        train_loss_accum = 0.0
        train_loss_classification_accum = 0.0
        train_loss_bbox_reg_accum = 0.0
        
        progress_bar = tqdm(data_loader_train, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for i, batch_data in enumerate(progress_bar):
            if batch_data is None or batch_data[0] is None: 
                print(f"Skipping empty or problematic batch {i+1}")
                continue
            
            images, targets = batch_data
            if not images or not targets: # Further check if collate_fn returned empty
                print(f"Skipping batch {i+1} due to no valid data after collation.")
                continue

            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            try:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                train_loss_accum += losses.item()
                train_loss_classification_accum += loss_dict.get('classification', torch.tensor(0.0)).item()
                train_loss_bbox_reg_accum += loss_dict.get('bbox_regression', torch.tensor(0.0)).item()   
                
                progress_bar.set_postfix(loss=f"{losses.item():.4f}")

            except Exception as e:
                print(f"Error during training batch {i+1}: {e}")
                # Log problematic image_ids if possible
                # for target in targets:
                # print(f"  Problem with image_id: {target['image_id'].item()}")
                continue 

        avg_train_loss = train_loss_accum / len(data_loader_train) if len(data_loader_train) > 0 else 0
        avg_train_loss_cls = train_loss_classification_accum / len(data_loader_train) if len(data_loader_train) > 0 else 0
        avg_train_loss_bbox = train_loss_bbox_reg_accum / len(data_loader_train) if len(data_loader_train) > 0 else 0
        
        model.eval()
        val_loss_accum = 0.0
        val_loss_classification_accum = 0.0
        val_loss_bbox_reg_accum = 0.0
        
        progress_bar_val = tqdm(data_loader_valid, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Valid]")
        with torch.no_grad():
            for batch_data_val in progress_bar_val:
                if batch_data_val is None or batch_data_val[0] is None:
                    print(f"Skipping empty or problematic validation batch")
                    continue
                
                images_val, targets_val = batch_data_val
                if not images_val or not targets_val:
                    print(f"Skipping validation batch due to no valid data after collation.")
                    continue

                images_val = list(img.to(DEVICE) for img in images_val)
                targets_val = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets_val]

                try:
                    model.train() # Temporarily set to train to get losses
                    loss_dict_val = model(images_val, targets_val)
                    model.eval() 

                    losses_val = sum(loss for loss in loss_dict_val.values())
                    val_loss_accum += losses_val.item()
                    val_loss_classification_accum += loss_dict_val.get('classification', torch.tensor(0.0)).item()
                    val_loss_bbox_reg_accum += loss_dict_val.get('bbox_regression', torch.tensor(0.0)).item()
                    progress_bar_val.set_postfix(loss=f"{losses_val.item():.4f}")
                except Exception as e:
                    print(f"Error during validation batch: {e}")
                    model.eval() 
                    continue

        avg_val_loss = val_loss_accum / len(data_loader_valid) if len(data_loader_valid) > 0 else 0
        avg_val_loss_cls = val_loss_classification_accum / len(data_loader_valid) if len(data_loader_valid) > 0 else 0
        avg_val_loss_bbox = val_loss_bbox_reg_accum / len(data_loader_valid) if len(data_loader_valid) > 0 else 0

        if lr_scheduler:
            lr_scheduler.step()

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Duration: {epoch_duration:.2f}s")
        print(f"  Avg Train Loss: {avg_train_loss:.4f} (Cls: {avg_train_loss_cls:.4f}, BoxReg: {avg_train_loss_bbox:.4f})")
        print(f"  Avg Valid Loss: {avg_val_loss:.4f} (Cls: {avg_val_loss_cls:.4f}, BoxReg: {avg_val_loss_bbox:.4f})")

        checkpoint_path = CHECKPOINT_SAVE_DIR / f"retinanet_resnet50_fpn_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
            'train_loss': avg_train_loss,
            'valid_loss': avg_val_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    print("--- RetinaNet Training Finished ---")

if __name__ == "__main__":
    # To run directly, you might modify the parameters at the top of the script
    # or integrate a simple config dictionary here as done in evaluate_*.py scripts.
    # For now, it assumes parameters are set in the global section.
    main()