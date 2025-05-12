# src/detection/train_faster_rcnn.py

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import v2 as T # New TorchVision transforms
from torch.utils.data import DataLoader, Dataset

import os
import json
from PIL import Image
import pathlib
import time

# --- Configuration ---
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
IMAGE_DIR_BASE = DATA_DIR / 'images'
ANNOTATION_DIR_BASE = DATA_DIR / 'labels' / 'coco' # Your COCO JSONs are here

# ----- PARAMETERS TO SET -----
DATASET_NAME_TRAIN = 'romanian' # or 'spanish'
SPLIT_TRAIN = 'train'
DATASET_NAME_VALID = 'romanian' # or 'spanish'
SPLIT_VALID = 'valid'

NUM_CLASSES = 2  # 1 class (license_plate) + 1 background
NUM_EPOCHS = 10 # Start with a few epochs to test the pipeline
BATCH_SIZE_TRAIN = 2 # Adjust based on your GPU memory
BATCH_SIZE_VALID = 1 # Usually 1 for validation unless memory allows more
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
CHECKPOINT_SAVE_DIR = PROJECT_ROOT / 'results' / 'FasterRCNN_checkpoints'
CHECKPOINT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
# ----------------------------

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {DEVICE}")

# --- Custom Dataset Class ---
class LicensePlateDataset(Dataset):
    def __init__(self, image_dir_base, coco_annot_path, dataset_name, split_name, transforms=None):
        self.image_dir = image_dir_base / dataset_name / split_name
        self.transforms = transforms
        self.imgs_info = [] # To store (image_path, image_id)
        self.annotations = {} # To store annotations keyed by image_id

        print(f"Loading COCO annotations from: {coco_annot_path}")
        if not coco_annot_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {coco_annot_path}")

        with open(coco_annot_path, 'r') as f:
            coco_data = json.load(f)

        # Create a map from image_id to image file_name and path
        img_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

        # Group annotations by image_id
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        # Populate imgs_info with images that have annotations
        for img_id, filename in img_id_to_filename.items():
            if img_id in self.annotations: # Only include images with annotations
                 # filename from COCO json is relative to 'images' dir e.g. 'romanian/train/img.jpg'
                 # so we construct full path from the project root
                full_img_path = IMAGE_DIR_BASE / filename
                if full_img_path.exists():
                    self.imgs_info.append({'path': full_img_path, 'id': img_id, 'file_name': filename})
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
            print(f"Error: Image file not found at {img_path}. Skipping this item.")
            # Return a dummy item or raise an error, depending on desired behavior
            # For now, let's try to skip by returning None and handling in collate_fn or dataloader
            return None, None # This will need careful handling in collate_fn

        target = {}
        boxes = []
        labels = []

        if image_id in self.annotations:
            for ann in self.annotations[image_id]:
                # COCO format: [xmin, ymin, width, height]
                xmin, ymin, width, height = ann['bbox']
                xmax = xmin + width
                ymax = ymin + height
                boxes.append([xmin, ymin, xmax, ymax])
                # Ensure your category_id in COCO is 1 for "license_plate"
                # PyTorch Faster R-CNN expects labels to start from 1 (0 is background)
                labels.append(ann['category_id']) # Assuming category_id=1 for license_plate

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([image_id]) # COCO eval might use this
        # Add area and iscrowd if your COCO annotations have them and evaluation needs them
        # target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)


        if self.transforms:
            # Convert PIL image to tensor first for torchvision.transforms.v2
            img_tensor = T.PILToTensor()(img)
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target

    def __len__(self):
        return len(self.imgs_info)

# --- Transforms ---
def get_transform(train):
    transforms = []
    # Converts PIL image to tensor and normalizes to [0, 1]
    transforms.append(T.ToDtype(torch.float, scale=True))
    if train:
        # Add data augmentation here if desired
        transforms.append(T.RandomHorizontalFlip(0.5))
        pass
    return T.Compose(transforms)

# --- Collate Function (handles None items from dataset) ---
def collate_fn(batch):
    # Filter out None items (from images that couldn't be loaded)
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch: # If all items in batch were None
        return None, None
    return tuple(zip(*batch))


# --- Model Definition ---
def get_model(num_classes):
    # Load a model pre-trained on COCO
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# --- Main Training Function ---
def main():
    # Paths to COCO annotation files
    train_annot_file = ANNOTATION_DIR_BASE / f"{DATASET_NAME_TRAIN}_{SPLIT_TRAIN}.json"
    valid_annot_file = ANNOTATION_DIR_BASE / f"{DATASET_NAME_VALID}_{SPLIT_VALID}.json"

    # Create datasets
    dataset_train = LicensePlateDataset(IMAGE_DIR_BASE, train_annot_file, DATASET_NAME_TRAIN, SPLIT_TRAIN, get_transform(train=True))
    dataset_valid = LicensePlateDataset(IMAGE_DIR_BASE, valid_annot_file, DATASET_NAME_VALID, SPLIT_VALID, get_transform(train=False))

    if not dataset_train or not len(dataset_train):
        print(f"Training dataset for {DATASET_NAME_TRAIN}/{SPLIT_TRAIN} is empty or failed to load. Exiting.")
        return
    if not dataset_valid or not len(dataset_valid):
        print(f"Validation dataset for {DATASET_NAME_VALID}/{SPLIT_VALID} is empty or failed to load. Exiting.")
        return

    # Create data loaders
    data_loader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=2, # os.cpu_count(),
        collate_fn=collate_fn
    )
    data_loader_valid = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE_VALID, shuffle=False, num_workers=2, # os.cpu_count(),
        collate_fn=collate_fn
    )

    # Get the model
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler (optional)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print("--- Starting Faster R-CNN Training ---")
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        model.train()
        train_loss_accum = 0.0
        train_loss_classifier_accum = 0.0
        train_loss_box_reg_accum = 0.0
        train_loss_objectness_accum = 0.0
        train_loss_rpn_box_reg_accum = 0.0

        for i, (images, targets) in enumerate(data_loader_train):
            if images is None or targets is None: # Batch was empty after filtering Nones
                print(f"Skipping empty batch {i+1}/{len(data_loader_train)}")
                continue

            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            try:
                loss_dict = model(images, targets) # This returns losses during training
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                train_loss_accum += losses.item()
                train_loss_classifier_accum += loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
                train_loss_box_reg_accum += loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
                train_loss_objectness_accum += loss_dict.get('loss_objectness', torch.tensor(0.0)).item()
                train_loss_rpn_box_reg_accum += loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item()

                if (i + 1) % 50 == 0: # Log every 50 batches
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{i+1}/{len(data_loader_train)}], Train Loss: {losses.item():.4f}")
            except Exception as e:
                print(f"Error during training batch {i+1}: {e}")
                print(f"Problematic images: {[info['file_name'] for idx, info in enumerate(dataset_train.imgs_info) if idx in [t['image_id'].item() for t in targets]]}") # This line might need adjustment based on how you access original image info from targets
                continue # Skip to next batch on error


        avg_train_loss = train_loss_accum / len(data_loader_train) if len(data_loader_train) > 0 else 0
        avg_train_loss_classifier = train_loss_classifier_accum / len(data_loader_train) if len(data_loader_train) > 0 else 0
        avg_train_loss_box_reg = train_loss_box_reg_accum / len(data_loader_train) if len(data_loader_train) > 0 else 0
        avg_train_loss_objectness = train_loss_objectness_accum / len(data_loader_train) if len(data_loader_train) > 0 else 0
        avg_train_loss_rpn_box_reg = train_loss_rpn_box_reg_accum / len(data_loader_train) if len(data_loader_train) > 0 else 0


        # Validation phase (simplified: only calculate loss)
        model.eval()
        val_loss_accum = 0.0
        val_loss_classifier_accum = 0.0
        val_loss_box_reg_accum = 0.0
        val_loss_objectness_accum = 0.0
        val_loss_rpn_box_reg_accum = 0.0

        with torch.no_grad():
            for images_val, targets_val in data_loader_valid:
                if images_val is None or targets_val is None: # Batch was empty
                    continue

                images_val = list(img.to(DEVICE) for img in images_val)
                targets_val = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets_val]

                # Important: For torchvision Faster R-CNN, to get losses in eval mode,
                # you might need to call model.train() temporarily or adapt the forward pass,
                # as model(images, targets) is the standard way for training.
                # A common practice for validation is to just get predictions and evaluate them separately
                # using a COCO evaluator, or wrap the loss calculation.
                # For simplicity, we'll try to get losses by ensuring model expects targets.
                # If this doesn't work as expected, one might need a custom evaluation function.
                model.train() # Temporarily set to train to get losses
                loss_dict_val = model(images_val, targets_val)
                model.eval() # Set back to eval

                losses_val = sum(loss for loss in loss_dict_val.values())
                val_loss_accum += losses_val.item()
                val_loss_classifier_accum += loss_dict_val.get('loss_classifier', torch.tensor(0.0)).item()
                val_loss_box_reg_accum += loss_dict_val.get('loss_box_reg', torch.tensor(0.0)).item()
                val_loss_objectness_accum += loss_dict_val.get('loss_objectness', torch.tensor(0.0)).item()
                val_loss_rpn_box_reg_accum += loss_dict_val.get('loss_rpn_box_reg', torch.tensor(0.0)).item()


        avg_val_loss = val_loss_accum / len(data_loader_valid) if len(data_loader_valid) > 0 else 0
        avg_val_loss_classifier = val_loss_classifier_accum / len(data_loader_valid) if len(data_loader_valid) > 0 else 0
        avg_val_loss_box_reg = val_loss_box_reg_accum / len(data_loader_valid) if len(data_loader_valid) > 0 else 0
        avg_val_loss_objectness = val_loss_objectness_accum / len(data_loader_valid) if len(data_loader_valid) > 0 else 0
        avg_val_loss_rpn_box_reg = val_loss_rpn_box_reg_accum / len(data_loader_valid) if len(data_loader_valid) > 0 else 0


        if lr_scheduler:
            lr_scheduler.step()

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Duration: {epoch_duration:.2f}s")
        print(f"  Avg Train Loss: {avg_train_loss:.4f} (Cls: {avg_train_loss_classifier:.4f}, BoxReg: {avg_train_loss_box_reg:.4f}, Obj: {avg_train_loss_objectness:.4f}, RPNBox: {avg_train_loss_rpn_box_reg:.4f})")
        print(f"  Avg Valid Loss: {avg_val_loss:.4f} (Cls: {avg_val_loss_classifier:.4f}, BoxReg: {avg_val_loss_box_reg:.4f}, Obj: {avg_val_loss_objectness:.4f}, RPNBox: {avg_val_loss_rpn_box_reg:.4f})")


        # Save checkpoint
        checkpoint_path = CHECKPOINT_SAVE_DIR / f"fasterrcnn_resnet50_fpn_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
            'train_loss': avg_train_loss,
            'valid_loss': avg_val_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    print("--- Faster R-CNN Training Finished ---")

if __name__ == "__main__":
    # Example: Set up argument parsing or just run main
    # For simplicity, running main directly. You might want to add argparse for CLI.
    main()