from datasets import load_dataset
import json
import os
import statistics
import urllib.request
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms
from tqdm import tqdm


class SIScoreDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Initializes SI-score dataset. Must be downloaded manually.

        Args:
            root (string): Root directory path.
            transform (callable, optional): Optional transform to be applied

        """
        self.root = root
        self.samples = []
        # Default transform: Resize to 224x224, ToTensor, Normalize (ImageNet)
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        self.transform = transform

        # Load official ImageNet class index (wnid -> 0-999)
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        class_idx = json.load(urllib.request.urlopen(url))
        self.wnid_to_idx = {v[1]: int(k) for k, v in class_idx.items()}

        # Only include classes actually present in the dataset folder
        root_path = Path(root)
        self.classes = [
            d.name for d in root_path.iterdir()
            if d.is_dir() and d.name in self.wnid_to_idx
        ]

        self.class_to_idx = {cls_name: self.wnid_to_idx[cls_name] for cls_name in self.classes}

        # Collect (path, class_idx) pairs
        for cls_name in self.classes:
            cls_dir = os.path.join(root, cls_name)
            for fname in os.listdir(cls_dir):
                path = os.path.join(cls_dir, fname)
                self.samples.append((path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def download_imagenet_subset(save_dir: str, min_images: int = 50, max_images: int = 100):
    """
    Downloads and filters the ImageNet-1k dataset based on class sample counts.
    """
    # Load ImageNet-1k in streaming mode
    from datasets import load_dataset
    ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, trust_remote_code=True)

    # Track counts
    counts = {i: 0 for i in range(1000)}  # one entry per class

    # Wrap dataset in tqdm
    pbar = tqdm(ds, desc="Filtering dataset", unit="samples")
    for i, example in enumerate(pbar):
        label = example["label"]

        # Skip if this class already has enough
        if counts[label] >= max_images:
            continue

        # Save image to its class folder
        class_dir = os.path.join(save_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)

        filename = os.path.join(class_dir, f"{counts[label]:04d}.jpg")

        img = example["image"].convert("RGB")
        img.save(filename, "JPEG")
        counts[label] += 1

        # Update tqdm description every 2000 samples
        nonzero = [c for c in counts.values() if c > 0]
        lowest = min(counts.values())
        avg = statistics.mean(nonzero) if nonzero else 0
        pbar.set_description(f"Lowest={lowest}, Avg={avg:.2f}")

        # Stop when all classes have at least min_images
        if all(c >= min_images for c in counts.values()):
            pbar.set_description(f"Done: Lowest={min_images}, Avg={statistics.mean(counts.values()):.2f}")
            break

    pbar.close()
    print("Done! Images saved in:", save_dir)


class ImageNetSubset(Dataset):
    """
    PyTorch Dataset for ImageNet subset stored in class-labeled folders.

    Directory structure:
        root/
            0/
                0000.jpg
                0001.jpg
                ...
            1/
                ...
            ...
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        """
        Initializes a small subset of the ImageNet dataset.

        Args:
            root (string): Root directory path.
            transform (callable, optional): Optional transform to be applied

        """
        self.root = Path(root)

        # Check if downloaded, if not, call the download function
        os.makedirs(str(self.root), exist_ok=True)
        if not os.listdir(str(self.root)):
            download_imagenet_subset(str(self.root))

        # Default transform: same ImageNet preprocessing (224x224 + Normalize)
        if transform is None:
            transform = models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        self.transform = transform

        # Gather all image paths and labels
        self.samples: List[Tuple[Path, int]] = []
        for class_dir in sorted(self.root.iterdir()):
            if class_dir.is_dir():
                label = int(class_dir.name)
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label
