import os
import random

import numpy as np
from box import Box
from torch_geometric.transforms import SamplePoints

from embedding_cache import LayerEmbeddingCache
from src.data.dataset.mnist_no_pil import NoPILMNIST, AffineTransformDataset, NoPILEMNIST
from src.data.transformation import get_transformation_sequence_images
from src.utils.transform_sequence import create_sampler

try:
    from src.data.dataset.geometric_wrapper import GeometricsDatasetWrapper
    from torch_geometric.datasets import ModelNet
    from torch_geometric.transforms import NormalizeScale, SamplePoints
except ImportError:
    ModelNet = None  # optional dependency
from src.data.dataset.si_score import SIScoreDataset, ImageNetSubset
from torchvision import models
from src.data.dataset.tu_berlin_sketch import TUBerlinDataset
from src.utils.transforms.apply import transform_strokes_affine, grid_resample_border
import torch.nn as nn


def _build_loaders(train, val, test, batch_size, dataloader_kwargs):
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    train_loader_no_shuffle = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False,
                                                          **dataloader_kwargs)  # Non-shuffled train loader
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, **dataloader_kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, **dataloader_kwargs)
    shuffle_val = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    return train_loader, train_loader_no_shuffle, val_loader, test_loader, shuffle_val


class PicklableSampler:
    """Picklable wrapper for transform sequence sampling."""

    def __init__(self, transform_sequence):
        self.transform_sequence = transform_sequence

    def __call__(self, batch_size=1):
        params = self.transform_sequence.sample_individual(batch_size)
        # Use the TransformSequence callable to convert params -> matrices
        T = self.transform_sequence(params)
        return T.squeeze(0) if batch_size == 1 else T


import torch
from typing import Callable, Iterable, Tuple


class BatchAugmentCollate:
    """
    Collate function that applies a vectorized augmentation to a whole batch.
    - sampler(batch_size) -> transformation batch T (shape: B x ...)
    - transform_sequence.transform(images, T) -> transformed images
    """

    def __init__(self,
                 sampler: Callable[[int], torch.Tensor],
                 datatype: str = "image",
                 clip: bool = True):
        self.sampler = sampler
        self.is_image = datatype == "image"
        self.clip = clip

    def __call__(self, batch: Iterable[Tuple[torch.Tensor, object]]):
        # batch: list of (img, target) pairs
        imgs = torch.stack([b[0] for b in batch], dim=0)  # B x C x H x W
        targets = [b[1] for b in batch]

        B = imgs.shape[0]
        # Get transformation matrices from sampler
        # sampler already returns matrices (via transform_sequence.__call__)
        T = self.sampler(B)
        # Ensure same device as images
        T = T.to(imgs.device)

        with torch.no_grad():
            # Use application_method directly instead of transform
            # since we already have matrices
            transformed = self.sampler.transform_sequence.application_method(imgs, T)

        if self.is_image and self.clip:
            if transformed.dim() == 4 or transformed.dim() == 3:
                transformed = torch.clamp(transformed, 0.0, 1.0)

        # Convert targets to tensor if possible
        try:
            targets = torch.tensor(targets)
        except Exception:
            pass

        return transformed, targets


def _maybe_transform(datasets_tuple, batch_size, dataset_info, dataloader_kwargs, transform_seq, seed=None,
                     clip_images=True, clip_min=None, clip_max=None):
    sampler = create_sampler(transform_seq)
    resample_func = transform_seq.application_method
    train, val, test = datasets_tuple
    train_t = AffineTransformDataset(train, sampler, return_transformation=False, batch_size=batch_size,
                                     resample_func=resample_func, seed=seed, clip_data=clip_images, clip_min=clip_min,
                                     clip_max=clip_max)
    val_t = AffineTransformDataset(val, sampler, return_transformation=False, batch_size=batch_size,
                                   resample_func=resample_func, seed=seed + 1, clip_data=clip_images, clip_min=clip_min,
                                   clip_max=clip_max)
    test_t = AffineTransformDataset(test, sampler, return_transformation=False, batch_size=batch_size,
                                    resample_func=resample_func, seed=seed + 2, clip_data=clip_images,
                                    clip_min=clip_min, clip_max=clip_max)
    train_loader_t = torch.utils.data.DataLoader(train_t, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    train_loader_t_no_shuffle = torch.utils.data.DataLoader(train_t, batch_size=batch_size, shuffle=False,
                                                            **dataloader_kwargs)  # Non-shuffled transformed train loader
    val_loader_t = torch.utils.data.DataLoader(val_t, batch_size=batch_size, shuffle=False, **dataloader_kwargs)
    test_loader_t = torch.utils.data.DataLoader(test_t, batch_size=batch_size, shuffle=False, **dataloader_kwargs)
    shuffle_val_loader_t = torch.utils.data.DataLoader(val_t, batch_size=batch_size, shuffle=True, **dataloader_kwargs)

    # Dynamic augmentation using DynamicAugmentDataset from mnist_no_pil
    # Dynamic augmentation - use picklable sampler for multiprocessing
    datatype = getattr(dataset_info, "datatype", "image")
    picklable_sampler = PicklableSampler(transform_seq)
    resample_fn = getattr(transform_seq, "application_method", None)
    collate_aug = BatchAugmentCollate(picklable_sampler, datatype=datatype)

    train_loader_augmented = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_aug,
        **dataloader_kwargs
    )

    return {
        "train_loader_transformed": train_loader_t,
        "train_loader_transformed_no_shuffle": train_loader_t_no_shuffle,
        "val_loader_transformed": val_loader_t,
        "test_loader_transformed": test_loader_t,
        "shuffle_val_loader_transformed": shuffle_val_loader_t,
        "train_loader_augmented": train_loader_augmented
    }


import torchvision.transforms as transforms


def get_mnist_dataset(path, batch_size=128, dataloader_kwargs=None, dataset_info=None, seed=42):
    sampler = None
    pad = getattr(dataset_info, "pad", 0) if dataset_info else 0
    base_tf = transforms.Compose([transforms.Pad(pad)]) if pad > 0 else None
    dataset_train_full = NoPILMNIST(path, train=True, download=True, transform=base_tf)
    n_classes = 10
    dataset_test = NoPILMNIST(path, train=False, download=True, transform=base_tf)

    # Split validation from training set (10% for validation)
    n_total_train = len(dataset_train_full)
    n_val = int(0.1 * n_total_train)  # 10% for validation
    n_train = n_total_train - n_val
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset_train_full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    dataloader_kwargs = dataloader_kwargs or dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader, train_loader_no_shuffle, val_loader, test_loader, shuffle_val = _build_loaders(
        dataset_train, dataset_val, dataset_test, batch_size, dataloader_kwargs
    )

    if dataset_info.transform_seq_name not in [None, "none", ""]:
        transform_seq = get_transformation_sequence_images(name=dataset_info.transform_seq_name,
                                                           resample_method=dataset_info.resample_method)
        sampler = create_sampler(transform_seq)

    if sampler is not None:
        transformed = _maybe_transform((dataset_train, dataset_val, dataset_test), batch_size, dataset_info,
                                       dataloader_kwargs, transform_seq, seed=seed)
        return Box({
            "train_dataset": dataset_train,
            "val_dataset": dataset_val,
            "test_dataset": dataset_test,
            "train_loader": train_loader,
            "train_loader_no_shuffle": train_loader_no_shuffle,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "shuffle_val_loader": shuffle_val,
            "n_classes": n_classes,
            **transformed
        })
    else:
        return Box({
            "train_dataset": dataset_train,
            "val_dataset": dataset_val,
            "test_dataset": dataset_test,
            "train_loader": train_loader,
            "train_loader_no_shuffle": train_loader_no_shuffle,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "shuffle_val_loader": shuffle_val,
            "n_classes": n_classes,
        })


def get_emnist_dataset(path, batch_size=128, dataloader_kwargs=None, dataset_info=None, seed=42):
    sampler = None
    pad = getattr(dataset_info, "pad", 0) if dataset_info else 0
    base_tf = transforms.Compose([transforms.Pad(pad)]) if pad > 0 else None
    dataset_train_full = NoPILEMNIST(path, train=True, transform=base_tf)
    n_classes = len(dataset_train_full.classes)
    dataset_test = NoPILEMNIST(path, train=False, transform=base_tf)

    # Split validation from training set (not from test!)
    n_total_train = len(dataset_train_full)
    n_val = int(0.1 * n_total_train)  # 10% for validation
    n_train = n_total_train - n_val
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset_train_full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    dataloader_kwargs = dataloader_kwargs or dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader, train_loader_no_shuffle, val_loader, test_loader, shuffle_val = _build_loaders(
        dataset_train, dataset_val, dataset_test, batch_size, dataloader_kwargs
    )

    if dataset_info.transform_seq_name not in [None, "none", ""]:
        transform_seq = get_transformation_sequence_images(name=dataset_info.transform_seq_name,
                                                           resample_method=dataset_info.resample_method)
        sampler = create_sampler(transform_seq)

    transformed = _maybe_transform((dataset_train, dataset_val, dataset_test), batch_size, dataset_info,
                                   dataloader_kwargs, transform_seq, seed=seed) if sampler is not None else {}
    return Box({
        "train_dataset": dataset_train,
        "val_dataset": dataset_val,
        "test_dataset": dataset_test,
        "train_loader": train_loader,
        "train_loader_no_shuffle": train_loader_no_shuffle,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "shuffle_val_loader": shuffle_val,
        "n_classes": n_classes,
        **transformed
    })


def get_coil100_dataset(path, batch_size=64, dataloader_kwargs=None, dataset_info=None, seed=42):
    sampler = None
    dataloader_kwargs = dataloader_kwargs or dict(num_workers=4, pin_memory=True, persistent_workers=True)
    dataset_all = COIL100Dataset(os.path.join(path, "coil-100"), transform=transforms.ToTensor())
    n_total = len(dataset_all)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    dataset_train, dataset_val, dataset_test_pre = torch.utils.data.random_split(
        dataset_all, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )
    background_color = dataset_all[0][0][:, 0, 0].numpy()
    extra_affine = {}
    if dataset_info.resample_method == "adjusted_grid_resample":
        # pass background color through the resample func init inside transformation factory
        extra_affine["resample_func_kwargs"] = {"background_color": background_color}
    train_loader, train_loader_no_shuffle, val_loader, test_loader, shuffle_val = _build_loaders(dataset_train,
                                                                                                 dataset_val,
                                                                                                 dataset_test_pre,
                                                                                                 batch_size,
                                                                                                 dataloader_kwargs)
    transform_seq = get_transformation_sequence_images(name=dataset_info.transform_seq_name,
                                                       resample_method=dataset_info.resample_method)
    transformed = _maybe_transform((dataset_train, dataset_val, dataset_test_pre), batch_size, dataset_info,
                                   dataloader_kwargs, transform_seq, seed=seed) if transform_seq is not None else {}
    return Box({
        "train_dataset": dataset_train,
        "val_dataset": dataset_val,
        "test_dataset": dataset_test_pre,
        "train_loader": train_loader,
        "train_loader_no_shuffle": train_loader_no_shuffle,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "shuffle_val_loader": shuffle_val,
        "n_classes": 100,
        **transformed
    })


class TransformingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data.idx = idx
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.dataset)


# ----------------------------------------------------
# Deterministic variant of SamplePoints
# ----------------------------------------------------
class DeterministicSamplePoints(SamplePoints):
    def __init__(self, num, seed=42, **kwargs):
        super().__init__(num, **kwargs)
        self.seed = seed

    def __call__(self, data):
        # Create a per-sample deterministic generator
        idx_seed = self.seed
        if hasattr(data, "idx"):
            idx_seed += int(data.idx)
        g = torch.Generator().manual_seed(idx_seed)

        pos, face = data.pos, data.face
        assert pos is not None and face is not None
        assert pos.size(1) == 3 and face.size(0) == 3

        pos_max = pos.abs().max()
        pos = pos / pos_max

        # face areas
        area = (pos[face[1]] - pos[face[0]]).cross(
            pos[face[2]] - pos[face[0]], dim=1
        )
        area = area.norm(p=2, dim=1).abs() / 2

        prob = area / area.sum()
        sample = torch.multinomial(prob, self.num, replacement=True, generator=g)
        face = face[:, sample]

        frac = torch.rand(self.num, 2, generator=g, device=pos.device)
        mask = frac.sum(dim=-1) > 1
        frac[mask] = 1 - frac[mask]

        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]

        if self.include_normals:
            data.normal = torch.nn.functional.normalize(vec1.cross(vec2, dim=1), p=2)

        pos_sampled = pos[face[0]]
        pos_sampled += frac[:, :1] * vec1
        pos_sampled += frac[:, 1:] * vec2

        pos_sampled = pos_sampled * pos_max
        data.pos = pos_sampled

        if self.remove_faces:
            data.face = None

        return data


# ----------------------------------------------------
# ModelNet10 Loader
# ----------------------------------------------------
def get_modelnet10_dataset(path, batch_size=32, dataloader_kwargs=None, dataset_info=None, seed=42):
    if ModelNet is None:
        raise ImportError("torch_geometric is required for ModelNet dataset")

    dataloader_kwargs = dataloader_kwargs or dict(num_workers=4, pin_memory=True, persistent_workers=True)

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Remove NormalizeScale to avoid interfering with rotation; model will PCA-then-Norm
    pre_transform = None
    pc_transform = SamplePoints(1024)  # stochastic for training
    deterministic_pc_transform = DeterministicSamplePoints(1024, seed=seed)

    data_path = os.path.join(path, "modelnet")

    # Load raw ModelNet dataset without pre-scaling
    train_dataset_pre_val = ModelNet(data_path, '10', True, transform=None, pre_transform=pre_transform,
                                     force_reload=False)
    test_raw = ModelNet(data_path, '10', False, transform=None, pre_transform=pre_transform, force_reload=False)

    # Split train/val deterministically
    n_train = int(0.9 * len(train_dataset_pre_val))
    n_val = len(train_dataset_pre_val) - n_train
    g = torch.Generator().manual_seed(seed)
    train_raw, val_raw = torch.utils.data.random_split(train_dataset_pre_val, [n_train, n_val], generator=g)

    # Wrap datasets using GeometricsDatasetWrapper + TransformingDataset
    dataset_train = GeometricsDatasetWrapper(TransformingDataset(train_raw, transform=pc_transform))
    dataset_train_no_aug = GeometricsDatasetWrapper(
        TransformingDataset(train_raw, transform=deterministic_pc_transform))
    dataset_val = GeometricsDatasetWrapper(TransformingDataset(val_raw, transform=deterministic_pc_transform))
    dataset_test_pre = GeometricsDatasetWrapper(TransformingDataset(test_raw, transform=deterministic_pc_transform))

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    train_loader_no_shuffle = torch.utils.data.DataLoader(dataset_train_no_aug, batch_size=batch_size, shuffle=False,
                                                          **dataloader_kwargs)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, **dataloader_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test_pre, batch_size=batch_size, shuffle=False,
                                              **dataloader_kwargs)
    shuffle_val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True,
                                                     **dataloader_kwargs)

    transform_seq = get_transformation_sequence_images(
        name=dataset_info.transform_seq_name,
        resample_method=dataset_info.resample_method
    )

    transformed = _maybe_transform(
        (dataset_train, dataset_val, dataset_test_pre),
        batch_size,
        dataset_info,
        dataloader_kwargs,
        transform_seq,
        seed=seed,

    )

    return Box({
        "train_dataset": dataset_train,
        "val_dataset": dataset_val,
        "test_dataset": dataset_test_pre,
        "train_loader": train_loader,
        "train_loader_no_shuffle": train_loader_no_shuffle,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "shuffle_val_loader": shuffle_val_loader,
        "n_classes": 10,
        **transformed
    })


# ----------------------------------------------------
# ModelNet40 Loader
# ----------------------------------------------------
def get_modelnet40_dataset(path, batch_size=32, dataloader_kwargs=None, dataset_info=None, seed=42):
    if ModelNet is None:
        raise ImportError("torch_geometric is required for ModelNet dataset")

    dataloader_kwargs = dataloader_kwargs or dict(num_workers=4, pin_memory=True, persistent_workers=True)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Remove NormalizeScale to avoid interfering with rotation; model will PCA-then-Norm
    pre_transform = None
    pc_transform = SamplePoints(1024)  # stochastic for training
    deterministic_pc_transform = DeterministicSamplePoints(1024, seed=seed)

    data_path = os.path.join(path, "modelnet40")

    train_dataset_pre_val = ModelNet(data_path, '40', True, transform=None, pre_transform=pre_transform,
                                     force_reload=False)
    test_raw = ModelNet(data_path, '40', False, transform=None, pre_transform=pre_transform, force_reload=False)

    n_train = int(0.9 * len(train_dataset_pre_val))
    n_val = len(train_dataset_pre_val) - n_train
    g = torch.Generator().manual_seed(seed)
    train_raw, val_raw = torch.utils.data.random_split(train_dataset_pre_val, [n_train, n_val], generator=g)

    # Wrap datasets
    dataset_train = GeometricsDatasetWrapper(TransformingDataset(train_raw, transform=pc_transform))
    dataset_train_no_aug = GeometricsDatasetWrapper(
        TransformingDataset(train_raw, transform=deterministic_pc_transform))
    dataset_val = GeometricsDatasetWrapper(TransformingDataset(val_raw, transform=deterministic_pc_transform))
    dataset_test_pre = GeometricsDatasetWrapper(TransformingDataset(test_raw, transform=deterministic_pc_transform))

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    train_loader_no_shuffle = torch.utils.data.DataLoader(dataset_train_no_aug, batch_size=batch_size, shuffle=False,
                                                          **dataloader_kwargs)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, **dataloader_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test_pre, batch_size=batch_size, shuffle=False,
                                              **dataloader_kwargs)
    shuffle_val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True,
                                                     **dataloader_kwargs)

    transform_seq = get_transformation_sequence_images(
        name=dataset_info.transform_seq_name,
        resample_method=dataset_info.resample_method
    )

    transformed = _maybe_transform(
        (dataset_train, dataset_val, dataset_test_pre),
        batch_size,
        dataset_info,
        dataloader_kwargs,
        transform_seq,
        seed=seed
    )

    return Box({
        "train_dataset": dataset_train,
        "val_dataset": dataset_val,
        "test_dataset": dataset_test_pre,
        "train_loader": train_loader,
        "train_loader_no_shuffle": train_loader_no_shuffle,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "shuffle_val_loader": shuffle_val_loader,
        "n_classes": 40,
        **transformed
    })


class ResizeNormalizeOnly(nn.Module):
    """Custom transform that only resizes and normalizes, skipping center crop."""

    def __init__(
            self,
            resize_size: int,
            mean: tuple[float, ...] = (0.485, 0.456, 0.406),
            std: tuple[float, ...] = (0.229, 0.224, 0.225),
            interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BILINEAR,
            antialias: bool = True,
    ):
        super().__init__()
        self.resize_size = resize_size
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        from torchvision.transforms import functional as F
        # Resize
        img = F.resize(img, [self.resize_size], interpolation=self.interpolation, antialias=self.antialias)
        # Convert to tensor if needed
        if not isinstance(img, torch.Tensor):
            img = F.pil_to_tensor(img)
        # Convert dtype
        img = F.convert_image_dtype(img, torch.float)
        # Normalize
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img


def get_siscore_dataset(path, batch_size=128, dataloader_kwargs=None, dataset_info=None, seed=42,
                        preprocessing_weights="vit"):
    """
    Load ImageNetSubset for train/val/test (90/5/5 split).
    For transformed datasets: test_transformed uses SIScoreDataset, train_transformed and val_transformed are empty.
    preprocessing_weights: "vit" (ViT_B_16_Weights.IMAGENET1K_V1) or "resnet" (ResNet50_Weights.IMAGENET1K_V2)
    """
    dataloader_kwargs = dataloader_kwargs or dict(num_workers=4, pin_memory=True, persistent_workers=True)

    # Select preprocessing transform based on argument
    if preprocessing_weights == "vit":
        preprocess_transform = models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        proprocess_si_score = preprocess_transform
    elif preprocessing_weights == "resnet":
        preprocess_transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()
        proprocess_si_score = preprocess_transform
    elif preprocessing_weights == "vit_no_crop":
        preprocess_transform = models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        proprocess_si_score = ResizeNormalizeOnly(224)
    elif preprocessing_weights == "resnet_no_crop":
        preprocess_transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()
        proprocess_si_score = ResizeNormalizeOnly(224)
    else:
        raise ValueError(f"Unknown preprocessing_weights: {preprocessing_weights}")

    # Load ImageNetSubset for regular train/val/test
    imagenet_subset_path = os.path.join(path, "si_score", "imagenet_subset")

    imagenet_dataset = ImageNetSubset(root=imagenet_subset_path, transform=preprocess_transform)
    if not os.path.isdir(imagenet_subset_path):
        raise RuntimeError(f"ImageNet subset directory not found at {imagenet_subset_path}")

    if len(imagenet_dataset) == 0:
        raise RuntimeError(f"No samples found in ImageNet subset at {imagenet_subset_path}")

    # Split 90/5/5 for train/val/test
    n_total = len(imagenet_dataset)
    n_train = int(0.9 * n_total)
    n_val = int(0.05 * n_total)
    n_test = n_total - n_train - n_val

    # Create the splits with a fixed random seed
    train_set, val_set, test_set = torch.utils.data.random_split(
        imagenet_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Create the regular loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    train_loader_no_shuffle = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                                          **dataloader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, **dataloader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **dataloader_kwargs)
    shuffle_val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, **dataloader_kwargs)

    # For the transformed test_loader, use SIScoreDataset
    siscore_path = os.path.join(path, "si_score", "rotation")
    if not os.path.isdir(siscore_path):
        raise RuntimeError(f"SI-Score rotation directory not found at {siscore_path}")

    siscore_dataset = SIScoreDataset(root=siscore_path, transform=proprocess_si_score)

    if len(siscore_dataset) == 0:
        raise RuntimeError(f"No samples found in SI-Score dataset at {siscore_path}")

    # Create rotation-only transformation sequence
    transform_seq = get_transformation_sequence_images(
        name=dataset_info.transform_seq_name,  # "si_score_default" -> rotation only
        resample_method=dataset_info.resample_method
    )

    clip_min = torch.tensor([-2.1179039301310043, -2.0357142857142856, -1.8044444444444445])
    clip_max = torch.tensor([2.2489082969432315, 2.428571428571429, 2.6399999999999997])

    # Apply transformations to train/val/test splits
    transformed = _maybe_transform(
        (train_set, val_set, test_set),
        batch_size,
        dataset_info,
        dataloader_kwargs,
        transform_seq,
        seed=seed,
        clip_images=True, clip_min=clip_min, clip_max=clip_max
    )

    test_loader_transformed = torch.utils.data.DataLoader(siscore_dataset, batch_size=batch_size, shuffle=False,
                                                          **dataloader_kwargs)

    # Determine number of classes from ImageNet subset
    n_classes = len({label for _, label in imagenet_dataset.samples})

    return Box({
        "train_dataset": train_set,
        "val_dataset": val_set,
        "test_dataset": test_set,
        "train_loader": train_loader,
        "train_loader_no_shuffle": train_loader_no_shuffle,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "shuffle_val_loader": shuffle_val_loader,
        "n_classes": n_classes,
        # Transformed loaders
        "train_loader_transformed": transformed["train_loader_transformed"],
        "train_loader_transformed_no_shuffle": transformed["train_loader_transformed_no_shuffle"],
        "val_loader_transformed": transformed["val_loader_transformed"],
        "test_loader_transformed": test_loader_transformed,
        "shuffle_val_loader_transformed": transformed["shuffle_val_loader_transformed"],
    })


def get_siscore_dataset_resnet(path, batch_size=128, dataloader_kwargs=None, dataset_info=None, seed=42):
    """
    SI-Score dataset with ResNet50 preprocessing (IMAGENET1K_V2).
    """
    return get_siscore_dataset(
        path,
        batch_size=batch_size,
        dataloader_kwargs=dataloader_kwargs,
        dataset_info=dataset_info,
        seed=seed,
        preprocessing_weights="resnet"
    )


def get_tu_berlin_dataset(path, batch_size=128, dataloader_kwargs=None, dataset_info=None, seed=42):
    max_len = getattr(dataset_info, "max_len", 200)
    interpolation_points = getattr(dataset_info, "interpolation_points", 2)
    dataset = TUBerlinDataset(path, max_len=max_len, interpolation_points=interpolation_points)
    n_classes = len(dataset.class_names)
    dataloader_kwargs = dataloader_kwargs or dict(num_workers=4, pin_memory=True, persistent_workers=True)
    # Split train/val/test (80/10/10)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    train_loader_no_shuffle = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                                          **dataloader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, **dataloader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, **dataloader_kwargs)
    shuffle_val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    # Transformation sequence for strokes
    transform_seq = get_transformation_sequence_images(name=dataset_info.transform_seq_name,
                                                       resample_method=dataset_info.resample_method)
    sampler = create_sampler(transform_seq)
    transformed = _maybe_transform((train_set, val_set, test_set), batch_size, dataset_info, dataloader_kwargs,
                                   transform_seq, seed=seed)
    return Box({
        "train_dataset": train_set,
        "val_dataset": val_set,
        "test_dataset": test_set,
        "train_loader": train_loader,
        "train_loader_no_shuffle": train_loader_no_shuffle,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "shuffle_val_loader": shuffle_val_loader,
        "n_classes": n_classes,
        "class_names": dataset.class_names,
        **transformed
    })


def get_dataset_info(name):
    """Build the dataset metadata used by the loaders.

    Args:
        name: Dataset identifier.

    Returns:
        Box: Dataset configuration for the requested name.
    """
    mnist_info = {
        'input_size': (1, 28, 28),
        'num_classes': 10,
        'transform_seq_name': 'mnist_default',
        'datatype': 'image',
        'name': name,
        'resample_method': 'grid_resample',
        'pad': 0,
        'epochs': 100,
        'batch_size': 128,
        'batch_size_search': 1024
    }
    mnist_info = Box(mnist_info)
    if name.lower() == "mnist":
        return mnist_info
    if name.lower() in ["rotatedmnist", "rotated_mnist"]:
        return Box({
            'input_size': (1, 28, 28),
            'num_classes': 10,
            'transform_seq_name': 'rotated_mnist_rotation_only',
            'datatype': 'image',
            'name': name,
            'resample_method': 'grid_resample',
            'pad': 0,
            'epochs': 100,
            'batch_size': 128,
            'batch_size_search': 1024
        })
    if name.lower() in ["biggermnist", "bigger_mnist"]:
        pad = 6
        size = 28 + 2 * pad
        return Box({
            'input_size': (1, size, size),
            'num_classes': 10,
            'transform_seq_name': 'biggermnist_default',  # changed from mnist_default
            'datatype': 'image',
            'name': name,
            'resample_method': 'grid_resample',
            'pad': pad,
            'epochs': 100,
            'batch_size': 128,
            'batch_size_search': 1024
        })
    if name.lower() == "emnist":
        return Box({
            'input_size': (1, 28, 28),
            'num_classes': 47,
            'transform_seq_name': 'emnist_default',
            'datatype': 'image',
            'name': name,
            'resample_method': 'grid_resample',
            'pad': 0,
            'epochs': 100,
            'batch_size': 128,
            'batch_size_search': 1024
        })
    if name.lower() in ["biggerextendedmnist", "bigger_emnist", "biggeremnist"]:
        pad = 6
        size = 28 + 2 * pad
        return Box({
            'input_size': (1, size, size),
            'num_classes': 47,
            'transform_seq_name': 'bigger_emnist_default',  # changed from emnist_default
            'datatype': 'image',
            'name': name,
            'resample_method': 'grid_resample',
            'pad': pad,
            'epochs': 100,
            'batch_size': 128,
            'batch_size_search': 1024
        })
    if name.lower() == "coil100":
        return Box({
            'input_size': (3, 128, 128),
            'num_classes': 100,
            'transform_seq_name': 'coil_default',
            'datatype': 'image',
            'name': name,
            'resample_method': grid_resample_border,
            'pad': 0,
            'epochs': 100,
            'batch_size': 32,
            'batch_size_search': 128
        })
    if name.lower() in ["modelnet10"]:
        return Box({
            'input_size': (1024, 3),
            'num_classes': 10,
            'transform_seq_name': 'modelnet_default',
            'datatype': 'pointcloud',
            'name': name,
            'resample_method': 'transform_3d_point_cloud',
            'pad': 0,
            'epochs': 200,
            'batch_size': 16,
            'batch_size_search': 16
        })
    if name.lower() in ["modelnet"]:
        return Box({
            'input_size': (1024, 3),
            'num_classes': 40,
            'transform_seq_name': 'modelnet_default',
            'datatype': 'pointcloud',
            'name': name,
            'resample_method': 'transform_3d_point_cloud',
            'pad': 0,
            'epochs': 200,
            'batch_size': 16,
            'batch_size_search': 8
        })
    if name.lower() == "si_score":
        return Box({
            'input_size': (3, 224, 224),
            'num_classes': 1000,
            'transform_seq_name': 'si_score_default',  # rotation only
            'datatype': 'image',
            'name': name,
            'resample_method': 'grid_resample',
            'pad': 0,
            'epochs': 100,
            'batch_size': 32,
            'batch_size_search': 32
        })
    if name.lower() == "si_score_resnet":
        return Box({
            'input_size': (3, 224, 224),
            'num_classes': 1000,
            'transform_seq_name': 'si_score_default',
            'datatype': 'image',
            'name': name,
            'resample_method': 'grid_resample',
            'pad': 0,
            'epochs': 100,
            'batch_size': 32,  # ResNet special case
            'batch_size_search': 32
        })
    if name.lower() == "si_score_resnet_no_crop":
        return Box({
            'input_size': (3, 224, 224),
            'num_classes': 1000,
            'transform_seq_name': 'si_score_default',  # rotation only
            'datatype': 'image',
            'name': name,
            'resample_method': 'grid_resample',
            'pad': 0,
            'epochs': 100,
            'batch_size': 32,
            'batch_size_search': 32
        })
    if name.lower() == "si_score_vit_no_crop":
        return Box({
            'input_size': (3, 224, 224),
            'num_classes': 1000,
            'transform_seq_name': 'si_score_default',  # rotation only
            'datatype': 'image',
            'name': name,
            'resample_method': 'grid_resample',
            'pad': 0,
            'epochs': 100,
            'batch_size': 32,  # ResNet special case
            'batch_size_search': 32
        })
    if name.lower() == "tu_berlin":
        return Box({
            'input_size': (200, 3),
            'num_classes': 250,
            'transform_seq_name': 'biggermnist_default',  # reuse bigger_mnist settings
            'datatype': 'stroke',
            'name': name,
            'resample_method': transform_strokes_affine,
            'max_len': 200,
            'interpolation_points': 2,
            'epochs': 50,
            'batch_size': 128,
            'batch_size_search': 128
        })
    raise ValueError(f"Dataset {name} not recognized")


# Helper to get dataaset
def get_dataset(info, path, batch_size=128, dataloader_kwargs=None, seed=42):
    """Load a dataset split bundle for the requested configuration.

    Args:
        info: Dataset metadata from `get_dataset_info`.
        path: Base data directory.
        batch_size: Batch size used for loaders.
        dataloader_kwargs: Extra keyword arguments for `DataLoader`.
        seed: Random seed used for deterministic splits.

    Returns:
        Box: Datasets and loaders for the requested dataset.
    """
    lname = info.name.lower()
    if lname in ["mnist", "rotatedmnist", "rotated_mnist", "biggermnist", "bigger_mnist"]:
        return get_mnist_dataset(path, batch_size=batch_size, dataloader_kwargs=dataloader_kwargs, dataset_info=info,
                                 seed=seed)
    if lname in ["emnist", "biggerextendedmnist", "bigger_emnist", "biggeremnist"]:
        return get_emnist_dataset(path, batch_size=batch_size, dataloader_kwargs=dataloader_kwargs, dataset_info=info,
                                  seed=seed)
    if lname == "coil100":
        return get_coil100_dataset(path, batch_size=batch_size, dataloader_kwargs=dataloader_kwargs, dataset_info=info,
                                   seed=seed)
    if lname in ["modelnet10"]:
        return get_modelnet10_dataset(path, batch_size=batch_size, dataloader_kwargs=dataloader_kwargs,
                                      dataset_info=info, seed=seed)
    if lname in ["modelnet"]:
        return get_modelnet40_dataset(path, batch_size=batch_size, dataloader_kwargs=dataloader_kwargs,
                                      dataset_info=info, seed=seed)
    if lname == "si_score":
        # Default: ViT preprocessing
        return get_siscore_dataset(path, batch_size=batch_size, dataloader_kwargs=dataloader_kwargs, dataset_info=info)
    if lname == "si_score_resnet":
        return get_siscore_dataset_resnet(path, batch_size=batch_size, dataloader_kwargs=dataloader_kwargs,
                                          dataset_info=info)
    if lname == "si_score_resnet_no_crop":
        return get_siscore_dataset(
            path,
            batch_size=batch_size,
            dataloader_kwargs=dataloader_kwargs,
            dataset_info=info,
            preprocessing_weights="resnet_no_crop"
        )
    if lname == "si_score_vit_no_crop":
        return get_siscore_dataset(
            path,
            batch_size=batch_size,
            dataloader_kwargs=dataloader_kwargs,
            dataset_info=info,
            preprocessing_weights="vit_no_crop"
        )
    if lname == "tu_berlin":
        return get_tu_berlin_dataset(path, batch_size=batch_size, dataloader_kwargs=dataloader_kwargs,
                                     dataset_info=info, seed=seed)
    raise ValueError(f"Dataset {info.name} not recognized")


# helper to get config of embedding cache which defines reduction methods.
def get_layer_embedding_cache_config(dataset: str,
                                     architecture: str,
                                     transform_name: str = None,
                                     dataset_info: Box = None,
                                     dim_override: int = None):
    """
    Return a config Box for building a LayerEmbeddingCache for the given dataset/architecture.
    """
    name = (dataset or "dataset").lower()
    arch = (architecture or "arch").lower()
    tf = transform_name or ""

    # sensible defaults per dataset
    if dim_override is not None:
        dim = int(dim_override)
    else:
        if name in ["emnist", "bigger_emnist", "biggeremnist"]:
            dim = 512
        elif name in ["mnist", "rotatedmnist", "rotated_mnist", "biggermnist", "bigger_mnist",
                      "biggerextendedmnist"]:
            dim = 1024
        elif name in ["coil100", "coil"]:
            dim = 2048
        elif name in ["modelnet", "modelnet10", "modelnet40"]:
            dim = 2048
        elif name in ["si_score", "siscore", "si-score", "si_score_resnet",
                      "si_score_resnet_no_crop", "si_score_vit_no_crop"]:
            dim = 4096
        elif name in ["tu_berlin", "tu_berlin"]:
            dim = 1024
        else:
            dim = 2048  # safe default
            raise Exception(
                f"Dataset {name} not recognized for default embedding cache dim; please provide dim_override")

    # Build cache name containing the dimension
    safe_arch = architecture.replace(" ", "_") if architecture else "arch"
    cache_name_train = f"{dataset}_{safe_arch}_{tf}_{dim}_embedding_cache_train"

    dtype = getattr(dataset_info, "datatype", None) if dataset_info is not None else None
    input_size = getattr(dataset_info, "input_size", None) if dataset_info is not None else None

    # Pointclouds / ModelNet -> only RP
    if dtype == "pointcloud" or name.startswith("modelnet"):
        reducer_names = ["rp"]
        reducer_kwargs = [{"n_components": dim, "method": "gaussian"}]
        reducer_dim_threshold = [(dim + 1, None)]

    elif dtype == "stroke" or name in ["tu_berlin", ]:
        # Strokes / TU Berlin
        reducer_names = ["rp", ]
        reducer_kwargs = [
            {"n_components": dim, "method": "gaussian"},
        ]
        reducer_dim_threshold = [
            (dim + 1, 20000),
        ]
    else:
        # Images / strokes
        if "vit" in arch:
            reducer_names = ["token_pool", ]
            reducer_kwargs = [
                {"method": "cls"},  # token_pool params: return CLS + mean of patches (adjustable)
            ]
            reducer_dim_threshold = [
                (dim + 1, None),  # token-based extractor takes priority
            ]
        else:
            reducer_names = ["image_transform", "collapse_image", "rp", ]
            reducer_kwargs = [
                {"reduce_dims": (3, 3), "rp_dim": dim, "average_pool": True},
                {},  # collapse_image: no params (global avg over H,W)
                {"n_components": dim, "method": "gaussian"},
            ]

            reducer_dim_threshold = [
                (dim + 1, None),  # image_transform for small->medium features
                (dim + 1, None),  # collapse_image (no params) - available as an alternative
                (dim + 1, 20000),  # rp fallback
            ]

    reducer_fit_batches = 0

    return Box({
        "cache_name_train": cache_name_train,
        "dim": dim,
        "reducer_name": reducer_names,
        "reducer_kwargs": reducer_kwargs,
        "reducer_dim_threshold": reducer_dim_threshold,
        "reducer_fit_batches": reducer_fit_batches,
    })


# method to create the cache.
def create_layer_embedding_cache(model,
                                 train_loader_no_shuffle,
                                 cache_config,
                                 embedding_cache_path,
                                 device=None,
                                 cache_dir_override=None):
    """
    Create and return a LayerEmbeddingCache instance from a cache_config Box/dict.

    Parameters:
      - model: the model instance whose layer embeddings will be cached.
      - train_loader_no_shuffle: the DataLoader used to compute/store embeddings.
      - cache_config: Box or dict returned by get_layer_embedding_cache_config (keys like
                      'cache_name_train','reducer_name','reducer_kwargs','reducer_dim_threshold','reducer_fit_batches').
      - embedding_cache_path: base directory where cache directories are created.
      - device: optional device to pass to the cache constructor (if supported).
      - cache_dir_override: optional explicit cache directory path; if provided this is used
                            instead of embedding_cache_path + cache_name_train.

    Returns:
      - instance of LayerEmbeddingCache

    """
    # Accept Box or dict-like
    cfg = cache_config or {}
    # Determine cache name / dir
    cache_name = None
    if isinstance(cfg, dict):
        cache_name = cfg.get("cache_name_train") or cfg.get("cache_name") or None
    else:
        # Box supports attribute access
        cache_name = getattr(cfg, "cache_name_train", None) or getattr(cfg, "cache_name", None)

    if cache_dir_override:
        final_cache_dir = cache_dir_override
    else:
        if cache_name is None:
            raise ValueError("cache_config must contain 'cache_name_train' or provide cache_dir_override")
        final_cache_dir = os.path.join(embedding_cache_path, cache_name)

    # Collect possible constructor kwargs
    ctor_kwargs = {"cache_dir": final_cache_dir}
    # optional keys we expect
    if isinstance(cfg, dict):
        if "reducer_name" in cfg:
            ctor_kwargs["reducer_name"] = cfg["reducer_name"]
        if "reducer_kwargs" in cfg:
            ctor_kwargs["reducer_kwargs"] = cfg["reducer_kwargs"]
        if "reducer_dim_threshold" in cfg:
            ctor_kwargs["reducer_dim_threshold"] = cfg["reducer_dim_threshold"]
        if "reducer_fit_batches" in cfg:
            ctor_kwargs["reducer_fit_batches"] = cfg["reducer_fit_batches"]
    else:
        # Box/attr access
        if getattr(cfg, "reducer_name", None) is not None:
            ctor_kwargs["reducer_name"] = cfg.reducer_name
        if getattr(cfg, "reducer_kwargs", None) is not None:
            ctor_kwargs["reducer_kwargs"] = cfg.reducer_kwargs
        if getattr(cfg, "reducer_dim_threshold", None) is not None:
            ctor_kwargs["reducer_dim_threshold"] = cfg.reducer_dim_threshold
        if getattr(cfg, "reducer_fit_batches", None) is not None:
            ctor_kwargs["reducer_fit_batches"] = cfg.reducer_fit_batches

    # Pass device if requested (ignore if LayerEmbeddingCache doesn't accept it)
    if device is not None:
        ctor_kwargs["device"] = device

    cache = LayerEmbeddingCache(model, train_loader_no_shuffle, **ctor_kwargs)
    return cache
