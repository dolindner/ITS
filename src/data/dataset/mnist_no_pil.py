"""
MNiST Dataset skipping the pil conversion as i do not need it.
"""
import math

import torch
from typing import Tuple, Any, Callable, Optional

import torchvision
from src.utils.transforms.apply import grid_resample


class NoPILEMNIST(torchvision.datasets.EMNIST):
    def __init__(self, root: str, train: bool = True,split: str = 'balanced',transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        """
        NoPILEMNIST dataset that skips the PIL conversion and uses tensors directly.

        Args:
            root: Dataset root directory.
            train: Whether to load the training split.
            split: EMNIST split name.
            transform: Optional transform applied to the image.
            target_transform: Optional transform applied to the target.

        Returns:
            None
        """


        super(NoPILEMNIST, self).__init__(root, split=split, download=True,train=train,transform=transform,target_transform=target_transform)
        # Convert data to tensor
        self.data = self.data.float() / 255.0
        #flip horizental then rotate data counterclockwise by 90 degrees
        self.data = torch.flip(self.data, dims=[2])
        self.data = torch.rot90(self.data, k=1, dims=[1, 2])


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index].unsqueeze(0), self.targets[index]

        # overwriting get item so no PIL conversion is done

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def pretransform_data(self, transform, batch_size=1000):
        """
        Pre-applies transformations to the entire dataset and stores the results.

        Args:
            transform: The transform to apply to the data
            batch_size: Batch size for processing to reduce memory usage
        Returns:
            self: Returns self for chaining
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_samples = len(self.data)
        num_batches = (num_samples + batch_size - 1) // batch_size
        transformed_data = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch = self.data[start_idx:end_idx].unsqueeze(1).float().to(device)

            with torch.no_grad():
                transformed_batch = transform(batch)

            transformed_data.append(transformed_batch.cpu())

        self.data = torch.cat(transformed_data, dim=0)
        return self

class NoPILMNIST(torchvision.datasets.MNIST):

    def __init__(self, root: str, train: bool = True,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                download: bool = False) -> None:
            super(NoPILMNIST, self).__init__(root, train, transform, target_transform, download)
            # Convert data to tensor
            self.data = self.data.float() / 255.0


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index].unsqueeze(0), self.targets[index]

        # overwriting get item so no PIL conversion is done

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def pretransform_data(self, transform, batch_size=1000):
        """
        Pre-applies transformations to the entire dataset and stores the results.

        Args:
            transform: The transform to apply to the data
            batch_size: Batch size for processing to reduce memory usage

        Returns:
            self: Returns self for chaining
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_samples = len(self.data)
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

        transformed_data = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch = self.data[start_idx:end_idx].unsqueeze(1).float().to(device)

            with torch.no_grad():
                transformed_batch = transform(batch)
            transformed_data.append(transformed_batch.cpu())
        self.data = torch.cat(transformed_data, dim=0)
        return self





class NoPILFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, download: bool = False) -> None:
            super(NoPILFashionMNIST, self).__init__(root, train, transform, target_transform, download)
            # Convert data to tensor
            self.data = self.data.float() / 255.0

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index].unsqueeze(0), self.targets[index]

        # overwriting get item so no PIL conversion is done

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def pretransform_data(self, transform, batch_size=1000):
        """
        Pre-applies transformations to the entire dataset and stores the results.

        Args:
            transform: Transform to apply to the data.
            batch_size: Batch size for processing to reduce memory usage.

        Returns:
            self: Returns self for chaining.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_samples = len(self.data)
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

        transformed_data = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch = self.data[start_idx:end_idx].unsqueeze(1).float().to(device)

            with torch.no_grad():
                transformed_batch = transform(batch)
            transformed_data.append(transformed_batch.cpu())
        self.data = torch.cat(transformed_data, dim=0)
        return self


class AffineTransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_function=None, batch_size=None,resample_func=grid_resample,return_transformation=False
                 ,clip_data=True,seed=None,clip_min=None,
        clip_max=None,
                 ):
        self.dataset = dataset
        self.transformation_matrices = []
        self.return_transformation = return_transformation
        self.resample_func = resample_func
        self.seed = seed

        # Set random seed for reproducibility if provided
        if self.seed is not None:
            random_new_state = torch.get_rng_state()
            torch.manual_seed(self.seed)

        if batch_size is None:
            # Per sample transformation
            for i in range(len(self.dataset)):
                T = transform_function()
                self.transformation_matrices.append(T.cpu())
        else:
            # Batch transformation
            num_samples = len(self.dataset)
            num_batches = (num_samples + batch_size - 1) // batch_size
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                num_batches = end_idx - start_idx
                T = transform_function(num_batches)
                self.transformation_matrices.append(T.cpu())

        if self.seed is not None:
            #restore random state
            torch.set_rng_state(random_new_state)


        self.transformation_matrices = torch.cat(self.transformation_matrices, dim=0)
        self.clip_data = clip_data
        self.clip_min = clip_min if clip_min is not None else 0.0
        self.clip_max = clip_max if clip_max is not None else 1.0



    def __getitem__(self, index: int, return_transformation=False) -> Tuple[Any, Any]:
        T = self.transformation_matrices[index]
        img, target = self.dataset[index]
        img = self.resample_func(img.unsqueeze(0), T).squeeze(0)
        if self.clip_data:
            #check that it is image data with 3 image dimensions
            if len(img.shape) == 3:
                clip_min = self.clip_min
                clip_max = self.clip_max

                # If clip_min/max are tensors (per-channel), reshape for broadcasting
                if isinstance(clip_min, torch.Tensor):
                    clip_min = clip_min.view(-1, 1, 1)
                if isinstance(clip_max, torch.Tensor):
                    clip_max = clip_max.view(-1, 1, 1)

                img = torch.clamp(img, clip_min, clip_max)
        if self.return_transformation or return_transformation:
            return img, target, T
        return img, target


    def __len__(self) -> int:
        return len(self.dataset)

# Dynamic augmentation keeps a fresh transform for every __getitem__ call.
class DynamicAugmentDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, sampler_fn, resample_fn, datatype,clip=True):
            self.dataset = base_dataset
            self.sampler_fn = sampler_fn
            self.resample_fn = resample_fn
            self.is_image = datatype == "image"
            self.clip = clip

        def __getitem__(self, index):
            img, target = self.dataset[index]
            T = self.sampler_fn()
            img = self.resample_fn(img.unsqueeze(0), T).squeeze(0)

            # Only clamp image data (3D tensors: C x H x W)
            if self.is_image and self.clip and len(img.shape) == 3:
                img = torch.clamp(img, 0.0, 1.0)
            return img, target

        def __len__(self):
            return len(self.dataset)



class BatchTransformDataLoader:
    """
    Wrapper DataLoader that applies a batched transformation.
    Expects each batch from the inner DataLoader to be a tuple (images, targets),
    and applies the provided transform to the entire images batch.
    """
    def __init__(self, dataloader, transform: Callable):
        self.dataloader = dataloader
        self.transform = transform

    def __iter__(self):
        for imgs, targets in self.dataloader:
            yield self.transform(imgs), targets

    def __len__(self):
        return len(self.dataloader)


class BatchAffineTransformLoader:
    """
    Wrapper DataLoader for AffineTransformDataset that applies the stored transformation
    matrices in batch. Expects each batch from the inner DataLoader to be a tuple:
    ((images, targets), transformation_matrices)
    and applies the resample function to transform the entire batch of images.
    """
    def __init__(self, dataloader, resample_func: Callable):
        self.dataloader = dataloader
        self.resample_func = resample_func

    def __iter__(self):
        for (imgs, targets), Ts in self.dataloader:
            transformed_imgs = self.resample_func(imgs, Ts)
            yield transformed_imgs, targets

    def __len__(self):
        return len(self.dataloader)


