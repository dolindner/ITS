from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Any

import numpy as np
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.utils.data import DataLoader, TensorDataset
from confidence.base_confidence import ConfidenceModule
from confidence.input_transform import InputTransform

class ClassicConfidenceBase(ConfidenceModule, ABC):
    """
    Base for classic methods operating on embeddings.
    fit(…) accepts either
      • Tensor[(N, D)] (and optional y: Tensor[(N,)]), or
      • DataLoader yielding batches of x or (x, y).
    Handles a possible input_transform that is applied to the features  before delegating to _fit.
    """

    def __init__(self, input_transform: Optional[InputTransform] = None):
        super().__init__()
        self.input_transform = input_transform

    def fit(
        self,
        data: Union[torch.Tensor, DataLoader],
        y: Optional[torch.Tensor] = None
    ) -> "ClassicConfidenceBase":
        # Gather all x and optional y into full tensors
        Y = None
        if isinstance(data, torch.Tensor) or isinstance(data, list):
            X, Y = data, y
        elif isinstance(data, np.ndarray):
            X = torch.from_numpy(data)
            if y is not None:
                Y = torch.from_numpy(y)
        elif isinstance(data, DataLoader):
            xs, ys = [], []
            for batch in data:
                if isinstance(batch, (list, tuple)):
                    x_batch, y_batch = batch[0], batch[1] if len(batch) > 1 else None
                else:
                    x_batch, y_batch = batch, None
                xs.append(x_batch)
                if y_batch is not None:
                    ys.append(y_batch)
            X = torch.cat(xs, dim=0)
            Y = torch.cat(ys, dim=0) if ys else None
        else:
            raise TypeError(f"Expected Tensor or DataLoader, got {type(data)}")

        # Apply transform on fitting data
        if self.input_transform:
            self.input_transform.fit(X, Y)  # may return X or (X, Y)
            X = self.input_transform.transform(X)

        # Delegate to subclass
        return self._fit(X, Y)

    @abstractmethod
    def _fit(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> "ClassicConfidenceBase":
        """
        Implement fitting logic on full tensors.
        """
        ...

    def forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        # Apply transform before any subclass logic
        if self.input_transform:
            x = self.input_transform.transform(x)
        return self._forward(x, y=y)

    @abstractmethod
    def _forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        """
        Subclasses implement this to compute confidence on the (transformed) x.
        """
        ...

    def to(self: "ClassicConfidenceBase", *args, **kwargs) -> "ClassicConfidenceBase":
        return super().to(*args, **kwargs)

    def cuda(self: "ClassicConfidenceBase", device: Optional[Union[int, torch.device]] = None) -> "ClassicConfidenceBase":
        return super().cuda(device)

    def cpu(self: "ClassicConfidenceBase") -> "ClassicConfidenceBase":
        return super().cpu()

    def save(self,file_path: str) -> None:
        """
        Save the model state to a file.
        """
        torch.save(self.state_dict(), file_path)

    def load(self, file_path: str) -> None:
        """
        Load the model state from a file.
        """
        self.load_state_dict(torch.load(file_path))

#TODO can this be removed am i using this even?
import torch
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from confidence.base_confidence import ConfidenceModule
from confidence.input_transform import InputTransform
from typing import Optional, Union, Dict, Any

class MLConfidenceBase(pl.LightningModule, ConfidenceModule, ABC):
    """
    Base for ML-based confidence modules (autoencoders, discriminators, etc.).
    Stores dataloader_kwargs and trainer_kwargs separately for pl.Trainer,
    handles input_transform, and registers performance-relevant settings
    (lr, max_epochs, batch_size) as hyperparameters.
    """

    def __init__(
        self,
        input_transform: Optional[InputTransform] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
            optimizer_type: Optional[torch.optim.Optimizer] = None,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            negative_sampling_module: Optional[Any] = None
    ):
        super().__init__()
        self.input_transform = input_transform
        self.trainer_kwargs = trainer_kwargs or {"max_epochs":10}
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.optimizer_type = optimizer_type or torch.optim.Adam
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-3}
        self.negative_sampling_module= negative_sampling_module
        self.feature_extractor = None
        if negative_sampling_module is not None:
            #check if attribute exists
            if hasattr(negative_sampling_module.strategy, 'feature_extractor'):
                self.feature_extractor = self.negative_sampling_module.strategy.feature_extractor if negative_sampling_module else None

        self.deterministic_val = False




    def fit(
        self,
        data: Union[torch.Tensor, DataLoader],
        y: Optional[torch.Tensor] = None,
            val_data: Optional[Union[torch.Tensor, DataLoader]] = None,
            val_y: Optional[torch.Tensor] = None
    ) -> "MLConfidenceBase":
        # Build DataLoader if raw Tensor provided
        if isinstance(data, torch.Tensor):
            X = data.to(self.device)
            if y is not None:
                Y = y.to(self.device)
                dataset = TensorDataset(X, Y)
            else:
                dataset = TensorDataset(X)
            loader = DataLoader(
                dataset,
                **self.dataloader_kwargs
            )
        elif isinstance(data, DataLoader):
            loader = data
        else:
            raise TypeError(f"Expected Tensor or DataLoader, got {type(data)}")

        trainer = pl.Trainer(
            **self.trainer_kwargs
        )
        trainer.fit(self, loader, val_dataloaders=val_data if val_data is not None else None)
        return self

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Apply transform before subclass logic
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)
        if self.input_transform:
            x = self.input_transform(x)
        return self._forward(x, y)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # Unpack and transform inputs
        x = batch[0].to(self.device)
        y = batch[-1].to(self.device) if isinstance(batch, (list, tuple)) and len(batch) > 1 else None

        if self.feature_extractor is not None:
            images = batch[0].to(self.device)

            images, y = self.negative_sampling_module((images, y))
            #TODO can be made more efficient
            x =  self.feature_extractor(images)
            if self.input_transform:
                x = self.input_transform(x)
        else:
            if self.input_transform:
                x = self.input_transform(x)
            x, y = self.negative_sampling_module((x, y)) if self.negative_sampling_module else (x, y)

        return self._training_step((x, y), batch_idx)


    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # Unpack and transform inputs
        x = batch[0].to(self.device)
        y = batch[-1].to(self.device) if isinstance(batch, (list, tuple)) and len(batch) > 1 else None

        if self.feature_extractor is not None:
            images = batch[0].to(self.device)

            images, y = self.negative_sampling_module((images, y),seed=batch_idx if self.deterministic_val else None)
            #TODO can be made more efficient
            x =  self.feature_extractor(images)
            if self.input_transform:
                x = self.input_transform(x)
        else:
            if self.input_transform:
                x = self.input_transform(x)
            x, y = self.negative_sampling_module((x, y),seed=batch_idx if self.deterministic_val else None) if self.negative_sampling_module else (x, y)


        return self._validation_step((x, y), batch_idx)

    @abstractmethod
    def _training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Subclasses implement this to compute and return training loss
        from (x, y) after transform.
        """

    def _validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        By default, mirror training_step behavior for validation.
        Subclasses can override if needed.
        """
        return self._training_step(batch, batch_idx)

    @abstractmethod
    def _forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Subclasses implement this to compute confidence given transformed x (and y if needed).
        """


