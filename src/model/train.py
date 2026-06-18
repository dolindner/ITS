import os
from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from confidence.supervised.ml.energy import EnergyPredictionConfidence

from model.basic_networks import make_deterministic

from model.classifier import Classifier, MyProgressBar
from safetensors import safe_open
from safetensors.torch import save_file

from pytorch_lightning.loggers import CSVLogger
try:
    from pytorch_lightning.loggers import WandbLogger
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

def train_and_get_model(
    model: nn.Module,
    model_dir_path: Path,
    model_name: str,
    train_loader,
    val_loader,
    trainer_kwargs = None,
    optimizer_type = torch.optim.AdamW,
    optimizer_kwargs = None,
    load_if_exists: bool = True,
    monitor: str = "val_acc",
    monitor_mode: str = "max",
    use_wandb: bool = True,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    log_path: str | Path | None = None,
        batch_transform = None,
custom_loss=None,
        strict=True, custom_loss_on_data=False
):
    """
    Train (unless cached) and save best model.
    log_path: optional root directory for logs & checkpoints. If None -> model_dir_path/model_name_logs.
    wandb_entity: optional; omit to use current user (not required).
    """
    # Ensure Path
    if not isinstance(model_dir_path, Path):
        model_dir_path = Path(model_dir_path)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    model_path = model_dir_path / f"{model_name}.safetensors"

    # Load existing weights (safetensors)
    if load_if_exists and (model_path.exists() or model_path.with_suffix(".pt").exists()):
        if model_path.with_suffix(".safetensors").exists():
            print(f"Loading existing weights from {model_path}")
            state_dict = {}
            with safe_open(str(model_path), framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
            model.load_state_dict(state_dict, strict=strict)
            model = model.eval()
            make_deterministic(model)
            return model, str(model_path)
        else:
            # fallback to .pt
            pt_path = model_path.with_suffix(".pt")
            print(f"Loading existing weights from {pt_path}")
            model.load_state_dict(torch.load(pt_path, map_location='cpu'), strict=strict)
            model = model.eval()
            make_deterministic(model)
            return model, str(pt_path)




    print(f"Training model and will save to {model_path}")

    # Determine logging root
    log_root = Path(log_path) if log_path is not None else (model_dir_path / f"{model_name}_logs")
    log_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = log_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    lightning_model = Classifier(
        model,
        optimizer_class=optimizer_type,
        optimizer_params={"lr": 1e-3} if optimizer_kwargs is None else optimizer_kwargs
        , batch_transform=batch_transform, custom_loss=custom_loss, custom_loss_on_data=custom_loss_on_data,
    )
    progress_bar = MyProgressBar()

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode=monitor_mode,
        save_top_k=1,
        verbose=True,
        dirpath=str(checkpoints_dir),
        filename=f"{model_name}" + "-{epoch:02d}-{" + monitor + ":.4f}",
    )

    # Logging setup (CSV always)
    csv_logger = CSVLogger(save_dir=str(log_root), name=model_name)
    loggers = [csv_logger]
    if use_wandb and _WANDB_AVAILABLE:
        try:
            wandb_kwargs = dict(
                project=wandb_project or model_name,
                name=model_name,
                save_dir=str(log_root),
                log_model=False
            )
            if wandb_entity:  # only pass if provided
                wandb_kwargs["entity"] = wandb_entity
            wandb_logger = WandbLogger(**wandb_kwargs)
            loggers.append(wandb_logger)
        except Exception:
            pass  # silently ignore wandb issues

    default_trainer_kwargs = {
        "accelerator": "auto",
        "max_epochs": 30,
        "precision": "16-mixed",
    }
    trainer_kwargs = default_trainer_kwargs if trainer_kwargs is None else {**default_trainer_kwargs, **trainer_kwargs}

    callbacks = trainer_kwargs.pop("callbacks", [])
    callbacks.extend([checkpoint_callback, progress_bar])
    trainer = pl.Trainer(callbacks=callbacks, logger=loggers, **trainer_kwargs)

    trainer.fit(lightning_model, train_loader, val_loader)

    best_ckpt = checkpoint_callback.best_model_path
    print(f"Best Lightning checkpoint: {best_ckpt}")

    lightning_model = Classifier.load_from_checkpoint(
        best_ckpt,
        model=model,
        optimizer_class=optimizer_type,
        optimizer_params={"lr": 1e-3} if optimizer_kwargs is None else optimizer_kwargs,
        batch_transform=batch_transform,custom_loss=custom_loss
    )
    model = lightning_model.model


    #delete pt or safetensors if they exist
    if model_path.with_suffix(".pt").exists():
        os.remove(model_path.with_suffix(".pt"))
    if model_path.with_suffix(".safetensors").exists():
        os.remove(model_path.with_suffix(".safetensors"))

    # Save safetensors
    try:
        save_file(model.state_dict(), str(model_path), metadata={"model_name": model_name})
    except Exception:
        # fallback: make contiguous
        try:
            state_dict = make_contiguous(model.state_dict())
            save_file(state_dict, str(model_path), metadata={"model_name": model_name})
        except Exception as e:
            #save usign torch.save
            torch.save(model.state_dict(), str(model_path.with_suffix(".pt")))
    print(f"Best model weights saved to {model_path}")
    model = model.eval()
    make_deterministic(model)
    return model, str(model_path)


def make_contiguous(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            new_state_dict[k] = v.contiguous()
        else:
            new_state_dict[k] = v
    return new_state_dict


import os
from pathlib import Path
import torch
from safetensors.torch import save_file
from safetensors import safe_open
from pytorch_lightning.callbacks import ModelCheckpoint
from model.classifier import MyProgressBar
from confidence.supervised.ml.energy import EnergyPredictionConfidence

def train_or_load_energy_model(
    model: torch.nn.Module,
    model_dir_path: Path,
    model_name: str,
    train_loader,
    val_loader,
    negative_sampling_module=None,
    trainer_kwargs=None,
    optimizer_type=torch.optim.Adam,
    optimizer_kwargs=None,
    loss_type="bce",
    gp_weight=100.0,
    load_if_exists=True,
    monitor="val_acc",
    monitor_mode="max",
    use_wandb: bool = True,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    log_path: str | Path | None = None,
    deterministic_val = False
):
    """
    Train EnergyPredictionConfidence model (unless cached) and save best weights using safetensors.
    All parameters are passed directly to EnergyPredictionConfidence.
    log_path: optional root directory for logs & checkpoints. If None -> model_dir_path/model_name_logs.
    wandb_entity: optional; omit to use current user (not required).
    """
    if not isinstance(model_dir_path, Path):
        model_dir_path = Path(model_dir_path)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    model_path = model_dir_path / f"{model_name}_energy_conf_best.safetensors"

    # Load existing weights if available
    if load_if_exists and model_path.exists():
        print(f"Loading existing weights from {model_path}")
        state_dict = {}
        with safe_open(str(model_path), framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        model.load_state_dict(state_dict)

        make_deterministic(model)
        model = model.eval()


        return EnergyPredictionConfidence(
            energy_model=model,
            loss_type=loss_type,
            negative_sampling_module=negative_sampling_module,
            trainer_kwargs=trainer_kwargs,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            gp_weight=gp_weight,
        )

    # Determine logging root - shortened to avoid Windows path length issues
    log_root = Path(log_path) if log_path is not None else (model_dir_path / "energy_logs")
    log_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = model_dir_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Setup checkpoint callback
    chkpointer = ModelCheckpoint(
        monitor=monitor,
        mode=monitor_mode,
        save_top_k=1,verbose=True,
        dirpath=str(checkpoints_dir),
        filename=f"{model_name}_energy_conf_best"
    )

    # Logging setup (CSV always) - use shorter name to avoid path length issues
    csv_logger = CSVLogger(save_dir=str(log_root), name="{model_name}_energy")
    loggers = [csv_logger]
    if use_wandb and _WANDB_AVAILABLE:
        try:
            wandb_kwargs = dict(
                project=wandb_project or f"{model_name}_energy",
                name=f"{model_name}_energy",
                save_dir=str(log_root),
                log_model=False
            )
            if wandb_entity:  # only pass if provided
                wandb_kwargs["entity"] = wandb_entity
            wandb_logger = WandbLogger(**wandb_kwargs)
            loggers.append(wandb_logger)
        except Exception:
            pass  # silently ignore wandb issues

    # Initialize EnergyPredictionConfidence
    energy_conf = EnergyPredictionConfidence(
        energy_model=model,
        loss_type=loss_type,
        negative_sampling_module=negative_sampling_module,
        trainer_kwargs={
            **(trainer_kwargs or {}),
            "callbacks": [MyProgressBar(), chkpointer],
            "logger": loggers,
        },
        optimizer_type=optimizer_type,
        optimizer_kwargs=optimizer_kwargs,
        gp_weight=gp_weight,
    ).cuda()

    if deterministic_val:
        energy_conf.deterministic_val = True

    # Train
    energy_conf.fit(train_loader, val_data=val_loader)

    # Load best checkpoint into model
    best_ckpt = chkpointer.best_model_path
    print(f"Best Lightning checkpoint: {best_ckpt}")

    energy_conf = EnergyPredictionConfidence.load_from_checkpoint(
        best_ckpt,
        energy_model=model,  # pass args needed to re-init
        loss_type=loss_type,
        negative_sampling_module=negative_sampling_module,
        trainer_kwargs=trainer_kwargs,
        optimizer_type=optimizer_type,
        optimizer_kwargs=optimizer_kwargs,
        gp_weight=gp_weight,
    )
    model = energy_conf.energy_model

    # Save safetensors
    try:
        save_file(model.state_dict(), str(model_path), metadata={"model_name": model_name})
    except Exception:
        # fallback: make contiguous
        state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
        save_file(state_dict, str(model_path), metadata={"model_name": model_name})

    print(f"Best model weights saved to {model_path}")
    make_deterministic(model)
    model = model.eval()
    energy_conf.energy_model = model
    return energy_conf

