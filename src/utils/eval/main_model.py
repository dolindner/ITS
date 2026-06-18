import torch
from torchmetrics import Accuracy, FBetaScore
from tqdm import tqdm


def evaluate_base_model(
        model,
        data_loader,
        device=None,
        show_progress: bool = True,
):
    """
    Evaluate a base model (no search/transform) on a dataloader and compute standard classification metrics.

    Args:
        model: PyTorch model used for predictions. Device is inferred from the model if `device` is None.
        data_loader: Iterable or DataLoader yielding tuples (inputs, targets).
        device: Optional torch.device to run evaluation on. If None, derived from model parameters.
        show_progress: If True, display a tqdm progress bar and live metric updates.

    Returns:
        dict:
            - accuracy: float or None if the loader is empty.
            - f2_macro: float (F2 score with macro averaging) or None if the loader is empty.
            - f2_micro: float (F2 score with micro averaging) or None if the loader is empty.
            - f2_weighted: float (F2 score with weighted averaging) or None if the loader is empty.

    Raises:
        ValueError: If model outputs (logits) have fewer than 2 dimensions (expected shape [batch, num_classes]).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Infer num_classes from first batch
    iterator = iter(data_loader)
    try:
        first_data, first_target = next(iterator)
    except StopIteration:
        return {
            "accuracy": None,
            "f2_macro": None,
            "f2_micro": None,
            "f2_weighted": None,
        }

    first_data = first_data.to(device)
    first_target = first_target.to(device)
    with torch.no_grad():
        logits = model(first_data)
        if logits.dim() < 2:
            raise ValueError("Model logits must have shape [batch, num_classes].")
        num_classes = logits.shape[-1]
        preds = logits.argmax(dim=-1)

    acc = Accuracy(task='multiclass', num_classes=num_classes).to(device)
    f2_macro = FBetaScore(task='multiclass', num_classes=num_classes, beta=2.0, average='macro').to(device)
    f2_micro = FBetaScore(task='multiclass', num_classes=num_classes, beta=2.0, average='micro').to(device)
    f2_weighted = FBetaScore(task='multiclass', num_classes=num_classes, beta=2.0, average='weighted').to(device)

    acc.update(preds, first_target)
    f2_macro.update(preds, first_target)
    f2_micro.update(preds, first_target)
    f2_weighted.update(preds, first_target)

    with torch.no_grad():
        iterator_wrapped = iterator
        if show_progress:
            iterator_wrapped = tqdm(iterator_wrapped, desc="BaseModel eval", leave=False)
        for data, target in iterator_wrapped:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            preds = logits.argmax(dim=-1)

            acc.update(preds, target)
            f2_macro.update(preds, target)
            f2_micro.update(preds, target)
            f2_weighted.update(preds, target)

            if show_progress:
                try:
                    iterator_wrapped.set_postfix({
                        "Acc": f"{acc.compute().item():.3f}",
                        "F2_mac": f"{f2_macro.compute().item():.3f}",
                        "F2_mic": f"{f2_micro.compute().item():.3f}",
                        "F2_w": f"{f2_weighted.compute().item():.3f}",
                    })
                except Exception:
                    pass

    return {
        "accuracy": acc.compute().item(),
        "f2_macro": f2_macro.compute().item(),
        "f2_micro": f2_micro.compute().item(),
        "f2_weighted": f2_weighted.compute().item(),
    }
