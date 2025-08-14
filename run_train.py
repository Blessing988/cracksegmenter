# ------------------------------------------------------------
# Generic training routine (place in the same script)
# ------------------------------------------------------------
import math
import torch
from typing import Dict, Any, Callable, Tuple


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    config: Dict[str, Any],
    *,
    num_epochs: int,
    save_model_path: str,
    save_metric_path: str,
    early_stopping_patience: int | None = None,
    train_step_fn: Callable | None = None,
    val_step_fn: Callable   | None = None,
    criterion: torch.nn.Module | None = None,  # only used by baseline
) -> Tuple[float, Dict[str, float]]:
    """
    Universal training loop with early stopping & checkpointing.

    Returns
    -------
    best_val_loss : float
    best_val_metrics : dict
    """

    # --------------------------------------------------------
    # Decide which helpers to use
    # --------------------------------------------------------
    if train_step_fn is None or val_step_fn is None:
        if config["model"]["baseline"]:
            train_step_fn = train_one_epoch_baseline
            val_step_fn   = validate_baseline
        else:
            train_step_fn = train_one_epoch
            val_step_fn   = validate

    # Early‑stopping patience
    if early_stopping_patience is None:
        early_stopping_patience = config["training"]["early_stopping_patience"]

    best_val_loss     = math.inf
    best_val_metrics  = {}
    epochs_no_improve = 0

    # --------------------------------------------------------
    # Main training loop
    # --------------------------------------------------------
    for epoch in range(num_epochs):
        if config["model"]["baseline"]:
            train_loss, train_metrics = train_step_fn(
                model, train_loader, criterion, optimizer,
                device, config["model"]["num_classes"]
            )
        else:
            train_loss, train_metrics = train_step_fn(
                model, train_loader, optimizer,
                device, config["model"]["num_classes"]
            )

        val_loss, val_metrics = val_step_fn(
            model, val_loader, device, config["model"]["num_classes"]
        )

        scheduler.step(val_loss)

        # ------------ logging ------------
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Train Metr : {train_metrics}")
        print(f"  Val   Loss : {val_loss:.4f}")
        print(f"  Val   Metr : {val_metrics}\n")

        # ------------ checkpoint / early‑stop ------------
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_val_metrics = val_metrics
            epochs_no_improve = 0

            torch.save(model.state_dict(), save_model_path)
            print("  ✔  New best model saved.")
        else:
            epochs_no_improve += 1
            print(f"  ✖  No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping triggered.")
            break

        # Persist metrics each epoch
        save_metrics(epoch, train_loss, train_metrics,
                     val_loss, val_metrics, save_metric_path)

    return best_val_loss, best_val_metrics
