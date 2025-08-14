# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
from datasets import  get_transforms, get_dataloader
from metrics import evaluate_metrics, save_metrics
from utils import save_checkpoint, dice_metric, xor_metric, hm_metric, create_mask
from create_model import define_model
import yaml
import numpy as np
from losses import loss_ce, loss_inter, loss_intra, BinaryDiceLoss
import math
import torch
from typing import Dict, Any, Callable, Tuple, Optional
import argparse

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


def train_one_epoch_baseline(model, dataloader, criterion, optimizer, device, num_classes=1):
    model.train()
    epoch_loss = 0
    metrics = {'IoU': [], 'Dice': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        
        if config['model']['architecture'] == 'FCN':
            outputs = model(images)['out']
        else:
            outputs = model(images)
            
        # if isinstance(outputs, tuple):  # Check if outputs is a tuple
        #     outputs, _ = outputs  # Extract the predictions (ignore attention maps)

        outputs = outputs.squeeze(1)
        loss = criterion(outputs, masks.float()) # This worked
        preds = torch.sigmoid(outputs) > 0.5

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)

        batch_metrics = evaluate_metrics(preds.cpu(), masks.cpu(), num_classes)
        for key in metrics:
            metrics[key].append(batch_metrics[key])

    epoch_loss /= len(dataloader.dataset)
    avg_metrics = {key: np.nanmean(metrics[key]) for key in metrics}

    return epoch_loss, list(avg_metrics.values())


def validate_baseline(model, dataloader, criterion, device, num_classes=1):
    model.eval()
    epoch_loss = 0
    metrics = {'IoU': [], 'Dice': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            if config['model']['architecture'] == 'FCN':
                outputs = model(images)['out']
            
            else:
                outputs = model(images)
                
            # if isinstance(outputs, tuple):  # Check if outputs is a tuple
            #     outputs, _ = outputs  # Extract the predictions (ignore attention maps)

            outputs = outputs.squeeze(1)
            loss = criterion(outputs, masks.float())
            preds = torch.sigmoid(outputs) > 0.5

            epoch_loss += loss.item() * images.size(0)

            # Calculate metrics
            batch_metrics = evaluate_metrics(preds.cpu(), masks.cpu(), num_classes)
            for key in metrics:
                metrics[key].append(batch_metrics[key])

    epoch_loss /= len(dataloader.dataset)

    # Average metrics
    avg_metrics = {key: np.nanmean(metrics[key]) for key in metrics}

    return epoch_loss, list(avg_metrics.values())
    
    
######################################
# train.py (snippet)
def train_one_epoch(model, dataloader, optimizer, device, num_classes=1):
    model.train()
    epoch_loss = 0
    metrics = {'IoU': [], 'Dice': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'HM Score': [], 'XOR Score' : [] }
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle varying number of outputs
        if isinstance(outputs, tuple) and len(outputs) == 8:
            output, att_score, attention_map_f, attention_map_s, attention_map_l, context_f, context_s, context_l = outputs
        else:
            output = outputs[0] if isinstance(outputs, tuple) else outputs
            att_score, context_f, context_s, context_l = None, None, None, None
        
        B, C, H, W = output.shape
        target = output.argmax(dim=1)
        
        # Compute CE loss
        ce_loss = loss_ce(output, target)
        
        # Compute additional losses if available
        total_inter_loss = 0
        total_intra_loss = 0
        if att_score is not None and context_f is not None:
            for i in range(B):
                if context_s is not None and context_l is not None:
                    inter_fs = loss_inter(context_f[i], context_s[i], torch.tensor([1]).to(device))
                    inter_sl = loss_inter(context_s[i], context_l[i], torch.tensor([1]).to(device))
                    total_inter_loss += 1.5 * (inter_fs + inter_sl)
                if att_score is not None:
                    L = att_score.size(-1)
                    total_intra_loss += 1.3 * loss_intra(att_score[i], torch.eye(L, device=device))
            total_inter_loss /= B
            total_intra_loss /= B
        
        loss = ce_loss + total_inter_loss + total_intra_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        
        batch_metrics_list = []
        for i in range(B):
            current_pred = target[i]
            current_mask = masks[i]
            current_pred_mask = create_mask(current_pred, current_mask, restrict_to_gt=True)
            image_metrics = evaluate_metrics(current_pred_mask, current_mask, num_classes)
            batch_metrics_list.append(image_metrics)
        
        batch_metrics = {key: np.mean([m[key] for m in batch_metrics_list]) for key in batch_metrics_list[0].keys()}
        for key in metrics:
            metrics[key].append(batch_metrics[key])

    epoch_loss /= len(dataloader.dataset)
    avg_metrics = {key: np.nanmean(metrics[key]) for key in metrics}
    return epoch_loss, list(avg_metrics.values())

def validate(model, dataloader, device, num_classes=1):
    model.eval()
    epoch_loss = 0
    metrics = {'IoU': [], 'Dice': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'HM Score': [], 'XOR Score' : [] }
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            if isinstance(outputs, tuple) and len(outputs) == 8:
                output, att_score, attention_map_f, attention_map_s, attention_map_l, context_f, context_s, context_l = outputs
            else:
                output = outputs[0] if isinstance(outputs, tuple) else outputs
                att_score, context_f, context_s, context_l = None, None, None, None
            
            B, C, H, W = output.shape
            target = output.argmax(dim=1)
            
            ce_loss = loss_ce(output, target)
            total_inter_loss = 0
            total_intra_loss = 0
            if att_score is not None and context_f is not None:
                for i in range(B):
                    if context_s is not None and context_l is not None:
                        inter_fs = loss_inter(context_f[i], context_s[i], torch.tensor([1]).to(device))
                        inter_sl = loss_inter(context_s[i], context_l[i], torch.tensor([1]).to(device))
                        total_inter_loss += 1.5 * (inter_fs + inter_sl)
                    if att_score is not None:
                        L = att_score.size(-1)
                        total_intra_loss += 1.3 * loss_intra(att_score[i], torch.eye(L, device=device))
                total_inter_loss /= B
                total_intra_loss /= B
            
            loss = ce_loss + total_inter_loss + total_intra_loss
            epoch_loss += loss.item() * images.size(0)
            
            batch_metrics_list = []
            for i in range(B):
                current_pred = target[i]
                current_mask = masks[i]
                current_pred_mask = create_mask(current_pred, current_mask, restrict_to_gt=True)
                image_metrics = evaluate_metrics(current_pred_mask, current_mask, num_classes)
                batch_metrics_list.append(image_metrics)
            
            batch_metrics = {key: np.mean([m[key] for m in batch_metrics_list]) for key in batch_metrics_list[0].keys()}
            for key in metrics:
                metrics[key].append(batch_metrics[key])

    epoch_loss /= len(dataloader.dataset)
    avg_metrics = {key: np.nanmean(metrics[key]) for key in metrics}
    return epoch_loss, list(avg_metrics.values())


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
    early_stopping_patience: Optional[int] = None,
    train_step_fn: Optional[Callable] = None,
    val_step_fn: Optional[Callable] = None,
    criterion: Optional[torch.nn.Module] = None,
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

        # ---------- TRAIN ----------
        if config["model"]["baseline"]:
            train_loss, train_metrics = train_step_fn(
                model, train_loader,
                criterion, optimizer, device,                 # <-- order fixed
                config["model"]["num_classes"],
            )
        else:
            train_loss, train_metrics = train_step_fn(
                model, train_loader,
                optimizer, device,                            # no criterion here
                config["model"]["num_classes"],
            )

        # ---------- VALIDATE ----------
        if config["model"]["baseline"]:
            val_loss, val_metrics = val_step_fn(
                model, val_loader,
                criterion, device,                            # <-- order fixed
                config["model"]["num_classes"],
            )
        else:
            val_loss, val_metrics = val_step_fn(
                model, val_loader,
                device, config["model"]["num_classes"],
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




# ---------- 1. argument parser ---------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a semantic-segmentation model from a YAML config"
    )

    # — required / common overrides
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Root directory to save checkpoints & logs (overrides YAML)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Mini-batch size (overrides YAML)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs (overrides YAML)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Force device selection (default: auto = cuda if available)",
    )

    # …add more flags as you need (lr, weight-decay, etc.)
    return parser
# ---------------------------------------------------------------------------


# ---------- 2. main training routine --------------------------------------
def main(args: argparse.Namespace) -> None:
    # 2-a  select device early so errors surface soon
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        # Optional: disable PyTorch caching if you truly need it off
        os.environ["PYTORCH_DISABLE_CACHE"] = "1"

    # 2-b  load YAML config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as f:
        config: dict = yaml.safe_load(f)

    # 2-c  apply CLI overrides (they win over YAML)
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.save_dir is not None:
        config["utils"]["save_dir"] = args.save_dir

    # 2-d  quick sanity printout
    print("=" * 50, "\nCONFIG VERIFICATION")
    print(f"Dataset        : {config['data']['dataset_name']}")
    print(f"Model Arch     : {config['model']['architecture']}")
    print(f"Batch size     : {config['training']['batch_size']}")
    print(f"Epochs         : {config['training']['num_epochs']}")
    print(f"Save directory : {config['utils']['save_dir']}")
    print("=" * 50)

    # 2-e  dataloaders, model, optimizer … exactly as you already do
    train_tfms, val_tfms = get_transforms()

    train_loader = get_dataloader(
        os.path.join(config["data"]["root_dir"], config["data"]["dataset_name"], "train", "images"),
        os.path.join(config["data"]["root_dir"], config["data"]["dataset_name"], "train", "masks"),
        config["training"]["batch_size"],
        config["data"]["num_workers"],
        config["model"]["num_classes"],
        transform=train_tfms,
        shuffle=True,
    )
    val_loader = get_dataloader(
        os.path.join(config["data"]["root_dir"], config["data"]["dataset_name"], "val", "images"),
        os.path.join(config["data"]["root_dir"], config["data"]["dataset_name"], "val", "masks"),
        config["training"]["batch_size"],
        config["data"]["num_workers"],
        config["model"]["num_classes"],
        transform=val_tfms,
        shuffle=False,
    )

    save_root   = config["utils"]["save_dir"]
    dataset     = config["data"]["dataset_name"]
    model_type  = config["model"]["architecture"]
    os.makedirs(os.path.join(save_root, dataset, model_type), exist_ok=True)
    model_path  = os.path.join(save_root, dataset, model_type, "best.pth")
    metric_path = os.path.join(save_root, dataset, model_type, "metrics_attention.csv")

    model = define_model(model_type).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    criterion = BinaryDiceLoss()

    best_loss, best_metrics = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        config,
        num_epochs=config["training"]["num_epochs"],
        save_model_path=model_path,
        save_metric_path=metric_path,
        criterion=criterion,
    )

    print(f"Training complete. Best val loss: {best_loss:.4f}")
# ---------------------------------------------------------------------------


# ---------- 3. bootstrap ---------------------------------------------------
if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
# ---------------------------------------------------------------------------
