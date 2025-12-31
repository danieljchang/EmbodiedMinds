from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch
import os
import glob
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_loader import build_dataloader

class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", device="cpu"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.to(device)

    def encode(self, texts):
        # texts: list[str]
        toks = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = toks["input_ids"].to(self.device)
        attn = toks["attention_mask"].to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attn)
        # pooled CLS embedding
        return out.pooler_output  # (B, hidden_dim)

def collate_fn(batch):
    # batch: list of dicts from EmbodiedDataset
    instrs = [b["instruction"] for b in batch]
    # demo_images: dataset provides demo_images (num_demos,3,H,W) per example
    # average demo frames per example to get one demo image (simple first step)
    demo_imgs = torch.stack([b["demo_images"].mean(dim=0) for b in batch], dim=0)
    current_imgs = torch.stack([b["current_image"] for b in batch], dim=0)
    # targets: take last valid action from first demo (or -1 padded)
    targets = []
    for b in batch:
        seq = b["demo_actions"][0]  # (max_steps,7)
        valid = (seq != -1).all(dim=1)
        idxs = valid.nonzero(as_tuple=False)
        if len(idxs) == 0:
            targets.append([-1]*7)
        else:
            last = idxs[-1].item()
            targets.append(seq[last].tolist())
    targets = torch.tensor(targets, dtype=torch.long)
    return instrs, demo_imgs, current_imgs, targets

def train(
    data_root=None,
    batch_size=8,
    epochs=3,
    lr=1e-4,
    device=None,
    use_3d_preprocessing=True,
    early_stopping_patience=5,
    early_stopping_min_delta=0.001,
    early_stopping_enabled=True,
    vlm_adapter: str = None,
):
    """
    Train the agent model with early stopping.
    
    Args:
        data_root: Path to data directory
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        lr: Learning rate
        device: Device to train on
        use_3d_preprocessing: Use 3D preprocessing
        early_stopping_patience: Number of epochs to wait before stopping if no improvement
        early_stopping_min_delta: Minimum change to qualify as an improvement
        early_stopping_enabled: Enable early stopping
    """
    # Import here to avoid circular import
    from src.models.agent_model import AgentModel
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    # Build dataloaders with 3D preprocessing and proper 80/10/10 splits
    train_dl = build_dataloader(
        batch_size=batch_size, 
        debug=False, 
        data_root=data_root, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with models
        use_3d_preprocessing=use_3d_preprocessing,
        device=device,
        split="train",  # 80% of data
        seed=42
    )
    val_dl = build_dataloader(
        batch_size=batch_size, 
        debug=False,  # Use full validation set, not debug mode
        data_root=data_root, 
        num_workers=0,
        use_3d_preprocessing=use_3d_preprocessing,
        device=device,
        split="val",  # 10% of data
        seed=42
    )

    # model
    bins = [101,101,101,121,121,121,2]
    model = AgentModel(
        token_dim=256,
        out_dim=512,
        bins=bins,
        device=device,
        vlm_adapter_name=vlm_adapter,
    ).to(device)

    # optimizer: only trainable params (projections + policy + heads)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)

    loss_fn = None  # OutputHeads.loss used below

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch} train")
        running_loss = 0.0
        for batch in pbar:
            if use_3d_preprocessing:
                # New 3D preprocessing format
                instructions = batch['instructions']
                demo_3d_objects = batch['demo_3d_objects']
                current_3d_objects = batch['current_3d_objects']
                demo_actions = batch.get('demo_actions', None)
                current_images = batch.get('current_images', None)
                targets = batch['targets'].to(device)

                logits = model.forward(
                    instructions,
                    demo_3d_objects,
                    current_3d_objects,
                    demo_actions,
                    current_images=current_images,
                )
            else:
                # Old format (for backward compatibility)
                instrs, demo_imgs, current_imgs, targets = batch
                demo_imgs = demo_imgs.to(device)
                current_imgs = current_imgs.to(device)
                targets = targets.to(device)
                # Note: Old format won't work with new AgentModel - would need old model
                raise NotImplementedError("Old format not supported with new AgentModel")

            loss = model.heads.loss(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (1 + pbar.n))

        # validation
        model.eval()
        correct = [0]*7
        total = 0
        all_val_predictions = []
        all_val_targets = []
        all_val_instructions = []
        
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="val"):
                if use_3d_preprocessing:
                    instructions = batch['instructions']
                    demo_3d_objects = batch['demo_3d_objects']
                    current_3d_objects = batch['current_3d_objects']
                    demo_actions = batch.get('demo_actions', None)
                    current_images = batch.get('current_images', None)
                    targets = batch['targets'].to(device)
                    
                    logits = model.forward(
                        instructions,
                        demo_3d_objects,
                        current_3d_objects,
                        demo_actions,
                        current_images=current_images,
                    )
                else:
                    raise NotImplementedError("Old format not supported")
                
                preds = model.heads.predict(logits).cpu()
                tgt = targets.cpu()
                
                # Save predictions and targets for analysis
                all_val_predictions.append(preds.numpy())
                all_val_targets.append(tgt.numpy())
                all_val_instructions.extend(instructions)
                
                mask = (tgt != -1)
                for i in range(7):
                    valid = mask[:, i]
                    if valid.sum().item() == 0:
                        continue
                    correct[i] += (preds[valid, i] == tgt[valid, i]).sum().item()
                total += tgt.size(0)
        accs = [c / max(1, total) for c in correct]
        avg_val_acc = sum(accs) / len(accs)
        train_loss = running_loss / len(train_dl)
        
        # Calculate validation loss (approximate from accuracy)
        # Lower accuracy = higher loss (inverse relationship)
        val_loss = 1.0 - avg_val_acc
        
        print(f"Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={avg_val_acc:.4f} val_accs={[f'{a:.3f}' for a in accs]}")
        
        # Early stopping check
        improved = False
        if early_stopping_enabled:
            if val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch
                improved = True
                print(f"  ✓ Validation improved! Best val_loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter}/{early_stopping_patience} epochs (best: {best_val_loss:.4f} at epoch {best_epoch})")
        
        # Save best checkpoint
        if improved or epoch == 0:
            try:
                ckpt = {"model_state": model.state_dict(), "bins": bins, "epoch": epoch, "val_loss": val_loss, "val_acc": avg_val_acc}
                torch.save(ckpt, f"checkpoints/agent_best.pt")
                print(f"  Saved best checkpoint: checkpoints/agent_best.pt")
            except RuntimeError as e:
                if "file write failed" in str(e) or "disk" in str(e).lower():
                    print(f"  ⚠️  Disk full! Cannot save checkpoint. Free up space and try again.")
                    # Try to delete old checkpoints
                    old_checkpoints = sorted(glob.glob("checkpoints/agent_epoch*.pt"), key=os.path.getmtime)[:-3]  # Keep last 3
                    for old_ckpt in old_checkpoints:
                        try:
                            os.remove(old_ckpt)
                            print(f"  Deleted old checkpoint: {old_ckpt}")
                        except:
                            pass
                else:
                    raise
        
        # Save epoch checkpoint (only keep last 3 to save space)
        try:
            ckpt = {"model_state": model.state_dict(), "bins": bins, "epoch": epoch, "val_loss": val_loss, "val_acc": avg_val_acc}
            torch.save(ckpt, f"checkpoints/agent_epoch{epoch}.pt")
            
            # Clean up old checkpoints (keep only last 3 + best)
            epoch_checkpoints = sorted(glob.glob("checkpoints/agent_epoch*.pt"), key=os.path.getmtime)
            if len(epoch_checkpoints) > 3:
                for old_ckpt in epoch_checkpoints[:-3]:
                    try:
                        os.remove(old_ckpt)
                    except:
                        pass
        except RuntimeError as e:
            if "file write failed" in str(e) or "disk" in str(e).lower():
                print(f"  ⚠️  Disk full! Cannot save epoch checkpoint. Free up space.")
            else:
                raise
        
        # Save training metrics (basic accuracy metrics)
        import json
        import numpy as np
        from pathlib import Path
        metrics_dir = Path("logs")
        metrics_dir.mkdir(exist_ok=True)
        
        # Concatenate all predictions and targets
        val_predictions = np.concatenate(all_val_predictions, axis=0) if all_val_predictions else []
        val_targets = np.concatenate(all_val_targets, axis=0) if all_val_targets else []
        
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracies": {
                "x": accs[0], "y": accs[1], "z": accs[2],
                "roll": accs[3], "pitch": accs[4], "yaw": accs[5],
                "gripper": accs[6]
            },
            "avg_accuracy": avg_val_acc,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "num_val_samples": len(val_predictions)
        }
        with open(metrics_dir / f"training_metrics_epoch_{epoch}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save predictions vs targets for detailed analysis
        predictions_file = metrics_dir / f"predictions_epoch_{epoch}.npz"
        np.savez(
            predictions_file,
            predictions=val_predictions,
            targets=val_targets,
            instructions=all_val_instructions
        )
        print(f"  Saved predictions to: {predictions_file}")
        
        # Early stopping
        if early_stopping_enabled and patience_counter >= early_stopping_patience:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered!")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
            print(f"No improvement for {patience_counter} epochs")
            print(f"{'='*60}\n")
            break

if __name__ == "__main__":
    # simple CLI-friendly entry
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Number of epochs to wait before early stopping",
    )
    p.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.001,
        help="Minimum change to qualify as improvement",
    )
    p.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping",
    )
    p.add_argument(
        "--vlm-adapter",
        type=str,
        default=None,
        help=(
            "Optional VLM adapter name to use instead of the legacy BERT text "
            "encoder. Examples: llama-3.2-11b-vision-ins, internvl-2.5-8b, "
            "internvl-3-8b, qwen2-vl-7b-ins, qwen2.5-vl-7b-ins, ovis2-16b, "
            "gemma-3-12b-it."
        ),
    )

    args = p.parse_args()
    train(
        data_root=args.data_root,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_enabled=not args.no_early_stopping,
        vlm_adapter=args.vlm_adapter,
    )