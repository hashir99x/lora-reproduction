"""
Trains DistilBERT on GLUE MRPC two ways - full fine-tuning and LoRA - and
logs accuracy, F1, trainable parameter count, training time, and hardware
for each, so they can be directly compared against each other and against
the paper's headline claim (LoRA matches full fine-tuning accuracy with
far fewer trainable parameters).
"""

import os
import json
import time
import random
import platform
import subprocess

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score

import config
from data import get_dataloaders
from model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_hardware_info():
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)",
    }
    try:
        cpu_model = subprocess.run(
            ["grep", "-m1", "model name", "/proc/cpuinfo"],
            capture_output=True, text=True, check=False
        ).stdout.strip()
        info["cpu_model"] = cpu_model.split(":", 1)[-1].strip() if cpu_model else "unknown"
    except Exception:
        info["cpu_model"] = "unknown"
    return info


def evaluate(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += out.loss.item()
            preds = torch.argmax(out.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return acc, f1, total_loss / len(val_loader)


def train_one_variant(mode, train_loader, val_loader, device):
    set_seed(config.SEED)
    model, trainable_params, total_params = build_model(mode)
    model.to(device)

    lr = config.LR_LORA if mode == "lora" else config.LR_FULL_FINETUNE
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=config.WEIGHT_DECAY,
    )

    print(f"\n=== Training mode: {mode} | trainable params: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.3f}%) | lr={lr} ===")

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}
    start_time = time.time()

    for epoch in range(config.EPOCHS):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out.loss.backward()
            optimizer.step()
            epoch_losses.append(out.loss.item())

        val_acc, val_f1, val_loss = evaluate(model, val_loader, device)
        mean_train_loss = float(np.mean(epoch_losses))
        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_f1"].append(val_f1)
        print(f"[{mode}] Epoch {epoch+1}/{config.EPOCHS} | train_loss={mean_train_loss:.4f} "
              f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

    total_time = time.time() - start_time

    os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
    save_path = os.path.join(config.WEIGHTS_DIR, f"{mode}_model.pt")
    torch.save(model.state_dict(), save_path)

    return {
        "mode": mode,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": 100 * trainable_params / total_params,
        "learning_rate": lr,
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "training_time_seconds": total_time,
        "training_time_minutes": total_time / 60.0,
        "final_val_accuracy": history["val_accuracy"][-1],
        "final_val_f1": history["val_f1"][-1],
        "best_val_accuracy": max(history["val_accuracy"]),
        "best_val_f1": max(history["val_f1"]),
        "history": history,
        "weights_path": save_path,
    }


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, val_loader, tokenizer = get_dataloaders()
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    results = {}
    for mode in ["full", "lora"]:
        results[mode] = train_one_variant(mode, train_loader, val_loader, device)

    log = {
        "hardware": get_hardware_info(),
        "hyperparameters": {
            "model_name": config.MODEL_NAME,
            "glue_task": config.GLUE_TASK,
            "max_length": config.MAX_LENGTH,
            "lora_r": config.LORA_R,
            "lora_alpha": config.LORA_ALPHA,
            "lora_target_modules": config.LORA_TARGET_MODULES,
            "lora_dropout": config.LORA_DROPOUT,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "lr_full_finetune": config.LR_FULL_FINETUNE,
            "lr_lora": config.LR_LORA,
            "weight_decay": config.WEIGHT_DECAY,
            "seed": config.SEED,
        },
        "results": {
            "full": {k: v for k, v in results["full"].items() if k != "history"},
            "lora": {k: v for k, v in results["lora"].items() if k != "history"},
        },
        "history": {
            "full": results["full"]["history"],
            "lora": results["lora"]["history"],
        },
    }

    with open(os.path.join(config.RESULTS_DIR, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print("\n=== Summary ===")
    print(f"Full fine-tuning : {results['full']['trainable_params']:,} trainable params "
          f"| acc={results['full']['final_val_accuracy']:.4f} | f1={results['full']['final_val_f1']:.4f} "
          f"| time={results['full']['training_time_minutes']:.2f} min")
    print(f"LoRA fine-tuning : {results['lora']['trainable_params']:,} trainable params "
          f"| acc={results['lora']['final_val_accuracy']:.4f} | f1={results['lora']['final_val_f1']:.4f} "
          f"| time={results['lora']['training_time_minutes']:.2f} min")
    reduction = results["full"]["trainable_params"] / results["lora"]["trainable_params"]
    print(f"Parameter reduction: {reduction:.1f}x fewer trainable params with LoRA")
    print("Training log saved to results/training_log.json")


if __name__ == "__main__":
    main()
