"""
Reads results/training_log.json (produced by train.py) and generates
comparison plots: trainable-parameter count (log scale), accuracy/F1
comparison, and training curves for both variants side by side.
"""

import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def main():
    with open(os.path.join(config.RESULTS_DIR, "training_log.json")) as f:
        log = json.load(f)

    full = log["results"]["full"]
    lora = log["results"]["lora"]
    hist_full = log["history"]["full"]
    hist_lora = log["history"]["lora"]

    # --- Trainable parameter count comparison (log scale) ---
    plt.figure(figsize=(6, 4.5))
    modes = ["Full fine-tuning", "LoRA"]
    params = [full["trainable_params"], lora["trainable_params"]]
    bars = plt.bar(modes, params, color=["#C44E52", "#4C72B0"])
    plt.yscale("log")
    plt.ylabel("Trainable parameters (log scale)")
    plt.title("Trainable Parameters: Full Fine-Tuning vs LoRA")
    for bar, val in zip(bars, params):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,}",
                  ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "trainable_params_comparison.png"), dpi=140)
    plt.close()

    # --- Accuracy / F1 comparison ---
    plt.figure(figsize=(6, 4.5))
    x = range(2)
    width = 0.35
    accs = [full["final_val_accuracy"], lora["final_val_accuracy"]]
    f1s = [full["final_val_f1"], lora["final_val_f1"]]
    plt.bar([i - width / 2 for i in x], accs, width, label="Accuracy", color="#4C72B0")
    plt.bar([i + width / 2 for i in x], f1s, width, label="F1", color="#DD8452")
    plt.xticks(list(x), modes)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("MRPC Validation Accuracy / F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "accuracy_f1_comparison.png"), dpi=140)
    plt.close()

    # --- Training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(hist_full["val_accuracy"], label="Full FT", color="#C44E52", marker="o")
    axes[0].plot(hist_lora["val_accuracy"], label="LoRA", color="#4C72B0", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation Accuracy")
    axes[0].set_title("Validation Accuracy over Training")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(hist_full["val_f1"], label="Full FT", color="#C44E52", marker="o")
    axes[1].plot(hist_lora["val_f1"], label="LoRA", color="#4C72B0", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation F1")
    axes[1].set_title("Validation F1 over Training")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, "training_curves.png"), dpi=140)
    plt.close()

    # --- Print summary table ---
    print("=" * 70)
    print(f"{'Metric':<28}{'Full Fine-Tuning':>20}{'LoRA':>20}")
    print("=" * 70)
    print(f"{'Trainable params':<28}{full['trainable_params']:>20,}{lora['trainable_params']:>20,}")
    print(f"{'% of total params':<28}{full['trainable_pct']:>19.3f}%{lora['trainable_pct']:>19.3f}%")
    print(f"{'Final val accuracy':<28}{full['final_val_accuracy']:>20.4f}{lora['final_val_accuracy']:>20.4f}")
    print(f"{'Final val F1':<28}{full['final_val_f1']:>20.4f}{lora['final_val_f1']:>20.4f}")
    print(f"{'Best val accuracy':<28}{full['best_val_accuracy']:>20.4f}{lora['best_val_accuracy']:>20.4f}")
    print(f"{'Best val F1':<28}{full['best_val_f1']:>20.4f}{lora['best_val_f1']:>20.4f}")
    print(f"{'Training time (min)':<28}{full['training_time_minutes']:>20.2f}{lora['training_time_minutes']:>20.2f}")
    print("=" * 70)
    reduction = full["trainable_params"] / lora["trainable_params"]
    print(f"LoRA trains {reduction:.1f}x fewer parameters than full fine-tuning.")
    print("\nSaved: results/trainable_params_comparison.png, "
          "results/accuracy_f1_comparison.png, results/training_curves.png")


if __name__ == "__main__":
    main()
