"""
Configuration for the LoRA (Hu et al., 2021) reproduction.

Core experiment: fine-tune a pretrained transformer on a GLUE task two
ways — (1) standard full fine-tuning (every parameter trainable) and
(2) LoRA fine-tuning (backbone frozen, only small rank-decomposition
matrices + the task head trainable) — and compare accuracy, F1, trainable
parameter count, and training time. This directly mirrors the paper's
own headline comparison (Table 2 / Table 8 in the paper: LoRA matches or
approaches full fine-tuning accuracy with orders of magnitude fewer
trainable parameters).
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ---------------------------------------------------------------------------
# DEVIATION FROM PAPER (#1 - model/scale):
# Paper's main results use RoBERTa-base/large, DeBERTa-XXL, GPT-2, and
# GPT-3 175B. We use DistilBERT-base-uncased (66M params) - small enough
# to fine-tune quickly on a single free Colab GPU, still a real pretrained
# transformer with real attention weight matrices to apply LoRA to.
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"   # paper: RoBERTa/DeBERTa/GPT-2/GPT-3

# ---------------------------------------------------------------------------
# DEVIATION FROM PAPER (#2 - task):
# Paper evaluates on the full GLUE benchmark (8 tasks) plus WikiSQL, SAMSum,
# and GPT-3-scale few-shot tasks. We evaluate on one GLUE task, MRPC
# (paraphrase detection), chosen because the paper itself reports MRPC
# with both accuracy AND F1 (matching this project's requested deliverables),
# and its ~3.7k training examples fine-tune quickly.
# ---------------------------------------------------------------------------
GLUE_TASK = "mrpc"     # paper: full GLUE suite (we reproduce 1 of 8 tasks)
MAX_LENGTH = 128
NUM_LABELS = 2

# ---------------------------------------------------------------------------
# LoRA hyperparameters (paper: rank r as low as 1-8 is often enough;
# applied to attention query/value projection matrices, W_q and W_v, in
# the paper's main GPT-3 experiments - we mirror that choice here on
# DistilBERT's equivalent q_lin/v_lin layers).
# ---------------------------------------------------------------------------
LORA_R = 8              # paper: typically tests r in {1,2,4,8,16,64}, r=8 is a common default
LORA_ALPHA = 16          # paper: alpha commonly set to 2x rank as a starting point
LORA_TARGET_MODULES = ["q_lin", "v_lin"]  # paper's main experiments target W_q, W_v
LORA_DROPOUT = 0.1

# ---------------------------------------------------------------------------
# Training hyperparameters.
# DEVIATION FROM PAPER (#3): paper uses task/model-specific tuned learning
# rates from a hyperparameter sweep (different LR for full-FT vs LoRA on
# each task/model, reported in their appendix tables). We use one fixed,
# reasonable LR per method (LoRA needs a notably higher LR than full
# fine-tuning since only a small number of parameters are updated - this
# matches the paper's own observation that LoRA benefits from larger LR).
# ---------------------------------------------------------------------------
EPOCHS = 4
BATCH_SIZE = 16
LR_FULL_FINETUNE = 2e-5    # paper-typical full fine-tuning LR for BERT-family models
LR_LORA = 1e-3             # paper-typical LoRA LR (much higher than full-FT)
WEIGHT_DECAY = 0.01
SEED = 42

WEIGHTS_DIR = os.path.join(RESULTS_DIR, "weights")
