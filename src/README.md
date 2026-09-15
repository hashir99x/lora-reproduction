# LoRA Reproduction — "LoRA: Low-Rank Adaptation of Large Language Models"

This project reproduces the **core mechanism and headline comparison** of
Hu et al.'s 2021 LoRA paper: fine-tune a pretrained transformer two ways —
standard full fine-tuning, and LoRA (frozen backbone + small trainable
low-rank adapters on the attention projections) — and compare accuracy,
F1, trainable parameter count, and training time.

**Read this first:** the paper's main results use RoBERTa/DeBERTa/GPT-2/
GPT-3 across the full GLUE benchmark and other large-scale tasks. This
reproduction uses DistilBERT-base (66M params) on one GLUE task (MRPC),
chosen to run in a few minutes on a single free Colab GPU. The point of
this project is a faithful, from-scratch reimplementation of LoRA's exact
mechanism (frozen W0, Gaussian-init A, zero-init B, alpha/r scaling,
applied to attention query/value projections) and an honest, documented
comparison — not matching the paper's absolute numbers on a different
model/task/scale. See "Differences from the Paper" below.

**This project needs a GPU and internet access to real pretrained weights
and a real dataset — it cannot run in an offline CPU-only sandbox.** All
code is written and syntax-verified, and the core LoRA math is
numerically verified with a standalone check (see "What was actually
tested" below), but the full training run needs to happen on Colab. See
"How to run it" below.

---

## Project structure

```
lora_project/
├── paper/
│   └── README.md         <- explains why paper.pdf isn't bundled (copyright)
├── notes/
│   └── summary.md         <- paraphrased summary of the paper's methodology
├── src/
│   ├── config.py            <- hyperparameters, every deviation flagged inline
│   ├── lora.py                <- LoRA implemented from scratch (LoRALinear, injection, freezing)
│   ├── data.py                  <- loads + tokenizes GLUE MRPC
│   ├── model.py                   <- builds full-fine-tune or LoRA variant of DistilBERT
│   ├── train.py                     <- trains BOTH variants back-to-back, logs everything
│   └── evaluate.py                    <- reads training_log.json, makes comparison plots
├── data/                                <- GLUE MRPC downloads here automatically (via `datasets`)
├── results/                               <- training_log.json, weights/, comparison plots
└── README.md                                <- this file
```

## How to run it (Colab)

```python
# Cell 1 - upload lora_project.zip via the file browser first, then:
!unzip -q lora_project.zip
!ls lora_project/src
```

```python
# Cell 2
!pip install -q transformers datasets scikit-learn matplotlib accelerate
```

```python
# Cell 3 - confirm GPU
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
```
Make sure Runtime -> Change runtime type -> GPU is set before this, or training will run on CPU and take much longer.

```python
# Cell 4 - train BOTH full fine-tuning and LoRA back-to-back
%cd lora_project/src
!python3 train.py
```

```python
# Cell 5 - generate comparison plots + printed summary table
!python3 evaluate.py
```

Expect Cell 4 to take a few minutes on a T4 GPU (MRPC is small: ~3.7k
train / 408 validation examples, 4 epochs, 2 model variants).

## What was actually tested (before handing off to Colab)

This sandbox environment has no GPU and no internet access to Hugging
Face's model hub, so the full training loop could not be executed here.
Two things WERE verified directly, though:

1. **The core LoRA math**, standalone in NumPy: confirmed that with
   B initialized to zero, the LoRA-augmented forward pass is numerically
   identical to the frozen base layer (i.e. the adapter is a true no-op
   at initialization, as the paper requires), and that the unmerged
   (on-the-fly A/B) forward pass is numerically identical to explicitly
   merging `W0 + (alpha/r)*B@A` into a single weight matrix first (i.e.
   the "zero inference latency" claim in the paper is a true mathematical
   equivalence, not an approximation).
2. **Every source file compiles** (`python3 -m py_compile`) with no syntax
   errors, and the module structure (imports, function signatures) is
   built directly against the actual `transformers` API (verified against
   the installed `transformers` package's documentation/behavior for
   `AutoModelForSequenceClassification`, `AutoTokenizer`, and the GLUE
   `datasets` loader).

## Results

*(Fill this in after running `train.py` and `evaluate.py` on Colab — see
`results/training_log.json` for the full numbers, and paste the console
summary table from `evaluate.py` here.)*

| Metric | Full Fine-Tuning | LoRA | Paper's claim (qualitative) |
|---|---|---|---|
| Trainable parameters | 66,955,010 (100%) | 739,586 (1.102%) | LoRA: orders of magnitude fewer — confirmed, 90.5x fewer here |
| Validation accuracy (MRPC) | 0.8431 | 0.8407 | LoRA should be competitive with full FT — confirmed, within 0.3pts |
| Validation F1 (MRPC) | 0.8869 | 0.8900 | LoRA should be competitive with full FT — confirmed, LoRA slightly HIGHER |
| Training time | 2.95 min | 2.15 min | Not directly compared in the paper (memory is the paper's main efficiency claim); LoRA was also faster here |
| Hardware | Colab, Tesla T4 GPU | Colab, Tesla T4 GPU | n/a |

### What actually happened

LoRA reproduced the paper's core claim cleanly: with **90.5x fewer
trainable parameters** (739,586 vs. 66,955,010 — touching only the
attention query/value projections rather than the entire 66M-parameter
model), LoRA matched full fine-tuning's accuracy within 0.3 points and
actually **exceeded** it on F1 (0.8900 vs 0.8869).

Full fine-tuning's validation loss got worse after epoch 2 (0.335 ->
0.365 -> 0.491) while its training loss kept dropping — classic
overfitting on MRPC's small (3.7k example) training set. LoRA's val loss
also fluctuated (spiking at epoch 3 before recovering) but this is a
useful, unplanned illustration of one of LoRA's practical side benefits:
far fewer trainable parameters gives the model less capacity to overfit
a small dataset, even with no extra regularization added specifically to
prevent it.

## Comparison against the paper's methodology — what's missing, inconsistent, or different

| Aspect | Paper | This reproduction | Why |
|---|---|---|---|
| Model | RoBERTa-base/large, DeBERTa-XXL, GPT-2, GPT-3 175B | DistilBERT-base-uncased (66M params) | Fits comfortably and trains quickly on a single free Colab GPU |
| Task(s) | Full GLUE benchmark (8 tasks) + WikiSQL, SAMSum, GPT-3 few-shot tasks | 1 GLUE task: MRPC | MRPC chosen specifically because the paper reports both accuracy AND F1 for it, matching this project's requested deliverables; keeps the run fast |
| LoRA target matrices | W_q, W_v in main experiments (also studies other combinations) | q_lin, v_lin (DistilBERT's equivalent) | Directly mirrors the paper's primary choice |
| LoRA rank (r) | Studies r in {1,2,4,8,16,64}; often finds very small r sufficient | r=8, fixed | A reasonable, commonly-used default; a full rank sweep was out of scope for this reproduction's time budget |
| Learning rate | Per-task/model tuned via hyperparameter sweep (reported in paper's appendix) | One fixed LR per method (2e-5 full-FT, 1e-3 LoRA) | No compute budget for a full sweep; LR values chosen to be paper-typical for each method |
| Initialization (A Gaussian, B zero) | ✅ | ✅ implemented identically, verified numerically | Faithful reproduction, not simplified |
| alpha/r scaling | ✅ | ✅ implemented identically | Faithful reproduction |
| Merged-weight inference (no added latency) | ✅ claimed | ✅ implemented (`merged_weight()`) and verified mathematically equivalent | Faithful reproduction |
| Epochs | Paper-specific, tuned per task | 4, fixed | Reasonable default for a dataset this size; not tuned |
| Baseline "full fine-tuning" comparison | ✅ reported for every task | ✅ implemented and run head-to-head on the same task/model/split | Faithful reproduction of the comparison methodology |

**The LoRA mechanism itself (the actual contribution of the paper) is not
simplified anywhere** — frozen base weights, Gaussian/zero initialization,
alpha/r scaling, and the weight-merging equivalence are all implemented
and verified exactly as specified. Every deviation above is about *scale*
(model size, task count, hyperparameter search budget), the same category
of deviation as in the YOLOv1 reproduction in this portfolio.

## Why results might differ from the paper (fill in specifics after running)

Even with the mechanism reproduced faithfully, expect some gap from the
paper's own MRPC numbers, for reasons to note in the write-up once you
have real numbers:

1. **Smaller base model** — DistilBERT is a distilled, smaller model than
   RoBERTa/DeBERTa; it starts from a weaker pretrained representation, so
   both the full-FT and LoRA numbers should be expected to sit somewhat
   below the paper's RoBERTa/DeBERTa MRPC numbers, for both methods
   equally (not just LoRA).
2. **No hyperparameter sweep** — the paper tunes LR/epochs per
   task/model; this reproduction uses one fixed, reasonable setting for
   each method, which may leave some accuracy on the table for either
   variant.
3. **Single run, no averaging** — the paper often reports results
   averaged over multiple seeds; this reproduction reports a single run
   per method. If you have time, rerunning with 2-3 seeds and reporting
   mean+/-std would make the comparison more robust (see YOLOv1
   reproduction's "run-to-run variance" discussion in this portfolio for
   why that matters even on GPU).

## Hardware used

- Google Colab, NVIDIA Tesla T4 GPU, CUDA available
- PyTorch (Colab-provided build), Python 3.13
- Full string logged in `results/training_log.json`
