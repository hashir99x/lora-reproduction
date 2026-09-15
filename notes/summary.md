# Paper Summary (in my own words)

**Paper:** Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen — "LoRA:
Low-Rank Adaptation of Large Language Models" (arXiv:2106.09685, 2021)

> A copy of the paper is **not** bundled in `paper/` for copyright reasons.
> Download it from https://arxiv.org/abs/2106.09685 and place it at
> `paper/paper.pdf` if you want it alongside this project.

## The problem LoRA solves

Fine-tuning a large pretrained language model on a new task normally means
updating every one of its parameters. For a model the size of GPT-3
(175 billion parameters), that means storing and optimizing a full
175-billion-parameter copy for every single task you want to adapt it to
— expensive to train, and expensive to store/switch between if you have
many downstream tasks. Earlier attempts to make fine-tuning cheaper
(adapter layers, prompt tuning, prefix tuning) either add extra inference
latency or are hard to optimize well, and often fall short of full
fine-tuning's accuracy.

## The core idea

The authors hypothesize that the *change* a pretrained model's weights
undergo during fine-tuning has a low "intrinsic rank" — i.e. even though
the weight matrices themselves are huge, the useful update to them during
adaptation can be well-approximated by a much smaller, low-rank matrix.

Concretely, for a frozen pretrained weight matrix W0, instead of learning
a full update ΔW of the same size, LoRA factors it as the product of two
much smaller matrices:

```
W = W0 + ΔW = W0 + (alpha/r) * B @ A
```

where `A` is `r × d_in` and `B` is `d_out × r`, with rank `r` chosen to
be tiny (often single digits) relative to the original dimensions. Only
`A` and `B` are trained; `W0` stays frozen throughout. This cuts the
number of trainable parameters for that layer from `d_in × d_out` down to
`r × (d_in + d_out)` — for GPT-3's 175B parameters, the paper reports
this brings trainable parameters down to about 4.7 million (roughly
10,000x fewer) while matching or approaching full fine-tuning quality.

## Initialization matters

`A` is initialized from a random Gaussian distribution, and `B` is
initialized to all zeros. This means the product `B @ A` — and therefore
the entire adaptation term — is exactly zero at the very start of
training. Training begins from the pretrained model's original behavior
unchanged, and the adaptation is learned gradually from there. This
detail is small but important, and is reproduced exactly in this
project's `lora.py`.

## Where LoRA is applied

The paper applies LoRA to the weight matrices inside the Transformer's
self-attention blocks — specifically the query and value projection
matrices (W_q and W_v) in their main experiments — while leaving the
feed-forward layers unmodified. They find adapting just W_q and W_v
already captures most of the benefit, and that spreading a fixed
parameter budget across more matrices at a smaller rank each tends to
work about as well as concentrating it on fewer matrices at a higher rank.

## Why it doesn't cost anything at inference

Because the adaptation is just an additive term to the original weight
matrix, LoRA's low-rank matrices can be explicitly multiplied out and
added into the frozen weight (`W = W0 + (alpha/r)*B@A`) once training is
done, producing a single ordinary weight matrix of the original size.
That means, unlike some other efficient-adaptation methods, a LoRA-tuned
model runs at exactly the same speed as the original at inference time —
there's no extra adapter computation left in the forward pass. This
project's `lora.py` includes a `merged_weight()` method demonstrating this
equivalence.

## Headline results

- On GPT-3 175B, LoRA matches or exceeds full fine-tuning accuracy across
  several benchmarks while training roughly 10,000x fewer parameters and
  requiring about 3x less GPU memory during training.
- On the GLUE benchmark with RoBERTa and DeBERTa, LoRA is competitive
  with, and on some tasks slightly better than, full fine-tuning, again
  with a small fraction of the trainable parameters.
- The paper also empirically studies how small `r` (the rank) can be
  while still working well, finding that surprisingly small ranks
  (even r=1 or r=2 in some settings) already capture most of the benefit
  — evidence supporting their low-intrinsic-rank hypothesis.

## Why this matters for the reproduction

This project's `lora.py` reimplements the exact mechanism described
above (frozen W0, Gaussian-init A, zero-init B, alpha/r scaling) from
scratch in PyTorch, rather than using an existing PEFT library, and
applies it to a real pretrained transformer's attention projections. The
experiment design mirrors the paper's own core comparison: train the same
base model two ways (full fine-tuning vs. LoRA) on the same task, and
compare accuracy, F1, and trainable parameter count directly. See
`README.md` for what had to be scaled down to run this on a single free
Colab GPU, and for the actual results.
