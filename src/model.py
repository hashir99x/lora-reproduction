"""
Builds the classification model in either mode:
  - "full": standard full fine-tuning, every parameter trainable
  - "lora": backbone frozen, LoRA adapters injected into attention
            query/value projections, only adapters + classifier head trainable
"""

from transformers import AutoModelForSequenceClassification

import config
import lora as lora_mod


def build_model(mode: str):
    assert mode in ("full", "lora")

    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=config.NUM_LABELS
    )

    if mode == "full":
        trainable = sum(p.numel() for p in model.parameters())
        total = trainable
        return model, trainable, total

    # mode == "lora"
    n_replaced = lora_mod.inject_lora(
        model,
        target_module_names=config.LORA_TARGET_MODULES,
        r=config.LORA_R,
        alpha=config.LORA_ALPHA,
        dropout=config.LORA_DROPOUT,
    )
    print(f"Injected LoRA into {n_replaced} linear layers "
          f"(target modules: {config.LORA_TARGET_MODULES})")

    trainable, total = lora_mod.freeze_base_model_except_lora_and_head(model)
    return model, trainable, total
