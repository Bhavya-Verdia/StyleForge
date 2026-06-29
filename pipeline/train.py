"""
train.py — QLoRA fine-tuning of Mistral-7B-Instruct on coding dataset.

Requires A100 / T4 / V100 GPU (Google Colab or Kaggle).
Run after data_prep.py has produced ./data/train.jsonl and ./data/eval.jsonl.

Usage:
    python train.py
    python train.py --epochs 1 --batch_size 2 --max_seq_length 512   # quick test
    python train.py --push_to_hub --hub_repo username/mistral-coding-qlora
"""

import argparse
import csv
import logging
import os

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"


# ─────────────────────────────────────────────────────────────
# Loss Logger Callback
# ─────────────────────────────────────────────────────────────
class LossLoggerCallback(TrainerCallback):
    """Appends step-level training metrics to a CSV file in real time."""

    def __init__(self, log_path: str = "training_loss.csv") -> None:
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["step", "loss", "learning_rate", "epoch"])

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if not logs:
            return
        if "loss" not in logs:
            return
        with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                state.global_step,
                round(logs.get("loss", 0), 6),
                logs.get("learning_rate", ""),
                round(logs.get("epoch", 0), 4),
            ])


# ─────────────────────────────────────────────────────────────
# Model / Tokenizer Loading
# ─────────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_id: str, hf_token: str | None):
    """Load base model in 4-bit NF4 quantization (BitsAndBytes)."""
    logger.info("Loading model: %s", model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",            # Normal Float 4
        bnb_4bit_compute_dtype=torch.float16,  # compute in fp16
        bnb_4bit_use_double_quant=True,        # nested quantization
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=hf_token,
        torch_dtype=torch.float16,
        trust_remote_code=False,
    )
    # Scrub any bfloat16 parameters that might have slipped through from the base model config
    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
            
    # Required for gradient checkpointing + QLoRA
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # needed for fp16 training stability

    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# LoRA
# ─────────────────────────────────────────────────────────────
def apply_qlora(model):
    """Wrap model with QLoRA adapters."""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # All attention + MLP projections for maximum expressivity
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    
    # Force trainable params (adapters) to float32 for fp16 mixed-precision stability
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
            

    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    torch.manual_seed(42)

    # GPU diagnostics
    logger.info("CUDA available : %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU            : %s", torch.cuda.get_device_name(0))
        logger.info("VRAM           : %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    if HAS_WANDB:
        wandb.init(
            project="styleforge-pipeline",
            config={
                "model_id": args.model_id,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "max_seq_length": args.max_seq_length,
                "lora_r": 16,
                "lora_alpha": 32
            }
        )
    else:
        logger.warning("wandb is not installed or configured. W&B logging disabled.")

    # ── Load model & tokenizer ──────────────────────────────
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.hf_token)

    # ── Apply QLoRA ─────────────────────────────────────────
    logger.info("Applying QLoRA adapters…")
    model = apply_qlora(model)

    # ── Load dataset ─────────────────────────────────────────
    data_files = {
        "train": os.path.join(args.data_dir, "train.jsonl"),
        "eval":  os.path.join(args.data_dir, "eval.jsonl"),
    }
    logger.info("Loading dataset from: %s", args.data_dir)
    dataset = load_dataset("json", data_files=data_files)
    logger.info("Train: %s  Eval: %s", f"{len(dataset['train']):,}", f"{len(dataset['eval']):,}")

    # ── SFTConfig ────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,         # effective batch = batch_size * 4
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",             # memory-efficient optimizer for QLoRA
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="wandb" if HAS_WANDB else "none",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    # ── SFTTrainer ────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        args=training_args,
        callbacks=[LossLoggerCallback(log_path=args.loss_log_path)],
    )

    # ── Train ─────────────────────────────────────────────────
    logger.info("\nStarting training… This will take ~2-4 hrs on an A100 for 3 epochs.")
    try:
        trainer.train()
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        if "out of memory" in str(exc).lower() or isinstance(exc, torch.cuda.OutOfMemoryError):
            logger.error("CUDA OOM during training. Try reducing batch size.")
        raise

    # ── Save adapter weights only (not merged model) ──────────
    logger.info("\nSaving adapter weights → %s", args.adapter_dir)
    os.makedirs(args.adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)

    # ── Optional: push adapters to Hub ────────────────────────
    if args.push_to_hub:
        if not args.hub_repo:
            raise ValueError("--hub_repo is required when --push_to_hub is set.")
        logger.info("Pushing adapter to Hub: %s", args.hub_repo)
        trainer.model.push_to_hub(args.hub_repo, token=args.hf_token)
        tokenizer.push_to_hub(args.hub_repo, token=args.hf_token)

    logger.info("\nTraining complete!")
    logger.info("  Adapter weights : %s", args.adapter_dir)
    logger.info("  Loss log CSV    : %s", args.loss_log_path)
    
    if HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune Mistral-7B-Instruct on coding dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_id",        type=str,   default=DEFAULT_MODEL_ID)
    parser.add_argument("--data_dir",        type=str,   default="./data")
    parser.add_argument("--output_dir",      type=str,   default="./checkpoints")
    parser.add_argument("--adapter_dir",     type=str,   default="./adapter_weights")
    parser.add_argument("--epochs",          type=int,   default=3)
    parser.add_argument("--batch_size",      type=int,   default=4)
    parser.add_argument("--max_seq_length",  type=int,   default=1024)
    parser.add_argument("--loss_log_path",   type=str,   default="./training_loss.csv")
    parser.add_argument("--push_to_hub",     action="store_true")
    parser.add_argument("--hub_repo",        type=str,   default=None)
    parser.add_argument("--hf_token",        type=str,   default=None)
    main(parser.parse_args())
