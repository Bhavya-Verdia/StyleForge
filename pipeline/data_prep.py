"""
data_prep.py — Dataset preparation for QLoRA fine-tuning on coding domain.

Dataset: iamtarun/python_code_instructions_18k_alpaca
Format:  Instruction-tuning JSONL with {instruction, input, output, text}
Usage:
    python data_prep.py --max_samples 5000 --output_dir ./data
    python data_prep.py --push_to_hub --hub_repo username/coding-qlora-data --hf_token YOUR_TOKEN
"""

import argparse
import json
import logging
import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from huggingface_hub import login
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Formatting with HF apply_chat_template
# ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Write clean, efficient, well-documented, and idiomatic Python code. "
    "Always include type hints and docstrings for functions."
)


def get_format_example(tokenizer):
    def format_example(example: dict) -> dict:
        """Map raw dataset row → instruction-tuning dict."""
        instruction = example.get("instruction", "").strip()
        inp = example.get("input", "") or ""
        output = example.get("output", "").strip()
        
        user_content = instruction
        if inp:
            user_content += f"\n\nInput:\n{inp}"
            
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]
        
        # This perfectly applies Mistral v0.3's native template!
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        
        return {
            "instruction": instruction,
            "input": inp,
            "output": output,
            "text": text,
        }
    return format_example


class InstructionData(BaseModel):
    instruction: str = Field(..., min_length=10)
    input: str = ""
    output: str = Field(..., min_length=30)
    text: str

def is_valid(example: dict, max_chars: int) -> bool:
    """Keep examples that have meaningful content and fit in context."""
    try:
        data = InstructionData(**example)
    except ValidationError:
        return False
        
    return (
        len(data.text) < max_chars
        and ("def " in data.output
             or "class " in data.output
             or "import " in data.output
             or "=" in data.output)
    )


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        # ── 1. Auth ──────────────────────────────────────────────
        if args.hf_token:
            login(token=args.hf_token)
            logger.info("HuggingFace login successful.")

        # ── 2. Load raw dataset ──────────────────────────────────
        logger.info("Loading dataset: %s", args.dataset)
        raw = load_dataset(args.dataset, split="train")
        logger.info("  Raw size: %s rows", f"{len(raw):,}")

        # ── 3. Sample ─────────────────────────────────────────────
        if args.max_samples and args.max_samples < len(raw):
            raw = raw.shuffle(seed=42).select(range(args.max_samples))
            logger.info("  Sampled to: %s rows", f"{len(raw):,}")

        # ── 3.5 Load Tokenizer for Chat Template ──────────────────
        logger.info("  Loading tokenizer: %s", args.model_id)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token)

        # ── 4. Format ─────────────────────────────────────────────
        logger.info("  Formatting into instruction-tuning format…")
        formatted = raw.map(
            get_format_example(tokenizer),
            remove_columns=raw.column_names,
            desc="Formatting",
        )

        # ── 5. Filter ─────────────────────────────────────────────
        max_chars = args.max_tokens * 4  # rough 4-chars-per-token heuristic
        logger.info("  Filtering (max ~%d tokens)…", args.max_tokens)
        filtered = formatted.filter(
            lambda x: is_valid(x, max_chars),
            desc="Filtering",
        )
        logger.info("  After filtering: %s rows", f"{len(filtered):,}")

        # ── 6. Train / eval split (95 / 5) ────────────────────────
        split = filtered.train_test_split(test_size=0.05, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]
        logger.info("  Train: %s  |  Eval: %s", f"{len(train_ds):,}", f"{len(eval_ds):,}")

        # ── 7. Save JSONL ─────────────────────────────────────────
        os.makedirs(args.output_dir, exist_ok=True)
        train_path = os.path.join(args.output_dir, "train.jsonl")
        eval_path = os.path.join(args.output_dir, "eval.jsonl")

        def save_jsonl(dataset, path: str) -> None:
            with open(path, "w", encoding="utf-8") as fh:
                for row in dataset:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info("  Saved → %s  (%s rows)", path, f"{len(dataset):,}")

        save_jsonl(train_ds, train_path)
        save_jsonl(eval_ds, eval_path)

        # ── 8. Push to Hub (optional) ─────────────────────────────
        if args.push_to_hub:
            if not args.hub_repo:
                raise ValueError("--hub_repo is required when --push_to_hub is set.")
            ds_dict = DatasetDict({"train": train_ds, "eval": eval_ds})
            ds_dict.push_to_hub(args.hub_repo, private=False)
            logger.info("Dataset pushed to HuggingFace Hub: https://huggingface.co/datasets/%s", args.hub_repo)

        logger.info("Data preparation complete!")

    except FileNotFoundError as exc:
        logger.error("Dataset or file not found: %s", exc)
        raise
    except OSError as exc:
        logger.error("Disk I/O error (check disk space / permissions): %s", exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error during data preparation: %s", exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare coding dataset for QLoRA fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str,
        default="iamtarun/python_code_instructions_18k_alpaca",
        help="HuggingFace dataset identifier",
    )
    parser.add_argument(
        "--model_id", type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace model ID (for loading tokenizer chat template)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=5000,
        help="Max rows to use from the dataset (None = all)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help="Rough max tokens per example (filtered by char count)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data",
        help="Directory to write train.jsonl and eval.jsonl",
    )
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument(
        "--hub_repo", type=str, default=None,
        help="HuggingFace Hub repo, e.g. username/coding-qlora-data",
    )
    parser.add_argument("--hf_token", type=str, default=None)
    main(parser.parse_args())
