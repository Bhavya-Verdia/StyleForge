"""
eval.py — Evaluate base model vs QLoRA fine-tuned model on held-out coding prompts.

Metrics:
    - ROUGE-1, ROUGE-2, ROUGE-L  (n-gram overlap with reference outputs)
    - Perplexity                  (NLL loss on 20 eval samples)
    - Average response length     (word count)

Output: eval_results.json

Usage:
    python eval.py --adapter_dir ./adapter_weights --output_path ./eval_results.json
"""

import argparse
import json
import logging
import math
import os

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import torch
from datasets import load_dataset
from peft import PeftModel
from rouge_score import rouge_scorer as rouge_scorer_module
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID  = "mistralai/Mistral-7B-Instruct-v0.3"
NUM_EVAL_SAMPLES  = 50
PERPLEXITY_SUBSET = 20   # perplexity computation on a smaller subset (GPU memory)


# ─────────────────────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────────────────────

def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_base(model_id: str, hf_token: str | None):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=_bnb_config(),
            device_map={"": 0},
            token=hf_token,
            torch_dtype=torch.float16,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.error(
                "CUDA out of memory while loading model. "
                "Try freeing GPU memory or using a smaller model."
            )
        raise
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_finetuned(model_id: str, adapter_dir: str, hf_token: str | None):
    base_model, tokenizer = load_base(model_id, hf_token)
    # Merge adapter into base weights for faster eval inference
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    system_prompt = (
        "You are an expert Python programmer. "
        "Write clean, efficient, well-documented, and idiomatic Python code. "
        "Always include type hints and docstrings for functions."
    )
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.error(
                "CUDA out of memory during generation. "
                "Try reducing max_new_tokens or freeing GPU memory."
            )
        raise
    new_ids = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────

def compute_rouge(preds: list[str], refs: list[str]) -> dict[str, float]:
    scorer = rouge_scorer_module.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    totals: dict[str, float] = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)
        for k in totals:
            totals[k] += scores[k].fmeasure
    n = len(preds)
    return {k: round(v / n, 4) for k, v in totals.items()}


def compute_bleu(preds: list[str], refs: list[str]) -> float:
    try:
        import sacrebleu
        # For BLEU, sacrebleu expects refs as a list of lists of strings
        bleu = sacrebleu.corpus_bleu(preds, [refs])
        return round(bleu.score, 4)
    except ImportError:
        return 0.0


def compute_perplexity(model, tokenizer, texts: list[str]) -> float:
    """Average perplexity over provided texts via NLL loss."""
    model.eval()
    total_nll = 0.0
    count = 0
    for text in texts:
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
            total_nll += out.loss.item()
            count += 1
    return round(math.exp(total_nll / max(count, 1)), 2)


# ─────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, samples: list[dict], label: str) -> dict:
    logger.info("── Evaluating: %s ──────────────────────────────", label)
    preds, refs = [], []

    for item in tqdm(samples, desc=f"  Generating [{label}]"):
        user_content = item["instruction"]
        if item.get("input"):
            user_content += f"\n\nInput:\n{item['input']}"
            
        pred = generate(model, tokenizer, user_content)
        preds.append(pred)
        refs.append(item["output"])

    rouge = compute_rouge(preds, refs)
    bleu  = compute_bleu(preds, refs)
    ppl   = compute_perplexity(
        model, tokenizer,
        [s["text"] for s in samples[:PERPLEXITY_SUBSET]]
    )
    avg_len = round(sum(len(p.split()) for p in preds) / len(preds), 1)

    result = {
        "model":               label,
        "rouge1":              rouge["rouge1"],
        "rouge2":              rouge["rouge2"],
        "rougeL":              rouge["rougeL"],
        "bleu":                bleu,
        "perplexity":          ppl,
        "avg_response_length": avg_len,
        "num_samples":         len(samples),
    }
    logger.info(
        "  ROUGE-1: %.4f  ROUGE-2: %.4f  ROUGE-L: %.4f  BLEU: %.4f",
        rouge["rouge1"], rouge["rouge2"], rouge["rougeL"], bleu,
    )
    logger.info("  Perplexity: %s  Avg length: %s words", ppl, avg_len)
    return result


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("CUDA available: %s", torch.cuda.is_available())

    # ── Load eval split ───────────────────────────────────────
    eval_path = os.path.join(args.data_dir, "eval.jsonl")
    ds = load_dataset("json", data_files={"eval": eval_path})["eval"]
    n  = min(NUM_EVAL_SAMPLES, len(ds))
    samples = [ds[i] for i in range(n)]
    logger.info("Evaluating on %d held-out samples", n)

    results = []

    # ── Base model ────────────────────────────────────────────
    logger.info("Loading base model…")
    base_model, base_tok = load_base(args.model_id, args.hf_token)
    results.append(evaluate(base_model, base_tok, samples, "base"))

    del base_model, base_tok
    torch.cuda.empty_cache()
    logger.info("CUDA memory cleared after base model.")

    # ── Fine-tuned model ──────────────────────────────────────
    if not os.path.exists(args.adapter_dir):
        logger.warning("Adapter directory not found: %s — skipping fine-tuned eval.", args.adapter_dir)
    else:
        logger.info("Loading fine-tuned model…")
        ft_model, ft_tok = load_finetuned(args.model_id, args.adapter_dir, args.hf_token)
        results.append(evaluate(ft_model, ft_tok, samples, "finetuned"))

        del ft_model, ft_tok
        torch.cuda.empty_cache()
        logger.info("CUDA memory cleared after fine-tuned model.")

    # ── Save results ──────────────────────────────────────────
    output = {"results": results, "eval_samples": n}
    with open(args.output_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    logger.info("Results saved → %s", args.output_path)
    logger.info(json.dumps(output, indent=2))
    
    if HAS_WANDB:
        wandb.init(project="styleforge-eval", name="eval-run")
        for res in results:
            prefix = res["model"]
            wandb.log({f"{prefix}/{k}": v for k, v in res.items() if isinstance(v, (int, float))})
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate base vs fine-tuned model on coding prompts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_id",    type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--data_dir",    type=str, default="./data")
    parser.add_argument("--adapter_dir", type=str, default="./adapter_weights")
    parser.add_argument("--output_path", type=str, default="./eval_results.json")
    parser.add_argument("--hf_token",    type=str, default=None)
    main(parser.parse_args())
