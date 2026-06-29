"""
convert_peft.py — Convert PEFT adapter weights to MLX format.

Usage:
    python convert_peft.py --adapter_dir ./adapter_weights
"""

import argparse
import json
import logging
import sys

from safetensors import safe_open
from safetensors.torch import save_file
import torch

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PEFT adapter weights to MLX-compatible format."
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Path to directory containing PEFT adapter files.",
    )
    args = parser.parse_args()

    adapter_dir = args.adapter_dir
    safetensors_path = f"{adapter_dir}/adapter_model.safetensors"
    out_safetensors_path = f"{adapter_dir}/adapters.safetensors"
    config_path = f"{adapter_dir}/adapter_config.json"

    # 1. Load the original PEFT safetensors
    logger.info("Loading PEFT safetensors from %s...", safetensors_path)
    try:
        state_dict = {}
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensor = f.get_tensor(k)

                # Convert PEFT key to MLX key
                # base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
                # -> model.layers.0.self_attn.q_proj.lora_a.weight
                new_k = k
                if new_k.startswith("base_model.model."):
                    new_k = new_k.replace("base_model.model.", "")

                # lora_A -> lora_a, lora_B -> lora_b
                new_k = new_k.replace("lora_A", "lora_a").replace("lora_B", "lora_b")

                state_dict[new_k] = tensor
    except FileNotFoundError:
        logger.error("Safetensors file not found: %s", safetensors_path)
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to load safetensors: %s", e, exc_info=True)
        sys.exit(1)

    # 2. Save the new safetensors
    logger.info("Saving MLX adapters.safetensors to %s...", out_safetensors_path)
    try:
        save_file(state_dict, out_safetensors_path)
    except Exception as e:
        logger.error("Failed to save safetensors: %s", e, exc_info=True)
        sys.exit(1)

    # 3. Update the adapter_config.json for MLX
    logger.info("Updating adapter_config.json...")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Add the MLX required fields
        config["fine_tune_type"] = "lora"
        config["num_layers"] = 32
        config["lora_parameters"] = {
            "rank": config.get("r", 16),
            "scale": float(config.get("lora_alpha", 32)),
            "dropout": config.get("lora_dropout", 0.05),
            "keys": config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    except FileNotFoundError:
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to update config: %s", e, exc_info=True)
        sys.exit(1)

    logger.info("Done! Ready for MLX.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
