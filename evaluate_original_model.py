# -*- coding: utf-8 -*-
"""
Evaluate the original model on the evaluation datasets.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
import json

from evaluation import evaluate_single_dataset
from utils_funcs import _get_attn_implementation

# 常量
from constant import *


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate the original model on the evaluation datasets.")
    parser.add_argument('--model_name', type=str, default='qwen2.5-3b',help='The model name (e.g., "qwen2.5-7b"). Must be a key in MODELHFID_MAP.')
    parser.add_argument('--device', type=str, default="cuda:0",help='The device map for the model.')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='The maximum number of new tokens to generate.')
    parser.add_argument("--eval_dataset_names", type=str, default="mmlupro,gpqa,mmlupro_chemistry,mmlupro_engineering,mmlupro_physics,math500_level3", help="Name(s) of the dataset(s) to use for evaluation. Multiple datasets can be separated by commas.")
    parser.add_argument("--is_quantized", type=int, default=1, choices=[0, 1], help="Whether the model is quantized. 0: not quantized, 1: quantized.")
    parser.add_argument("--total_samples", type=int, default=500, help="Total number of samples to evaluate.")
    parser.add_argument("--category_samples", type=int, default=0, help="Number of samples for each category.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    args = parser.parse_args()

    total_samples = None
    if args.category_samples == 0:
        category_samples = None
    else:
        category_samples = args.category_samples
    
    eval_dataset_names = [item.strip() for item in args.eval_dataset_names.split(",")]

    return args, eval_dataset_names, total_samples, category_samples

def load_model(args):
    # model_id = MODELHFID_MAP[args.model_name]
    model_id = os.path.join(HF_CACHE_DIR, MODELHFID_MAP[args.model_name])
    if args.is_quantized == 1:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    attn_implementation = _get_attn_implementation(torch.bfloat16)
    device_map = {"": args.device.strip()}
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left", device_map=device_map)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Setting pad_token to eos_token for evaluation tokenizer.")
    model_kwargs = {
        "pretrained_model_name_or_path": model_id,
        "trust_remote_code": True,
        "device_map": device_map,
        "torch_dtype": torch.bfloat16,
    }

    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.eval()
    model.generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return model, tokenizer


def main():
    """Main function to run the finetuning pipeline."""
    args, eval_dataset_names, total_samples, category_samples = parse_args()

    model, tokenizer = load_model(args)

    if args.is_quantized == 1:
        quant_prefix = "nf4"
    else:
        quant_prefix = "bf16"

    output_dir = os.path.join(RESULT_PATH, f"output_original_model/{quant_prefix}_{args.model_name}")
    os.makedirs(output_dir, exist_ok=True)
    eval_dataset_dir = EVAL_DATASET_PATH

    all_results = {}
    summary_metrics = {}

    for eval_dataset_name in eval_dataset_names:
        try:
            # if eval_dataset_name == "math500":
            #     total_samples = 500
            # else:
            #     total_samples = 200
            total_samples = EVAL_DATASET_SIZE[eval_dataset_name]
            predictions, metrics = evaluate_single_dataset(
                model, tokenizer, output_dir, eval_dataset_name, eval_dataset_dir,
                batch_size=args.batch_size,
                category_samples=category_samples,
                total_samples=total_samples,
            )
            all_results[eval_dataset_name] = {
                "predictions": predictions,
                "metrics": metrics
            }
            summary_metrics[eval_dataset_name] = metrics
            print(f"Evaluation for {eval_dataset_name} completed. Metrics: {metrics}")
        except Exception as e:
            summary_metrics[eval_dataset_name] = {
                "error": str(e),
                "dataset_name": eval_dataset_name
            }
            print(f"Evaluation for {eval_dataset_name} failed. Error: {e}")

    # Calculate overall summary statistics
    total_samples = sum(metrics.get("total_samples", 0) for metrics in summary_metrics.values() if "error" not in metrics)
    total_correct = sum(metrics.get("correct_samples", 0) for metrics in summary_metrics.values() if "error" not in metrics)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    summary_metrics["overall_summary"] = {
        "total_datasets": len(eval_dataset_names),
        "successful_datasets": len([m for m in summary_metrics.values() if "error" not in m]),
        "failed_datasets": len([m for m in summary_metrics.values() if "error" in m]),
        "total_samples": total_samples,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy
    }

    # Save all results
    summary_file = os.path.join(output_dir, f"t{total_samples}_c{category_samples}_evaluation_summary.json")
    with open(summary_file, "w", encoding='utf-8') as f:
        json.dump(summary_metrics, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()