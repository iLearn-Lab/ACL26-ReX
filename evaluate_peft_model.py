# -*- coding: utf-8 -*-
"""
A comprehensive script for instruction finetuning of causal language models
using PEFT (LoRA and custom HypraLoRA) and Hugging Face Transformers.

This script is designed to be modular and configurable, covering the entire
pipeline from data preparation to model training and evaluation.
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys
import argparse
import json
import logging
import torch
from pathlib import Path
from typing import Union
from evaluation import evaluate_single_dataset, load_model_for_evaluation
from custom_lora import *
from datetime import datetime

# 在文件顶部添加模块级别的logger
logger = logging.getLogger(__name__)

# 常量
from constant import *

def setup_logging(output_dir: Union[str, Path]):
    """Configures logging to output to both stdout and a file."""
    log_level = logging.INFO
    output_dir = Path(output_dir)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(output_dir / "evaluation.log", mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def parse_args():
    """
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a Causal LM with LoRA."
    )
    parser.add_argument(
        '--model_save_path', type=str,
        help='The model save path.'
    )
    parser.add_argument('--max_new_tokens', type=int, default=2048,help='The max new tokens.')
    parser.add_argument("--eval_dataset_names", type=str, default="mmlupro,gpqa,mmlupro_chemistry,mmlupro_engineering,mmlupro_physics,math500_level3", help="Name(s) of the dataset(s) to use for evaluation. Multiple datasets can be separated by commas (e.g., 'mmlupro,supergpqa,math').")

    args = parser.parse_args()
    ft_config_path = os.path.join(args.model_save_path, "finetune_config.json")
    ft_config = json.load(open(ft_config_path))

    ft_config['gen_args']['max_new_tokens'] = args.max_new_tokens
    if args.eval_dataset_names == "":
        eval_dataset_names = [item.strip() for item in ft_config['data_args']['test_name'].split(',')]
    else:
        eval_dataset_names = [item.strip() for item in args.eval_dataset_names.split(',')]
    print(eval_dataset_names)
    return ft_config, eval_dataset_names


# --- Main Execution ---

def main():
    """Main function to run the finetuning pipeline."""
    # 1. Configuration
    ft_config, eval_dataset_names = parse_args()

    batch_size = 16
    
    output_dir = ft_config['training_args']['output_dir']
    setup_logging(output_dir)

    # 8. Evaluate
    if len(eval_dataset_names) > 0:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        device_map = {"": "cuda:0"}
        eval_dataset_dir = EVAL_DATASET_PATH
        all_results = {}
        summary_metrics = {}

        model, tokenizer = load_model_for_evaluation(ft_config, device_map)
        for dataset_name in eval_dataset_names:
            total_samples = EVAL_DATASET_SIZE[dataset_name]
            try:
                predictions, metrics = evaluate_single_dataset(
                    model, tokenizer, output_dir, dataset_name, eval_dataset_dir,
                    batch_size=batch_size,
                    category_samples=None,
                    total_samples=total_samples,
                )
                all_results[dataset_name] = {
                    "predictions": predictions,
                    "metrics": metrics
                }
                summary_metrics[dataset_name] = metrics
            except Exception as e:
                summary_metrics[dataset_name] = {
                    "error": str(e),
                    "dataset_name": dataset_name
                }

        # Calculate overall summary statistics
        total_samples = sum(metrics.get("total_samples", 0) for metrics in summary_metrics.values() if "error" not in metrics)
        total_correct = sum(metrics.get("correct_samples", 0) for metrics in summary_metrics.values() if "error" not in metrics)
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # Add overall summary to metrics
        summary_metrics["overall_summary"] = {
            "total_datasets": len(eval_dataset_names),
            "successful_datasets": len([m for m in summary_metrics.values() if "error" not in m]),
            "failed_datasets": len([m for m in summary_metrics.values() if "error" in m]),
            "total_samples": total_samples,
            "total_correct": total_correct,
            "overall_accuracy": overall_accuracy
        }

        # Save summary metrics for all datasets
        summary_file = os.path.join(ft_config['training_args']['output_dir'], f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_evaluation_summary.json")
        with open(summary_file, "w", encoding='utf-8') as f:
            json.dump(summary_metrics, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()

