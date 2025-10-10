# -*- coding: utf-8 -*-
"""
A comprehensive script for instruction finetuning of causal language models
using PEFT (LoRA and custom HypraLoRA) and Hugging Face Transformers.

This script is designed to be modular and configurable, covering the entire
pipeline from data preparation to model training and evaluation.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import random
import numpy as np

import torch
import transformers
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    GenerationConfig,
)

from custom_lora import VAELoRAConfig

from peft import LoraConfig, TaskType, get_peft_model
from peft import prepare_model_for_kbit_training

from config_schemas import (
    DataArguments,
    GenerationArguments,
    LoRAArguments,
    VAELoRAArguments,
    QuantizationArguments,
    FinetuneConfig,
)
from data_processor import prepare_datasets, DataCollatorForInstructionTuning
from utils_funcs import parse_unknown_args, apply_overrides, save_finetune_config
from utils_funcs import _get_attn_implementation

# 常量
from constant import *
from evaluation import evaluate_single_dataset, load_model_for_evaluation

# 在文件顶部添加模块级别的logger
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """
    Set random seeds for reproducible training.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=False)
    
    # Set environment variable for PYTHONHASHSEED
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to {seed} for reproducible training")

def get_device() -> str:
    """
    Get the device to use for finetuning.
    """
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            return "auto"  # 改为返回具体设备，而不是 "auto"
        else:
            return "cuda:0"
    else:
        raise RuntimeError("No GPU available")


def parse_args() -> FinetuneConfig:
    """
    Parses command-line arguments and a JSON config file to create a unified
    configuration object for finetuning.
    """
    # 1. Basic CLI parser for initial arguments
    parser = argparse.ArgumentParser(
        description="Finetune a Causal LM with LoRA or HypraLoRA."
    )
    parser.add_argument(
        '--model_name', type=str, default='llama3.2-3b-instruct',
        help='The model name (e.g., "llama3.2-3b-instruct"). Must be a key in MODELHFID_MAP.'
    )
    parser.add_argument(
        '--adapter_name', type=str, default='default', choices=['lora', 'vaelora'],
        help='The adapter type to use for finetuning.'
    )
    parser.add_argument(
        '--data_config', type=str, default='default',
        help='Data configuration name (without .json extension). File will be loaded from configs/data/{name}.json'
    )
    parser.add_argument(
        '--adapter_config', type=str, default='default',
        help='Adapter configuration name (without .json extension). File will be loaded from configs/adapter/{name}.json'
    )
    parser.add_argument(
        '--quant_config', type=str, default='default',
        help='Quantization configuration name (without .json extension). File will be loaded from configs/quant/{name}.json'
    )
    parser.add_argument(
        '--training_config', type=str, default='default',
        help='Training configuration name (without .json extension). File will be loaded from configs/training/{name}.json'
    )
    parser.add_argument(
        '--generation_config', type=str, default='default',
        help='Generation configuration name (without .json extension). File will be loaded from configs/generation/{name}.json'
    )
    parser.add_argument(
        '--target_modules', type=int, default=4,
        help='Target modules to use for finetuning. 0: q_proj, k_proj, v_proj, o_proj; 1: gate_proj, up_proj, down_proj; 2: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj'
    )
    args, unknown = parser.parse_known_args()

    # 解析未知参数
    override_args = parse_unknown_args(unknown)
    print(f"Parsed unknown arguments: {override_args}")

    # 2. Select the appropriate adapter arguments class
    if args.adapter_name == 'lora':
        AdapterArguments = LoRAArguments
    elif args.adapter_name == 'vaelora':
        AdapterArguments = VAELoRAArguments
    else:
        raise ValueError(f"Unknown adapter_name: {args.adapter_name}")

    # 3. Load and merge JSON configuration files into a single dictionary
    merged_config = {}
    
    # Define configuration file paths based on the provided names
    config_files = {
        'data': f"configs/data/{args.data_config}.json",
        'adapter': f"configs/adapter/{args.adapter_config}.json", 
        'quant': f"configs/quant/{args.quant_config}.json",
        'training': f"configs/training/{args.training_config}.json",
        'generation': f"configs/generation/{args.generation_config}.json"
    }
    
    # Load each configuration file
    for config_type, config_file in config_files.items():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                merged_config.update(config_data)
                print(f"Loading {config_type} config: {os.path.abspath(config_file)}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file {config_file}: {e}")
            raise

    # 4. Use HfArgumentParser to parse the merged dictionary into dataclasses
    parser_classes = (DataArguments, AdapterArguments, QuantizationArguments, TrainingArguments, GenerationArguments)
    hf_parser = HfArgumentParser(parser_classes)
    data_args, adapter_args, quant_args, training_args, gen_args = hf_parser.parse_dict(args=merged_config)

    # 应用覆盖到各个配置对象
    apply_overrides(training_args, override_args, "training_args")
    apply_overrides(adapter_args, override_args, "adapter_args")
    apply_overrides(data_args, override_args, "data_args")
    apply_overrides(quant_args, override_args, "quant_args")
    apply_overrides(gen_args, override_args, "gen_args")
    
    # 4. Post-process and augment arguments
    model_hfid = os.path.join(HF_CACHE_DIR, MODELHFID_MAP[args.model_name])
    if adapter_args.target_modules is None:
        if args.target_modules == 0:
            adapter_args.target_modules = MODEL_TARGETMODULES_MAP2[args.model_name]
        elif args.target_modules == 1:
            adapter_args.target_modules = MODEL_TARGETMODULES_MAP[args.model_name]
        elif args.target_modules == 2:
            adapter_args.target_modules = MODEL_TARGETMODULES_MAP3[args.model_name]
        elif args.target_modules == 4:
            adapter_args.target_modules = MODEL_TARGETMODULES_MAP4[args.model_name]
        else:
            raise ValueError(f"Unknown target_modules: {args.target_modules}")

    if quant_args.load_in_4bit:
        quant_name = "nf4"
    elif quant_args.load_in_8bit:
        quant_name = "int8"
    else:
        quant_name = "bf16"

    # Create a structured output directory based on the config path and model params
    output_path = Path(os.path.join(RESULT_PATH, training_args.output_dir))
    timestamp = os.environ.get("PARALLEL_TIMESTAMP", datetime.now().strftime('%Y%m%d_%H%M%S'))
    output_dir = output_path / f"{quant_name}_{args.model_name}" / f"{data_args.name}" / f"{args.adapter_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    training_args.output_dir = str(output_dir)
    training_args.run_name = f"{args.model_name}_{args.adapter_name}"
    training_args.logging_dir = str(output_dir)

    if abs(data_args.val_split - 0.0) < 0.00001:
        training_args.eval_strategy = "no"

    if quant_args.bnb_4bit_compute_dtype_str == "bfloat16":
        training_args.bf16 = True
    elif quant_args.bnb_4bit_compute_dtype_str == "float16":
        training_args.fp16 = True
    else:
        raise RuntimeError(f"Unsupported bnb_4bit_compute_dtype_str: {quant_args.bnb_4bit_compute_dtype_str}")

    data_args.data_cache_dir = PROJECT_DATA_PATH

    device = get_device()
    logger.info(f"Using device: {device}")

    eval_dataset_names = []
    if len(data_args.test_name) > 0:
        eval_dataset_names = [name.strip() for name in data_args.test_name.split(',')]
    else:
        eval_dataset_names = []

    # 5. Assemble and return the final configuration object
    ft_config = FinetuneConfig(
        model_name=args.model_name,
        model_hfid=model_hfid,
        adapter_name=args.adapter_name,
        config_path=str(output_dir),
        device=device,
        data_args=data_args,
        adapter_args=adapter_args,
        quant_args=quant_args,
        training_args=training_args,
        gen_args=gen_args,
    )

    return ft_config, eval_dataset_names


# --- Logging Setup ---
def setup_logging(ft_config: FinetuneConfig):
    """Configures logging to output to both stdout and a file."""
    log_level = logging.INFO
    output_dir = Path(ft_config.training_args.output_dir)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(output_dir / "finetune.log", mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set transformers verbosity
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.setLevel(log_level)

    logger.info(f"Logging configured at level {logging.getLevelName(log_level)}")
    logger.info(f"Log file will be saved to: {output_dir / 'finetune.log'}")

# --- Model & Tokenizer Preparation ---
def prepare_model_and_tokenizer(ft_config: FinetuneConfig):
    """
    Loads the tokenizer and model, applying quantization and PEFT configuration.
    This function contains the logic to dynamically handle different device mapping
    strategies based on whether quantization is enabled, ensuring compatibility

    with both single-GPU and multi-GPU (DDP) training.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ft_config.model_hfid,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token to eos_token for tokenizer.")
    
    # Quantization configuration
    quant_config = ft_config.quant_args
    if quant_config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, quant_config.bnb_4bit_compute_dtype_str),
            bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
        )
        logger.info(f"4-bit quantization enabled with config: {bnb_config}")
    elif quant_config.load_in_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=quant_config.llm_int8_threshold,
            bnb_4bit_compute_dtype=getattr(torch, quant_config.bnb_4bit_compute_dtype_str),
            llm_int8_enable_fp32_cpu_offload=quant_config.llm_int8_enable_fp32_cpu_offload,
        )
        logger.info(f"8-bit quantization enabled with config: {bnb_config}")
    else:
        bnb_config = None
    
    logger.info(f"bnb_config: {bnb_config}")
    if ft_config.device == "auto":
        device_map = "auto"  # 直接传递字符串，让Transformers自动处理
    else:
        device_map = {'': ft_config.device}  # 具体设备使用字典形式
    attn_implementation = _get_attn_implementation(getattr(torch, quant_config.bnb_4bit_compute_dtype_str))

    model_kwargs = {
        "pretrained_model_name_or_path": ft_config.model_hfid,
        # "cache_dir": HF_CACHE_DIR,
        "trust_remote_code": True,
        "torch_dtype": getattr(torch, quant_config.bnb_4bit_compute_dtype_str),
        "use_cache": False,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    
    if ft_config.adapter_name == 'vaelora':
        from custom_lora.vaelora import LlamaForCausalLM
        from custom_lora.vaelora.modeling_mistral import MistralForCausalLM
        if "llama" in ft_config.model_name:
            model = LlamaForCausalLM.from_pretrained(**model_kwargs)
        elif "mistral" in ft_config.model_name:
            model = MistralForCausalLM.from_pretrained(**model_kwargs)
        else:
            raise ValueError(f"Unsupported model name: {ft_config.model_name}")
    else:
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    if ft_config.training_args.gradient_checkpointing and quant_config.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=ft_config.training_args.gradient_checkpointing)

    # PEFT configuration
    embedder = None
    adapter_args = ft_config.adapter_args
    if ft_config.adapter_name == 'lora':
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=adapter_args.lora_rank,
            lora_alpha=adapter_args.lora_alpha,
            lora_dropout=adapter_args.lora_dropout,
            target_modules=adapter_args.target_modules,
            bias="none"
        )
    elif ft_config.adapter_name == 'vaelora':
        peft_config = VAELoRAConfig(
            task_type=TaskType.CAUSAL_LM,
            r=adapter_args.r,
            r_b=adapter_args.r_b,
            d_z=adapter_args.d_z,
            topk=adapter_args.topk,
            lora_alpha=adapter_args.lora_alpha,
            enc_hidden=adapter_args.enc_hidden,
            head_hidden=adapter_args.head_hidden,
            mixer_weight_init=adapter_args.mixer_weight_init,
            free_bits=adapter_args.free_bits,
            use_bf16_compute=adapter_args.use_bf16_compute,
            target_modules=adapter_args.target_modules,
            exclude_modules=adapter_args.exclude_modules,
            fan_in_fan_out=adapter_args.fan_in_fan_out,
            bias=adapter_args.bias,
            modules_to_save=adapter_args.modules_to_save
        )
    else:
        raise ValueError(f"Unsupported adapter type: {ft_config.adapter_name}")
    
    model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
    logger.info(f"Applied {ft_config.adapter_name} configuration.")
    model.print_trainable_parameters()
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}")

    # Generation configuration
    gen_config = ft_config.gen_args
    model.generation_config = GenerationConfig(
        max_new_tokens=gen_config.max_new_tokens,
        do_sample=gen_config.do_sample,
        temperature=gen_config.temperature,
        top_p=gen_config.top_p,
        top_k=gen_config.top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    return model, tokenizer, embedder


# --- Main Execution ---

def main():
    """Main function to run the finetuning pipeline."""
    # 1. Configuration
    ft_config, eval_dataset_names = parse_args()
    
    # 2. Set random seed for reproducibility
    set_seed(ft_config.training_args.seed)
    
    # 3. Setup Logging
    setup_logging(ft_config)
    logger.info(f"Starting finetuning for model: {ft_config.model_name} with adapter: {ft_config.adapter_name}")
    logger.info(f"Output directory: {ft_config.training_args.output_dir}")
    logger.info(f"Finetuning configuration:\n{ft_config}")

    # 4. Save Configuration to JSON
    save_finetune_config(ft_config)

    # 5. Prepare Model and Tokenizer
    model, tokenizer, embedder = prepare_model_and_tokenizer(ft_config)

    # 6. Prepare Datasets
    # We get back the tokenized train/eval sets and the raw eval set for metrics
    train_dataset, eval_dataset = prepare_datasets(ft_config, tokenizer, embedder)
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset is not None:
        logger.info(f"Validation dataset size: {len(eval_dataset)}")
    else:
        logger.info(f"Validation dataset is None")

    # 7. Setup Trainer
    data_collator = DataCollatorForInstructionTuning(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=ft_config.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        )

    # 8. Train
    logger.info("Starting model training...")
    # Support resuming from a checkpoint
    _ = trainer.train()
    
    # --- Save Full Training and Evaluation History ---
    logger.info("Saving full training and evaluation history...")
    log_history = trainer.state.log_history
    
    # Separate training and evaluation logs for clarity and easier plotting
    train_history = [log for log in log_history if 'loss' in log and 'eval_loss' not in log]
    eval_history = [log for log in log_history if 'eval_loss' in log]

    history_data = {
        "train_history": train_history,
        "eval_history": eval_history
    }

    history_file_path = os.path.join(ft_config.training_args.output_dir, "log_history.json")
    with open(history_file_path, "w") as f:
        json.dump(history_data, f, indent=4)
    logger.info(f"Full log history saved to {history_file_path}")

    logger.info("Training finished. Saving model and metrics.")
    
    final_model_save_path = os.path.join(ft_config.training_args.output_dir, "final_model")
    model.save_pretrained(final_model_save_path)
    tokenizer.save_pretrained(final_model_save_path)
    logger.info(f"Final model saved to {final_model_save_path}")

    # 9. Evaluate
    if len(eval_dataset_names) > 0:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # device_map = {"": "cuda:0"}
        device_map = "auto"
        eval_dataset_dir = EVAL_DATASET_PATH
        all_results = {}
        summary_metrics = {}

        ft_config_path = os.path.join(ft_config.training_args.output_dir, "finetune_config.json")
        ft_config_dict = json.load(open(ft_config_path))

        model, tokenizer = load_model_for_evaluation(ft_config_dict, device_map)

        batch_size = ft_config_dict['training_args']['per_device_eval_batch_size']
        output_dir = ft_config_dict['training_args']['output_dir']

        model_name = ft_config_dict['model_name']
        if ("7b" in model_name) or ("8b" in model_name):
            batch_size = 24
        else:
            batch_size = 24

        for dataset_name in eval_dataset_names:
            try:
                total_samples = EVAL_DATASET_SIZE[dataset_name]
                predictions, metrics = evaluate_single_dataset(
                    model, tokenizer, output_dir, dataset_name, eval_dataset_dir,
                    batch_size=batch_size,
                    category_samples=None,
                    total_samples=total_samples,
                    embedder=embedder,
                )
                all_results[dataset_name] = {
                    "predictions": predictions,
                    "metrics": metrics
                }
                summary_metrics[dataset_name] = metrics
            except Exception as e:
                logger.error(f"Failed to evaluate dataset {dataset_name}: {str(e)}")
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
        summary_file = os.path.join(ft_config_dict['training_args']['output_dir'], "evaluation_summary.json")
        with open(summary_file, "w", encoding='utf-8') as f:
            json.dump(summary_metrics, f, indent=4, ensure_ascii=False)
        logger.info(f"Evaluation summary saved to {summary_file}")

if __name__ == "__main__":
    main()

