# evaluation/evaluation.py
import logging
import json
import os
from typing import Dict
import torch
import tqdm
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer

from utils_parser import extract_answer
from utils_grader import check_is_correct

from transformers import AutoTokenizer, GenerationConfig, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel
from constant import *
from utils_funcs import _get_attn_implementation

logger = logging.getLogger(__name__) # Logger for this module

def load_model_for_evaluation(ft_config_dict: Dict, device_map: Dict[str, int] = None, target_dir: str = None):
    """
    Loads the base model, adapter, and tokenizer from a specified path for evaluation.
    """
    if target_dir is None:
        target_dir = "final_model"

    final_model_save_path = os.path.join(ft_config_dict['training_args']['output_dir'], target_dir)
    
    # Load tokenizer saved with the model
    tokenizer = AutoTokenizer.from_pretrained(final_model_save_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token to eos_token for evaluation tokenizer.")

    # Quantization config
    quant_args = ft_config_dict['quant_args']
    if quant_args['load_in_4bit']:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_args['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=getattr(torch, quant_args['bnb_4bit_compute_dtype_str']),
            bnb_4bit_use_double_quant=quant_args['bnb_4bit_use_double_quant'],
        )
    elif quant_args['load_in_8bit']:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=getattr(torch, quant_args['bnb_4bit_compute_dtype_str']),
            llm_int8_threshold=quant_args['llm_int8_threshold'],
            llm_int8_enable_fp32_cpu_offload=quant_args['llm_int8_enable_fp32_cpu_offload'],
        )
    else:
        bnb_config = None
    logger.info(f"bnb_config: {bnb_config}")
    attn_implementation = _get_attn_implementation(getattr(torch, quant_args['bnb_4bit_compute_dtype_str']))
    model_kwargs = {
        "pretrained_model_name_or_path": ft_config_dict['model_hfid'],
        # "cache_dir": HF_CACHE_DIR,
        "trust_remote_code": True,
        "torch_dtype": getattr(torch, quant_args['bnb_4bit_compute_dtype_str']),
        "use_cache": False,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    # Load base model
    logger.info(f"Loading model and adapter from: {final_model_save_path}")
    base_model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model = PeftModel.from_pretrained(base_model, final_model_save_path)

    model.eval()
    logger.info("Successfully loaded PEFT model for evaluation.")

    # Generation config
    # Attention: Because do_sample is False, temperature is not used.
    gen_config_args = ft_config_dict['gen_args']
    model.generation_config = GenerationConfig(
        max_new_tokens=gen_config_args['max_new_tokens'],
        do_sample=gen_config_args['do_sample'],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    return model, tokenizer


def perform_generation(model, tokenizer, test_dataset: Dataset, embedder: SentenceTransformer = None, **kwargs):
    """
    Generates predictions for the entire test dataset.
    """
    all_predictions = []
    logger.info(f"Generating predictions for {len(test_dataset)} test samples...")

    for example in test_dataset:
        inp = example.get('input', '') or ""
        user_content = example['instruction'] + f"\n{inp}"
        prompt = tokenizer.apply_chat_template(
            [[{"role": "user", "content": user_content}]],
            tokenize=False,
            add_generation_prompt=True,
        )

        # inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ft_config.data_args.max_seq_length).to(model.device)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, pad_to_multiple_of=8).to(model.device)
        if embedder is not None:
            embeddings = embedder.encode(prompt, batch_size=128, convert_to_tensor=True)
            inputs["external_embedding"] = embeddings.to(model.device)
        
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                generation_config=model.generation_config
            )
        input_length = inputs.input_ids.shape[1]
        generated_ids = output_ids[0, input_length:]
        decoded_prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
        pred_answer = extract_answer(decoded_prediction)
        is_correct = check_is_correct(pred_answer, example.get('answer', ''))

        all_predictions.append({
            "instruction": example['instruction'],
            "input": inp,
            "reference_answer": str(example.get('answer', '')),
            "reference_solution": str(example.get('output', '')),
            "decoded_prediction": decoded_prediction,
            "extracted_prediction": pred_answer,
            "is_correct": is_correct
        })
    logger.info("Generation complete.")
    return all_predictions


def perform_batch_generation(model, tokenizer, test_dataset: Dataset, batch_size: int = 8, embedder: SentenceTransformer = None):
    """
    Generates predictions for the entire test dataset in batches.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer to use for encoding/decoding
        test_dataset: The dataset to generate predictions for
        batch_size: The number of examples to process in each batch
        
    Returns:
        List of dictionaries containing predictions and metadata
    """
    all_predictions = []
    total_examples = len(test_dataset)
    logger.info(f"Generating predictions for {total_examples} test samples with batch size {batch_size}...")
    
    # Process dataset in batches
    for batch_start in range(0, total_examples, batch_size):
        # Get the current batch (handle last batch which might be smaller)
        batch_end = min(batch_start + batch_size, total_examples)
        batch = test_dataset.select(range(batch_start, batch_end))
        
        # Prepare batch inputs
        user_content_list = []
        for example in batch:
            inp = example.get('input', '') or ""
            user_content = example['instruction'] + f"\n{inp}"
            user_content_list.append([{
                "role": "user",
                "content": user_content
            }])

        batch_prompts = tokenizer.apply_chat_template(
            user_content_list,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize all prompts in the batch
        batch_inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True, 
            add_special_tokens=True
        ).to(model.device)

        if embedder is not None:
            embeddings = embedder.encode(batch_prompts, batch_size=128, convert_to_tensor=True, show_progress_bar=False)
            embeddings = embeddings.to(model.device)
            # batch_inputs["external_embedding"] = embeddings


        with torch.inference_mode():
            if embedder is not None:
                router_A_weights = model.groupexplora_router_A(embeddings)
                router_B_weights = model.groupexplora_router_B(embeddings)
                model.router_weights["router_A_weights"] = router_A_weights
                model.router_weights["router_B_weights"] = router_B_weights
            
            batch_outputs = model.generate(
                **batch_inputs,
                generation_config=model.generation_config
            )
        
        # Process each example in the batch
        for i, example in enumerate(batch):
            # Extract input length for this specific example (may vary due to padding)
            input_length = len(batch_inputs.input_ids[i])
            # Extract generated tokens (excluding input)
            generated_ids = batch_outputs[i, input_length:]
            # Decode the generated text
            decoded_prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
            # Extract the answer from the prediction
            pred_answer = extract_answer(decoded_prediction)
            # Check if the prediction is correct
            is_correct = check_is_correct(pred_answer, example.get('answer', ''))
            
            # Store the prediction and metadata
            all_predictions.append({
                "instruction": example['instruction'],
                "raw_id": example.get('raw_id', ''),
                "input": example.get('input', '') or "",
                "reference_answer": str(example.get('answer', '')),
                "reference_solution": str(example.get('output', '')),
                "decoded_prediction": decoded_prediction,
                "extracted_prediction": pred_answer,
                "is_correct": is_correct
            })
    
    logger.info("Batch generation complete.")
    return all_predictions


def filter_by_category(dataset_dict, samples_per_category=100, seed=42):
    """
    Filter dataset to keep only specified number of samples per category.
    
    Args:
        dataset_dict: DatasetDict containing the dataset
        samples_per_category: Number of samples to keep per category
        seed: Random seed for reproducibility
        
    Returns:
        Filtered DatasetDict
    """

    from datasets import DatasetDict, concatenate_datasets
    # Get the dataset
    dataset = dataset_dict['test']
    
    # Group by category
    categories = dataset.unique('category')
    filtered_datasets = []
    
    for category in categories:
        # Filter samples for this category
        cat_samples = dataset.filter(lambda x: x['category'] == category)
        
        # If category has more samples than needed, randomly select
        if len(cat_samples) > samples_per_category:
            cat_samples = cat_samples.shuffle(seed=seed).select(range(samples_per_category))
        
        filtered_datasets.append(cat_samples)
    
    # Combine all filtered category datasets
    filtered_dataset = concatenate_datasets(filtered_datasets)
    
    # Shuffle the final dataset
    filtered_dataset = filtered_dataset.shuffle(seed=seed)
    
    return DatasetDict({'test': filtered_dataset})


def evaluate_single_dataset(
        model, tokenizer, output_dir: str,
        dataset_name: str,
        eval_dataset_dir: str,
        seed: int = 42,
        batch_size: int = 8,
        category_samples: int = None,
        total_samples: int = None,
        embedder: SentenceTransformer = None,
    ):
    """
    Evaluate a single dataset and return predictions and metrics.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        ft_config: Finetuning configuration
        dataset_name: Name of the dataset to evaluate
        eval_dataset_dir: Directory containing evaluation datasets
        
    Returns:
        Tuple of (predictions, metrics)
    """
    logger.info(f"Starting evaluation for dataset: {dataset_name}")
    
    # Load the test dataset
    test_dataset = load_dataset(
        path=eval_dataset_dir,
        data_files={
            "test": f"{dataset_name}.json"
        }
    )
    test_dataset['test'] = test_dataset['test'].shuffle(seed=seed)
    if dataset_name != "math500":
        if total_samples is not None:
            if len(test_dataset['test']) > total_samples:
                test_dataset['test'] = test_dataset['test'].select(range(total_samples))
    
    test_dataset = test_dataset['test']

    # Generate predictions
    if isinstance(batch_size, int) and batch_size > 1:
        all_predictions = perform_batch_generation(model, tokenizer, test_dataset, batch_size, embedder)
    else:
        all_predictions = perform_generation(model, tokenizer, test_dataset, embedder)

    # Calculate metrics
    correct_count = sum(1 for pred in all_predictions if pred['is_correct'])
    accuracy = correct_count / len(all_predictions) if all_predictions else 0.0
    test_metrics = {
        "test_accuracy": accuracy,
        "total_samples": len(all_predictions),
        "correct_samples": correct_count,
        "dataset_name": dataset_name
    }

    # Save predictions
    total_samples = len(all_predictions)
    predictions_file = os.path.join(output_dir, f"predictions_{dataset_name}_t{total_samples}.json")
    with open(predictions_file, "w", encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=4, ensure_ascii=False)
    logger.info(f"Detailed test predictions for {dataset_name} saved to {predictions_file}")

    # Save metrics
    metrics_file = os.path.join(output_dir, f"metrics_{dataset_name}_t{total_samples}.json")
    with open(metrics_file, "w", encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=4)
    logger.info(f"Test metrics for {dataset_name} saved to {metrics_file}")
    
    logger.info(f"Evaluation completed for {dataset_name}. Accuracy: {accuracy:.4f} ({correct_count}/{len(all_predictions)})")
    
    return all_predictions, test_metrics
