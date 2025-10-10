# -*- coding: utf-8 -*-
import logging
from typing import Dict, Any
import copy
from dataclasses import dataclass
from typing import Sequence
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer
import torch # 导入 torch 以便在 preprocess_for_inference 中指定返回类型
from sentence_transformers import SentenceTransformer
import os
import time

from config_schemas import FinetuneConfig


def prepare_datasets(ft_config: FinetuneConfig, tokenizer: AutoTokenizer, embedder: SentenceTransformer=None):
    """
    Loads, preprocesses, and prepares the training, validation, and test datasets.
    """
    data_args = ft_config.data_args
    logger = logging.getLogger(__name__)
    
    # Load dataset from path
    data_path = f"{ft_config.data_args.data_cache_dir}/{ft_config.data_args.name}"

    data_files = {}
    data_files["train"] = f"{data_args.train_name}.json"

    if len(data_args.val_name) > 0:
        data_files["val"] = f"{data_args.val_name}.json"

    dataset = load_dataset(path=data_path, data_files=data_files)
    
    logger.info(f"Loaded dataset: {data_args.name}. Splits: {list(dataset.keys())}")
    logger.info(f"{dataset}")
    logger.info(f"Train dataset size: {len(dataset['train'])}")

    # Shuffle and sample datasets
    train_dataset = dataset["train"].shuffle(seed=ft_config.training_args.seed)
    
    # Split training set for validation if no validation set is provided
    if len(data_args.val_name) > 0:
        eval_dataset = dataset["val"].shuffle(seed=ft_config.training_args.seed)
    else:
        if abs(data_args.val_split - 0.0) < 0.00001:
            eval_dataset = None
            logger.info(f"Validation dataset is None: original ratio {data_args.val_split}")
        else:
            logger.info(f"No validation split found. Splitting from train set with ratio {data_args.val_split}")
            split_dataset = train_dataset.train_test_split(test_size=data_args.val_split, shuffle=True, seed=ft_config.training_args.seed)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']        

    if data_args.max_train_samples is not None:
        max_train_samples = int(data_args.max_train_samples) if isinstance(data_args.max_train_samples, str) else data_args.max_train_samples
        if max_train_samples < len(train_dataset):
            train_dataset = train_dataset.select(range(max_train_samples))
    
    if (data_args.max_val_samples is not None) and (eval_dataset is not None):
        max_val_samples = int(data_args.max_val_samples) if isinstance(data_args.max_val_samples, str) else data_args.max_val_samples
        if max_val_samples < len(eval_dataset):
            eval_dataset = eval_dataset.select(range(max_val_samples))

    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset is not None:
        logger.info(f"Validation dataset size: {len(eval_dataset)}")
    else:
        logger.info(f"Validation dataset is None")

    def preprocess_function(examples):
        all_full_texts = []
        all_prompt_texts = []
        embedding_texts = []
        for instruction, inp, output in zip(examples['instruction'], examples['input'], examples['output']):
            user_content = instruction + f"\n{inp}"
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_messages = [{"role": "user", "content": user_content}]
                response_messages = [{"role": "assistant", "content": output}]
                # 得到只包含 user 的 prompt 文本 和 完整对话文本
                prompt_only = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                full_text = tokenizer.apply_chat_template(prompt_messages + response_messages, tokenize=False, add_generation_prompt=False)
            else:
                raise ValueError("Tokenizer does not support chat template.")

            all_prompt_texts.append(prompt_only)
            all_full_texts.append(full_text)
            embedding_texts.append(inp)

        # 修复：对 prompt 和 full text 使用一致的 tokenization 参数
        prompt_inputs = tokenizer(
            all_prompt_texts,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding=False,
            add_special_tokens=True,
            return_tensors=None
        )
        model_inputs = tokenizer(
            all_full_texts,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding=False,
            add_special_tokens=True,
            return_tensors=None
        )

        if embedder is not None:
            embeddings = embedder.encode(embedding_texts, batch_size=128, convert_to_tensor=True, show_progress_bar=False)
            model_inputs["external_embedding"] = embeddings

        labels = copy.deepcopy(model_inputs["input_ids"])
        for i in range(len(labels)):
            prompt_len = len(prompt_inputs["input_ids"][i])
            # 确保不会超出实际序列长度
            actual_prompt_len = min(prompt_len, len(labels[i]))
            # 屏蔽 prompt 部分
            labels[i][:actual_prompt_len] = [-100] * actual_prompt_len

        model_inputs["labels"] = labels
        return model_inputs
    
    with ft_config.training_args.main_process_first(desc="dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            # num_proc=ft_config.training_args.dataloader_num_workers,
            remove_columns=train_dataset.column_names,
            desc="Running tokenizer on train dataset",
        )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                # num_proc=ft_config.training_args.dataloader_num_workers,
                remove_columns=eval_dataset.column_names,
                desc="Running tokenizer on validation dataset",
            )

    return train_dataset, eval_dataset


@dataclass
class DataCollatorForInstructionTuning(object):
    """
    专门为指令微调设计的数据整理器。
    
    这个 collator 专门处理您预处理函数生成的数据格式：
    - input_ids: 完整的输入序列 (prompt + response)
    - attention_mask: 注意力掩码
    - labels: 已经正确掩码的标签 (prompt 部分为 -100)
    
    它确保所有序列在批次内填充到相同长度，同时保持标签的正确性。
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 从每个实例中提取必要的字段
        input_ids = [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in instances]
        labels = [torch.tensor(instance["labels"], dtype=torch.long) for instance in instances]
        
        # 修复：正确处理 external_embedding
        external_embeddings = []
        for instance in instances:
            if "external_embedding" in instance:
                # 如果已经是张量，直接使用；否则转换为张量
                if isinstance(instance["external_embedding"], torch.Tensor):
                    external_embeddings.append(instance["external_embedding"])
                else:
                    external_embeddings.append(torch.tensor(instance["external_embedding"], dtype=torch.float32))
        
        # 堆叠成批次张量
        if external_embeddings:
            external_embedding = torch.stack(external_embeddings, dim=0)
        else:
            external_embedding = None
        
        # 使用 PyTorch 的 pad_sequence 进行填充
        # input_ids 用 pad_token_id 填充
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id,
            padding_side=self.tokenizer.padding_side
        )
        
        # labels 用 -100 填充（这是损失函数忽略的值）
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=-100,
            padding_side=self.tokenizer.padding_side
        )
        
        # 生成注意力掩码：非 pad_token 的位置为 1，pad_token 的位置为 0
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        # 只有当 external_embedding 存在时才添加
        if external_embedding is not None:
            result["external_embedding"] = external_embedding
            
        return result