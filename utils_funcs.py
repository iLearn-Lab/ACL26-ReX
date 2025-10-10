import json
from pathlib import Path
from typing import Union, Iterable, Dict, Any, Optional, List
import pandas as pd
import torch
import os
from config_schemas import FinetuneConfig
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer


def extract_llm_response_from_think(raw_response: str) -> str:
    """
    从大语言模型的原始输出中提取思考链（<think>...</think>）之后的核心回复。

    这个函数会查找第一个 "</think>" 标签，并返回该标签之后的所有文本。
    如果文本在 </think> 后有前导空格，这些空格会被移除。
    如果原始回复中不包含 "</think>" 标签，函数将返回完整的原始字符串。

    Args:
        raw_response: LLM 生成的原始字符串，可能包含 <think>...</think> 结构。

    Returns:
        处理过的字符串，只包含核心回复部分。
    """
    # 定义思考链的闭合标签
    closing_tag = "</think>"

    # 使用 split 方法，通过闭合标签将字符串分割一次
    # maxsplit=1 参数确保只在第一个找到的闭合标签处进行分割
    parts = raw_response.split(closing_tag, 1)

    # 如果分割后的列表长度大于1，说明找到了闭合标签
    if len(parts) > 1:
        # 第二部分 (parts[1]) 就是我们需要的内容
        # 使用 .lstrip() 移除可能存在的前导空格或换行符
        return parts[1].lstrip()
    else:
        # 如果列表长度不大于1，说明没有找到闭合标签，返回原始字符串
        return raw_response


def get_token_length(tokenizer: AutoTokenizer, text: str | List[Dict[str, str]]) -> int | List[int]:
    """
    Get the token length of the text.

    Args:
        tokenizer: The tokenizer to use.
        text: The text to get the token length of.

    Returns:
        The token length of the text.
    """
    if isinstance(text, str):
        return len(tokenizer(text)["input_ids"])
    elif isinstance(text, list):
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer(prompt)["input_ids"]
            return len(tokens)
        else:
            raise ValueError("Tokenizer does not support chat template.")
    else:
        raise ValueError(f"Invalid input type: {type(text)}")


def get_text_embeddings(embedder: SentenceTransformer, texts: List[str], batch_size: int = 256, **kwargs) -> np.ndarray:
    """
    使用加载的嵌入器为文本列表生成嵌入向量。
    如果聚类被禁用或嵌入器加载失败，则返回 None。

    Args:
        texts (List[str]): 需要生成嵌入的文本列表。
        batch_size (int): 批量处理大小。
        kwargs: embedder.encode 中的其他参数，如 show_progress_bar, convert_to_numpy 等。

    Returns:
        Optional[np.ndarray]: 嵌入向量数组，或在禁用/失败时返回 None。
    """
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        **kwargs
    )
    return embeddings


def _get_attn_implementation(torch_dtype: Optional[torch.dtype]) -> Optional[str]:
    """根据 dtype 和可用性确定注意力实现。"""
    logger = logging.getLogger(__name__)
    attn_implementation = None
    if torch.cuda.is_available() and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        # 仅当使用 bf16 或 fp16 时才考虑优化注意力
        if torch_dtype in [torch.bfloat16, torch.float16]:
            # 优先 Flash Attention 2 (需要安装 flash-attn)
            # Flash Attention 2 通常与 bf16 配合效果更好
            try:
                import flash_attn # 检查是否安装
                # 如果需要，可以添加版本检查: flash_attn.__version__
                attn_implementation = "flash_attention_2"
                logger.info(f"Flash Attention 2 available. Setting attn_implementation='flash_attention_2' (recommended for dtype={torch_dtype}).")
            except ImportError:
                logger.warning("flash-attn library not found or import failed. Falling back to 'sdpa'.")
                attn_implementation = "sdpa" # Scaled Dot Product Attention (PyTorch >= 2.0)
        else:
            logger.info(f"Optimal attention implementation requires bfloat16 or float16. Current dtype: {torch_dtype}. Using default.")
    else:
        logger.info("Not setting specific attention implementation (CUDA/SDPA unavailable or dtype mismatch).")
    return attn_implementation


def save_finetune_config(ft_config: FinetuneConfig):
    """
    Save the finetune configuration to a JSON file in the output directory.
    """

    # def convert_to_serializable(obj):
    #     """Convert non-serializable objects to serializable formats."""
    #     if hasattr(obj, '__dict__'):
    #         # For dataclass objects, convert to dict
    #         if hasattr(obj, '__dataclass_fields__'):
    #             result = {}
    #             for field_name, field_value in obj.__dict__.items():
    #                 result[field_name] = convert_to_serializable(field_value)
    #             return result
    #         else:
    #             return obj.__dict__
    #     elif isinstance(obj, (list, tuple)):
    #         return [convert_to_serializable(item) for item in obj]
    #     elif isinstance(obj, dict):
    #         return {key: convert_to_serializable(value) for key, value in obj.items()}
    #     elif hasattr(obj, '__name__'):  # For types like torch.float16
    #         return obj.__name__ if hasattr(obj, '__name__') else str(obj)
    #     elif isinstance(obj, Path):
    #         return str(obj)
    #     else:
    #         try:
    #             # Try JSON serialization
    #             json.dumps(obj)
    #             return obj
    #         except (TypeError, ValueError):
    #             # If not serializable, convert to string
    #             return str(obj)
    
    # Convert the entire config to a serializable format
    # config_dict = convert_to_serializable(ft_config)
    config_dict = ft_config.to_dict()
    
    # Save to JSON file
    config_file_path = os.path.join(ft_config.training_args.output_dir, "finetune_config.json")
    with open(config_file_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)
    print(f"Finetune configuration saved to {config_file_path}")

# 解析 unknown 参数
def parse_unknown_args(unknown_args: List[str]) -> Dict[str, str]:
    """
    解析 unknown 参数，支持 --key=value 和 --key value 格式
    返回一个字典，键是参数名（去掉--前缀），值是参数值
    """
    parsed_args = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--'):
            if '=' in arg:
                # --key=value 格式
                key, value = arg[2:].split('=', 1)
                parsed_args[key] = value
                i += 1
            else:
                # --key value 格式
                key = arg[2:]
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                    value = unknown_args[i + 1]
                    parsed_args[key] = value
                    i += 2
                else:
                    # 布尔标志，没有值
                    parsed_args[key] = True
                    i += 1
        else:
            i += 1
    return parsed_args



# 应用 unknown 参数覆盖配置
def apply_overrides(config_obj: Any, overrides: Dict[str, str], obj_name: str = ""):
    """
    将覆盖参数应用到配置对象上
    """
    for key, value in overrides.items():
        if hasattr(config_obj, key):
            original_value = getattr(config_obj, key)
            # 尝试转换为原始类型
            try:
                if isinstance(original_value, bool):
                    new_value = value.lower() in ('true', '1', 'yes', 'on') if isinstance(value, str) else bool(value)
                elif isinstance(original_value, int):
                    new_value = int(value)
                elif isinstance(original_value, float):
                    new_value = float(value)
                else:
                    new_value = value
                
                setattr(config_obj, key, new_value)
                print(f"Override {obj_name}.{key}: {original_value} -> {new_value}")
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert {key}={value} for {obj_name}: {e}")
        else:
            print(f"Warning: Unknown parameter {key} for {obj_name}")


def is_gpu_id_valid(gpu_id: Optional[int]) -> bool:
    """
    检查指定的 GPU ID 是否有效。

    Args:
        gpu_id (Optional[int]): 要检查的 GPU ID。None 表示自动选择，视为有效。

    Returns:
        bool: 如果 GPU ID 有效或为 None，则返回 True，否则返回 False。
    """
    if gpu_id is None:
        # None 代表 'auto' 或默认行为，是有效的
        return True

    if not torch.cuda.is_available():
        # 如果 CUDA 不可用，任何指定的 GPU ID 都无效
        print(f"Warning: CUDA is not available, but target_gpu_id={gpu_id} was specified.")
        return False

    num_gpus = torch.cuda.device_count()
    if 0 <= gpu_id < num_gpus:
        # GPU ID 在有效范围内
        return True
    else:
        # GPU ID 超出范围
        print(f"Error: Invalid target_gpu_id={gpu_id}. Available GPUs: {num_gpus} (indices 0 to {num_gpus - 1}).")
        return False


def read_jsonl_to_dataframe(
        file_path: Union[str, Path],
        encoding: str = "utf-8",
        lines: bool = True,
        convert_dates: bool = True,
        dtype: Dict[str, Any] = None,
        chunksize: int = None
) -> Union[pd.DataFrame, Iterable[pd.DataFrame]]:
    """
    将 JSON Lines 文件读取为 Pandas DataFrame

    Args:
        file_path: JSONL 文件路径
        encoding: 文件编码 (默认: utf-8)
        lines: 是否按行解析 (必须为 True)
        convert_dates: 自动转换日期字段 (默认: True)
        dtype: 指定列数据类型，例如 {"age": "int32"}
        chunksize: 分块读取的行数，None 表示全部读取

    Returns:
        DataFrame 或 DataFrame 迭代器 (当 chunksize 指定时)

    Raises:
        ValueError: 输入参数无效时
        FileNotFoundError: 文件不存在时
        JSONDecodeError: JSON 解析失败时
    """
    # 参数校验
    if not lines:
        raise ValueError("读取 JSONL 必须设置 lines=True")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 读取数据
    try:
        return pd.read_json(
            file_path,
            lines=True,
            encoding=encoding,
            convert_dates=convert_dates,
            dtype=dtype,
            chunksize=chunksize
        )
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"JSON 解析失败于文件 {file_path}", e.doc, e.pos
        ) from e


def write_dataframe_to_jsonl(
        df: pd.DataFrame,
        file_path: Union[str, Path],
        orient: str = "records",
        force_ascii: bool = False,
        date_format: str = "iso",
        mode: str = "w"
) -> None:
    """
    将 DataFrame 写入 JSON Lines 文件

    Args:
        df: 要写入的 DataFrame
        file_path: 输出文件路径
        encoding: 文件编码 (默认: utf-8)
        orient: 数据方向 (必须为 "records")
        force_ascii: 是否强制 ASCII 编码 (默认: False)
        date_format: 日期格式化方式，"iso" 或 "epoch"
        mode: 写入模式 ("w" 覆盖 / "a" 追加)

    Raises:
        ValueError: 输入参数无效时
        PermissionError: 文件写入权限不足时
    """
    # 参数校验
    if orient != "records":
        raise ValueError("JSONL 格式要求 orient='records'")

    if mode not in ("w", "a"):
        raise ValueError("模式必须是 'w' (覆盖) 或 'a' (追加)")

    # 转换数据
    try:
        df.to_json(
            path_or_buf=file_path,
            orient=orient,
            lines=True,
            force_ascii=force_ascii,
            date_format=date_format,
            mode=mode,
        )
        print("Finish Writing DataFrame to JSONL")
    except PermissionError as e:
        raise PermissionError(f"无写入权限: {file_path}") from e
