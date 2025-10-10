
from typing import List, Optional, Union, Literal, Any # <--- 导入 Any 用于通用类型
from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class DataArguments:
    """数据集相关配置"""
    data_cache_dir: str = field(
        default=None,
        metadata={"help": "Hugging Face 数据集缓存目录路径"}
    )
    name: str = field(
        default="math500",
        metadata={"help": "数据集名称 (例如 'gsm8k')"}
    )
    train_name: str = field(
        default="train",
        metadata={"help": "训练数据集名称"}
    )
    val_name: str = field(
        default="val",
        metadata={"help": "验证数据集名称"}
    )
    test_name: str = field(
        default="test",
        metadata={"help": "测试数据集名称"}
    )
    val_split: float = field(
        default=0.1,
        metadata={"help": "在没有 validation 的情况下，从训练集中划分验证集的比例"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The maximum sequence length of tokenization."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Load only the first N training samples. Set to None to load all."}
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Load only the first N validation samples. Set to None to load all."}
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Load only the first N test samples. Set to None to load all."}
    )


@dataclass
class LoRAArguments:
    """LoRA 相关配置"""
    adapter_name: str = field(
        default="lora",
        metadata={"help": "Adapter name"}
    )
    lora_rank: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    target_modules: List[str] = field(
        default=None,
        metadata={"help": "LoRA target modules"}
    )

@dataclass
class VAELoRAArguments:
    """VAELoRA 相关配置"""
    adapter_name: str = field(
        default="vaelora",
        metadata={"help": "Adapter name"}
    )
    r: int = field(
        default=8,
        metadata={"help": "Rank dimension"}
    )
    r_b: int = field(
        default=16,
        metadata={"help": "Rank dimension of the Experience Bank"}
    )
    d_z: int = field(
        default=16,
        metadata={"help": "Dimension of the latent space"}
    )
    topk: int = field(
        default=8,
        metadata={"help": "Top-k"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "The alpha parameter for LoRA scaling."}
    )
    enc_hidden: int = field(
        default=64,
        metadata={"help": "Hidden dimension of the VAE encoder"}
    )
    head_hidden: int = field(
        default=64,
        metadata={"help": "Hidden dimension of the VAE decoder"}
    )
    free_bits: float = field(
        default=0.75,
        metadata={"help": "Free bits for each latent dim"}
    )
    beta_coh: float = field(
        default=0.1,
        metadata={"help": "coh loss"}
    )
    beta_kl: float = field(
        default=0.1,
        metadata={"help": "KL Loss"}
    )
    beta_recon: float = field(
        default=0.1,
        metadata={"help": "recontruction loss"}
    )
    if_max: int = field(
        default=0,
        metadata={"help": "Whether use max feature in recontrucsion."}
    )
    use_bf16_compute: bool = field(
        default=True,
        metadata={"help": "Use bf16 compute"}
    )
    mixer_weight_init: str = field(
        default="kaiming_uniform",
        metadata={"help": "How to initialize the weights of the mixer weight. Options: 'kaiming_uniform', 'identity', 'orthogonal'"}
    )
    vaelora_dropout: float = field(
        default=0.0,
        metadata={"help": "VAELoRA dropout"}
    )
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to replace with VAELoRA"}
    )
    exclude_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from VAELoRA"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"}
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for VAELoRA. Can be 'none', 'all' or 'vaelora_only'"}
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of modules apart from VAELoRA layers to be set as trainable and saved in the final checkpoint"}
    )


@dataclass
class GenerationArguments:
    """Generation 相关配置"""
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum new tokens"}
    )
    do_sample: bool = field(
        default=False,
        metadata={"help": "Do sample"}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Temperature"}
    )
    top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p"}
    )
    top_k: int = field(
        default=2,
        metadata={"help": "Top-k"}
    )

    def __post_init__(self):
        if self.do_sample:
            if self.temperature is None:
                self.temperature = 0.7
                print(f"Temperature is set to {self.temperature} because do_sample is True.")
            if self.top_p is None:
                self.top_p = 0.95
                print(f"Top-p is set to {self.top_p} because do_sample is True.")
            if self.top_k is None:
                self.top_k = 2
                print(f"Top-k is set to {self.top_k} because do_sample is True.")

@dataclass
class QuantizationArguments:
    """Quantization 相关配置"""
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4bit"}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8bit"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "4bit quantization type"}
    )
    bnb_4bit_compute_dtype_str: str = field(
        default="bfloat16",
        metadata={"help": "4bit quantization compute dtype"}
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,
        metadata={"help": "4bit quantization use double quant"}
    )
    llm_int8_threshold: float = field(
        default=6.0,
        metadata={"help": "8bit quantization threshold"}
    )
    llm_int8_enable_fp32_cpu_offload: bool = field(
        default=True,
        metadata={"help": "8bit quantization enable fp32 cpu offload"}
    )


@dataclass
class FinetuneConfig:
    """A unified dataclass to hold all finetuning configurations."""
    model_name: str
    model_hfid: str
    adapter_name: str
    config_path: str
    device: str
    data_args: DataArguments
    adapter_args: Union[LoRAArguments, VAELoRAArguments]
    quant_args: QuantizationArguments
    training_args: TrainingArguments
    gen_args: GenerationArguments

    def to_dict(self):
        """Convert the FinetuneConfig to a dictionary that can be serialized to JSON."""
        import json
        from pathlib import Path
        
        def convert_to_serializable(obj):
            """Convert non-serializable objects to serializable formats."""
            if hasattr(obj, '__dict__'):
                # For dataclass objects, convert to dict
                if hasattr(obj, '__dataclass_fields__'):
                    result = {}
                    for field_name, field_value in obj.__dict__.items():
                        result[field_name] = convert_to_serializable(field_value)
                    return result
                else:
                    # For other objects with __dict__ (like TrainingArguments)
                    result = {}
                    for attr_name, attr_value in obj.__dict__.items():
                        if not attr_name.startswith('_'):  # Skip private attributes
                            result[attr_name] = convert_to_serializable(attr_value)
                    return result
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif hasattr(obj, '__name__'):  # For types like torch.float16
                return obj.__name__ if hasattr(obj, '__name__') else str(obj)
            elif isinstance(obj, Path):
                return str(obj)
            else:
                try:
                    # Try JSON serialization
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    # If not serializable, convert to string
                    return str(obj)
        
        return convert_to_serializable(self)
