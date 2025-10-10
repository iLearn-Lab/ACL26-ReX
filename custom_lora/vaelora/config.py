# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union, List

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class VAELoRAConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`VAELoRAConfig`].
    VAELoRA (Variational Autoencoder LoRA) is a LoRA variant.
        1. Problem: Different domains (mathematics/law/physics/chemistry, etc.) require different "skills/experience." We parameterize "skills/experience" as a set of LoRA weight collections for several injection layers, denoted as $\{\Delta W^{(\ell)}\}_{\ell\in\mathcal S}$. Given an input $x$, the goal is to utilize a generative model $p_\psi(\Delta\theta \mid z, \text{cond})$ to produce sample-level $\Delta\theta$ (where $\theta$ represents the parameters of the Base LLM), thereby improving task performance.

        2. Core Assumption: There exists a low-dimensional latent variable $z$ that captures the structure of the "skill combination required by the sample." An Encoder/Router $q_\phi(z\mid x)$ can extract $z$ from the input, and a Decoder can conditionally generate multi-layer LoRA parameters.

    Args:
        target_modules: List[str], e.g., ["q_proj", "v_proj"]
        rank r, dict width r_b, latent dim d_z
        enc_hidden: int, a small MLP (4x latent by default)
        head_hidden: int, a small MLP (4x latent by default)
        s_max_init: float, scaling cap s_max schedule (you can update during training)
        free_bits: float, free-bits (nats) for each latent dim (VAE)
        use_bf16_compute: bool, if True: use bf16 compute
    """
    r: int = 8
    r_b: int = 16
    d_z: int = 16
    topk: int = 8
    lora_alpha: int = field(default=8, metadata={"help": "The alpha parameter for LoRA scaling."})

    # VAE encoder hidden width
    enc_hidden: int = 64

    # emitter hidden width
    head_hidden: int = 64

    # free-bits (nats) for each latent dim (VAE)
    free_bits: float = 0.75
    beta_coh: float = 0.1
    beta_kl: float = 0.1
    beta_recon: float = 0.1
    if_max: int = 0

    # precision knobs
    use_bf16_compute: bool = True
    mixer_weight_init: str = field(
        default="kaiming_uniform",
        metadata={"help": "How to initialize the weights of the mixer weight. Options: 'kaiming_uniform', 'identity', 'orthogonal'"}
    )

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to replace with Lora"}
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from VAELoRA"}
    )
    vaelora_dropout: float = field(default=0.0, metadata={"help": "VAELoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for VAELoRA. Can be 'none', 'all' or 'vaelora_only'"})
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of modules apart from VAELoRA layers to be set as trainable and saved in the final checkpoint"}
    )


    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.VAELORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
