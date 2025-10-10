# Copyright 2024-present the HuggingFace Inc. team.
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

import warnings
from dataclasses import asdict
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import ModulesToSaveWrapper, _get_submodules
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

from .config import VAELoRAConfig
from .layer import VAELinear, VAELayer


class VAELoRAModel(BaseTuner):
    """
    Creates VAELoRA model from a pretrained transformers model.
    """
    prefix: str = "vaelora_"

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def _check_new_adapter_config(self, config: VAELoRAConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.
        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.
        """
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(vaelora_config, key):
        return check_target_module_exists(vaelora_config, key)
    
    def _create_and_replace(
        self,
        vaelora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "fan_in_fan_out": vaelora_config.fan_in_fan_out,
            "bias": bias,
        }
        
        if isinstance(target, VAELinear):
            target.update_layer(
                adapter_name=adapter_name,
                r=vaelora_config.r,
                r_b=vaelora_config.r_b,
                d_z=vaelora_config.d_z,
                free_bits=vaelora_config.free_bits,
                enc_hidden=vaelora_config.enc_hidden,
                head_hidden=vaelora_config.head_hidden,
                topk=vaelora_config.topk,
                lora_alpha=vaelora_config.lora_alpha,
                mixer_weight_init=vaelora_config.mixer_weight_init,
                beta_recon=vaelora_config.beta_recon,
                beta_coh=vaelora_config.beta_coh,
                beta_kl=vaelora_config.beta_kl,
                if_max=vaelora_config.if_max,
            )
        else:
            new_module = self._create_new_module(
                vaelora_config=vaelora_config,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer
        
        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)
        
        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "vaelora_" in name:
                if not any(p.device == meta for p in module.parameters()):
                    module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False
        
        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue
            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "vaelora_only":
                for m in model.modules():
                    if isinstance(m, VAELayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
    
    @staticmethod
    def _create_new_module(vaelora_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = vaelora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = vaelora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Conv1d`, `transformers.pytorch_utils.Conv1D`."
            )

        new_module = VAELinear(
            target, 
            adapter_name, 
            r=vaelora_config.r,
            r_b=vaelora_config.r_b,
            d_z=vaelora_config.d_z,
            free_bits=vaelora_config.free_bits,
            enc_hidden=vaelora_config.enc_hidden,
            head_hidden=vaelora_config.head_hidden,
            topk=vaelora_config.topk,
            lora_alpha=vaelora_config.lora_alpha,
            mixer_weight_init=vaelora_config.mixer_weight_init,
            beta_recon=vaelora_config.beta_recon,
            beta_coh=vaelora_config.beta_coh,
            beta_kl=vaelora_config.beta_kl,
            if_max=vaelora_config.if_max,
            **kwargs
        )

        return new_module
    
    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config
    
    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)
    
    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabled, the model will use the original model weights.
        """
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.
        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```
        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, VAELayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name
    
    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config
    
    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        if merge:
            if getattr(self.model, "quantization_method", None) == "gptq":
                raise ValueError("Cannot merge VAELoRA layers when the model is gptq quantized")
        
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                self._replace_module(parent, target_name, target.get_base_layer(), target)
        return self.model
    
    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, VAELayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []
    
    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        """
        This method merges the VAELoRA layers into the base model. This is needed if someone wants to use the base
        model as a standalone model.
        """
        return self._unload_and_optionally_merge(
            merge=True, progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )
    
    def unload(self):
        """
        Gets back the base model by removing all the VAELoRA modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)
    
    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int, int, int]:
        """
        Get the number of savable parameters for VAELoRA model.
        """
        vae_params = 0
        bank_params = 0
        ft_params = 0
        llm_total_params = 0
        
        for n, p in self.named_parameters():
            llm_total_params += p.numel()
            
            if self.prefix in n:  # vaelora_ parameters
                if "vaelora_U" in n:
                    bank_params += p.numel()
                elif "vaelora_V" in n:
                    bank_params += p.numel()
                else:
                    vae_params += p.numel()
                
                if "vaelora_" in n:
                    ft_params += p.numel()
        
        return vae_params, bank_params, ft_params, llm_total_params      

    def print_savable_parameters(self) -> None:
        """
        Print the number of savable parameters for VAELoRA model.
        """
        vae_params, bank_params, ft_params, llm_total_params = self.get_nb_savable_parameters()
        print(f"VAELoRA bank params to-be-saved (float32-equivalent): {bank_params:,d}")
        print(f"VAELoRA VAE params to-be-saved (float32-equivalent): {vae_params:,d}")
        print(f"Total VAELoRA parameters:               {ft_params:,d}")
        print(f"Total LLM parameters:                   {llm_total_params:,d}")
        print("-"*60)
        print(f"VAELoRA ratio:    {100 * ft_params / llm_total_params:.4f}%")
        print(f"VAE ratio:       {100 * vae_params / ft_params:.4f}%")
        print(f"Bank ratio:      {100 * bank_params / ft_params:.4f}%")
        print("="*60)