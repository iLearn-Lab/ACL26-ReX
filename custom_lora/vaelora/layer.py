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

import warnings
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .vae import EmitterHead, ReconstructionLoss, KLLoss, CoherenceLoss
from .vae import VAEEncoder, VAEDecoder, TopkRouter


class VAELayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights

    adapter_layer_names = (
        "vaelora_encoder", 
        "vaelora_decoder", 
        "vaelora_head", 
        "vaelora_router_A",
        "vaelora_router_B",
        "vaelora_U", 
        "vaelora_V", 
        "vaelora_w",
        "vaelora_reconstruction_loss", 
        "vaelora_kl_loss", 
        "vaelora_coh_loss"
    )

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

        self.r = {}
        self.r_b = {}
        self.d_z = {}
        self.topk = {}
        self.lora_scale = {}
        self.free_bits = {}
        self.enc_hidden = {}
        self.head_hidden = {}
        self.mixer_weight_init = {}
        self.beta_recon = {}
        self.beta_kl = {}
        self.beta_coh = {}
        self.if_max = {}

        # VAE Encoder & Decoder
        self.vaelora_encoder = nn.ModuleDict({})
        self.vaelora_decoder = nn.ModuleDict({})

        self.vaelora_head = nn.ModuleDict({})
        self.vaelora_router_A = nn.ModuleDict({})
        self.vaelora_router_B = nn.ModuleDict({})

        # Shared dictionaries (trainable, per layer)
        # U: [d_out, r_b], V: [r_b, d_in]
        # Initialize with (approx) orthogonal columns/rows for stability
        self.vaelora_U = nn.ParameterDict({})
        self.vaelora_V = nn.ParameterDict({})
        self.vaelora_w = nn.ParameterDict({})

        self.vaelora_reconstruction_loss = nn.ModuleDict({})
        self.vaelora_kl_loss = nn.ModuleDict({})
        self.vaelora_coh_loss = nn.ModuleDict({})
        

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
            self, adapter_name: str, r: int, r_b: int, d_z: int, free_bits: float, 
            enc_hidden: int, head_hidden: int, topk: int, lora_alpha: int, mixer_weight_init: str,
            beta_recon: float, 
            beta_kl: float, 
            beta_coh: float, 
            if_max: int
        ):
        """
        Update the layer with the given adapter name.

        Args:
            adapter_name: The name of the adapter.
            r: The rank of the LoRA matrices.
            r_b: The rank of the Experience Bank.
            d_z: The dimension of the latent space.
        """
        self.r[adapter_name] = r
        self.r_b[adapter_name] = r_b
        self.d_z[adapter_name] = d_z
        self.free_bits[adapter_name] = free_bits
        self.enc_hidden[adapter_name] = enc_hidden
        self.head_hidden[adapter_name] = head_hidden
        self.topk[adapter_name] = topk
        self.lora_scale[adapter_name] = lora_alpha / r
        self.mixer_weight_init[adapter_name] = mixer_weight_init
        self.beta_coh[adapter_name] = beta_coh
        self.beta_kl[adapter_name] = beta_kl
        self.beta_recon[adapter_name] = beta_recon
        self.if_max[adapter_name] = if_max

        self.vaelora_encoder[adapter_name] = VAEEncoder(self.in_features, d_z, hidden=enc_hidden, if_max=if_max)
        self.vaelora_decoder[adapter_name] = VAEDecoder(d_z, self.in_features, hidden=enc_hidden)
        self.vaelora_head[adapter_name] = EmitterHead(d_z, r_b, r, hidden=head_hidden)
        self.vaelora_router_A[adapter_name] = TopkRouter(r_b=r_b, topk=topk, gamma=0.0001)
        self.vaelora_router_B[adapter_name] = TopkRouter(r_b=r_b, topk=topk, gamma=0.0001)

        self.vaelora_reconstruction_loss[adapter_name] = ReconstructionLoss(beta=beta_recon)
        self.vaelora_kl_loss[adapter_name] = KLLoss(beta=beta_kl)
        self.vaelora_coh_loss[adapter_name] = CoherenceLoss(r_b=r_b, beta=beta_coh)

        U = nn.Parameter(torch.empty(self.out_features, r_b))
        V = nn.Parameter(torch.empty(r_b, self.in_features))
        w = nn.Parameter(torch.empty(r, r))
        self.vaelora_U[adapter_name] = U
        self.vaelora_V[adapter_name] = V
        self.vaelora_w[adapter_name] = w
        
        self.reset_lora_parameters(adapter_name)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)
        
    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.vaelora_U.keys():
            with torch.no_grad():
                nn.init.orthogonal_(self.vaelora_U[adapter_name])  # U^T U ~ I
                nn.init.orthogonal_(self.vaelora_V[adapter_name])  # V V^T ~ I (good start)

                if self.mixer_weight_init[adapter_name] == "kaiming_uniform":
                    nn.init.kaiming_uniform_(self.vaelora_w[adapter_name], a=math.sqrt(5))
                elif self.mixer_weight_init[adapter_name] == "identity":
                    nn.init.eye_(self.vaelora_w[adapter_name])
                elif self.mixer_weight_init[adapter_name] == "orthogonal":
                    nn.init.orthogonal_(self.vaelora_w[adapter_name])
                else:
                    raise ValueError(f"Unknown mixer_weight_init: {self.mixer_weight_init[adapter_name]}")

class VAELinear(nn.Linear, VAELayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int,
        r_b: int,
        d_z: int,
        free_bits: float,
        enc_hidden: int,
        head_hidden: int,
        topk: int,
        lora_alpha: int,
        beta_recon: float, 
        beta_kl: float, 
        beta_coh: float, 
        if_max: int,
        mixer_weight_init: str,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        VAELayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.fan_in_fan_out = fan_in_fan_out
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
        self.update_layer(adapter_name, r, r_b, d_z, free_bits, enc_hidden, head_hidden, topk, lora_alpha, mixer_weight_init, beta_recon, beta_kl, beta_coh, if_max)
        
    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        warnings.warn(f"VAELoRA: unsupported merge operation")
        return

    def unmerge(self) -> None:
        warnings.warn(f"VAELoRA: unsupported unmerge operation")
        return
    
    def _emit_AB(self, z: torch.Tensor, active_adapter: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        U: [d_out, r_b]
        V: [r_b, d_in]
        P_A: [B, r_b, r]
        P_B: [B, r, r_b]

        Return A: [B, d_out, r], B: [B, r, d_in]
        A = U @ P_A, B = P_B @ V

        A: [B, d_out, r]
        B: [B, r, d_in]
        """
        U = self.vaelora_U[active_adapter]  # [d_out, r_b]
        w = self.vaelora_w[active_adapter]  # [r, r]
        V = self.vaelora_V[active_adapter]  # [r_b, d_in]
        P_A, P_B = self.vaelora_head[active_adapter](z)  # [B, r_b, r], [B, r, r_b]
        # batched matmul
        # A[b] = U @ P_A[b]
        # A = torch.einsum("or,brk->bok", U, P_A)     # [B, d_out, r]
        # B[b] = P_B[b] @ V
        # B = torch.einsum("brq,qd->brd", P_B, V)     # [B, r, d_in]

        # (B, r, r_b) (r_b, d_out) -> (B, r, d_out)
        A = self.vaelora_router_A[active_adapter](P_A.transpose(-2, -1), U.transpose(-2, -1))
        A = A.transpose(-2, -1)  # [B, d_out, r]
        A = torch.matmul(A, w)   # [B, d_out, r] @ [r, r] -> [B, d_out, r]
        
        # (B, r, r_b) (r_b, d_in) -> (B, r, d_in)
        B = self.vaelora_router_B[active_adapter](P_B, V)
        return A, B
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        VAELoRA forward pass implementing: Y = W_0 x + x A B

        Args:
            x: Input tensor of shape (batch_size, seq_len, in_features)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, out_features)
        """
        
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)

        for active_adapter in self.active_adapters:
            if active_adapter not in self.vaelora_encoder.keys():
                continue

            x = x.to(self.vaelora_U[active_adapter].dtype)
            mu, logvar, origin_x = self.vaelora_encoder[active_adapter](x)
            x_reconstructed = self.vaelora_decoder[active_adapter](mu, logvar)
            A, B = self._emit_AB(mu, active_adapter)

            # x: [B, T, d_in], B^T: [B, d_in, r] -> tmp: [B, T, r]
            temp = torch.bmm(x, B.transpose(-2, -1))
            # tmp @ A^T: [B, T, r] @ [B, r, d_out] -> [B, T, d_out]
            result = result + torch.bmm(temp, A.transpose(-2, -1)) * self.lora_scale[active_adapter]

            kl_loss = self.vaelora_kl_loss[active_adapter](mu, logvar, self.free_bits[active_adapter])
            coh_loss = self.vaelora_coh_loss[active_adapter](self.vaelora_U[active_adapter], self.vaelora_V[active_adapter])
            reconstruction_loss = self.vaelora_reconstruction_loss[active_adapter](origin_x, x_reconstructed)
        result = result.to(previous_dtype)
        return result