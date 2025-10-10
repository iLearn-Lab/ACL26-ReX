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

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional

# -----------------------------
# Micro-VAE Encoder
# -----------------------------

class VAEEncoder(nn.Module):
    def __init__(self, d_in: int, d_z: int, hidden: int, if_max=0):
        super().__init__()
        self.d_z = d_z
        self.if_max = if_max == 1

        if self.if_max:
            self.d_in = d_in * 2
        else:
            self.d_in = d_in

        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden, d_z)
        self.logvar_head = nn.Linear(hidden, d_z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, d]
        return mu, logvar: [B, d_z]
        """
        assert x.dim() == 3, "Input must be [B, T, d]"
        if self.if_max:
            x_mean = x.mean(dim=1)
            x_max = x.max(dim=1)
            x_in = torch.concat([x_mean, x_max], dim=-1)
        else:
            x_in = x.mean(dim=1)
        h = self.mlp(x_in)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar, x_in

class VAEDecoder(nn.Module):
    def __init__(self, d_z: int, d_out: int, hidden: int):
        super().__init__()
        self.d_z = d_z
        self.d_out = d_out
        self.mlp = nn.Sequential(
            nn.Linear(d_z, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.out_head = nn.Linear(hidden, d_out)
    
    def _encode_z(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Re-parameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        mu: [B, d_z]
        logvar: [B, d_z]
        return x: [B, d_out]
        """
        z = self._encode_z(mu, logvar)
        h = self.mlp(z)
        x = self.out_head(h)
        return x



#     Simple VAE: pool(last_dim=d) -> 2*enc_hidden -> d_z (mu/logvar)
#     Pooling = [mean; max] over sequence length. If input is [B, d], T=1.


# -----------------------------
# Hyper Head: z -> P_A (r_b x r), P_B (r x r_b)
# -----------------------------

class EmitterHead(nn.Module):
    def __init__(self, d_z: int, r_b: int, r: int, hidden: int):
        super().__init__()
        self.r_b = r_b
        self.r = r
        out_dim = r_b * r + r * r_b  # P_A and P_B
        self.net = nn.Sequential(
            nn.Linear(d_z, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: [B, d_z]
        returns:
          P_A: [B, r_b, r], P_B: [B, r, r_b]
        """
        B = z.shape[0]
        out = self.net(z)  # [B, out_dim]
        # split
        pa_flat = out[..., : self.r_b * self.r]
        pb_flat = out[..., self.r_b * self.r :]
        P_A = pa_flat.view(B, self.r_b, self.r)
        P_B = pb_flat.view(B, self.r, self.r_b)
        return P_A, P_B


class ReconstructionLoss(nn.Module):
    def __init__(self, beta, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta
    
    def forward(self, x: torch.Tensor, x_reconstructed: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x, x_reconstructed) * self.beta

class KLLoss(nn.Module):
    def __init__(self, beta, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta
    
    def _kl_vae(self, mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
        """KL divergence with numerical stability"""
        # 数值稳定性保护
        logvar_safe = torch.clamp(logvar, min=-10, max=10)
        mu_safe = torch.clamp(mu, min=-10, max=10)
        
        # KL散度计算
        kl_per_dim = -0.5 * (1 + logvar_safe - mu_safe.pow(2) - logvar_safe.exp())
        # if free_bits > 0:
        #     kl_per_dim = torch.clamp_min(kl_per_dim, free_bits)
        return kl_per_dim.sum(dim=-1).mean() * self.beta
    
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
        return self._kl_vae(mu, logvar, free_bits)


class CoherenceLoss(nn.Module):
    def __init__(self, r_b: int, beta: float):
        super().__init__()
        self.eps = 1e-12
        I = torch.diag(torch.ones(r_b))
        self.register_buffer("Iu", I)
        self.register_buffer("Iv", I)
        self.beta = beta
    
    def _coherence_penalty(self, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Coherence penalty with maximum safety.
        Avoid any complex operations that might cause recursion.
        """
        # 矩阵乘法
        UtU = torch.matmul(U.transpose(-2, -1), U)
        VVt = torch.matmul(V, V.transpose(-2, -1))

        # 直接计算差值的平方和，避免调用任何可能有问题的函数
        diff1 = UtU - self.Iu.to(UtU.device)
        diff2 = VVt - self.Iv.to(VVt.device)

        # 最直接的Frobenius范数计算
        frob1 = torch.sum(diff1 * diff1)
        frob2 = torch.sum(diff2 * diff2)

        return (frob1 + frob2) * self.beta

    def forward(
        self,
        U: torch.Tensor,
        V: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Straightforward forward pass without any complex context management.
        """

        coh_loss = self._coherence_penalty(U, V)

        return coh_loss

class TopkRouter(nn.Module):
    def __init__(
        self,
        r_b: int,
        topk: int,
        temperature: float = 1.0,
        gamma: float = 1e-4,
        tol: float = 0.05,
        update_on_train_only: bool = True,
        max_bias_abs: Optional[float] = None,
    ):
        super().__init__()
        assert topk >= 1, "topk must be >= 1"
        assert r_b >= topk, "r_b must be >= topk"
        self.r_b = int(r_b)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.tol = float(tol)
        self.update_on_train_only = bool(update_on_train_only)
        self.max_bias_abs = max_bias_abs

        # 使用 Parameter 但不需要梯度
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(r_b, dtype=torch.float32),
            requires_grad=False
        )

    def reset_bias(self):
        """Zero the bias buffer (call after module/device moves if needed)."""
        with torch.no_grad():
            self.e_score_correction_bias.zero_()

    @torch.no_grad()
    def _aux_free_bias_update(self, topk_indices: torch.Tensor, r_b: int):
        """
        Update the selection bias based on per-step expert loads.
        topk_indices: (B, r, topk) indices chosen in THIS forward.
        r_b: number of candidates.
        """
        # Compute per-candidate load for this step
        flat_idx = topk_indices.reshape(-1)
        loads = torch.bincount(flat_idx, minlength=r_b).to(self.e_score_correction_bias.dtype)

        # Target load: total selections / r_b
        total_selections = flat_idx.numel()
        target = total_selections / float(r_b)
        hi = target * (1.0 + self.tol)
        lo = target * (1.0 - self.tol)

        # Piecewise nudging: over-used -> decrease bias; under-used -> increase bias
        over = loads > hi
        under = loads < lo

        self.e_score_correction_bias[over] -= self.gamma
        self.e_score_correction_bias[under] += self.gamma

        # Optional safety clamp on bias magnitude
        if self.max_bias_abs is not None:
            torch.clamp_(self.e_score_correction_bias, min=-self.max_bias_abs, max=self.max_bias_abs)

    def forward(self, Pb: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Pb: (B, r, r_b), logits for candidates (higher -> more likely to be selected)
        V:  (r_b, f_out), candidate vectors
        return: B of shape (B, r, f_out)
        """
        assert Pb.dim() == 3, "Pb must be (B, r, r_b)"
        assert V.dim() == 2, "V must be (r_b, f_out)"
        Bsz, R, Rb = Pb.shape
        assert V.shape[0] == Rb, "Pb last dim (r_b) must match V.shape[0]"
        assert self.topk <= Rb, "topk cannot exceed r_b"

        # Use Pb + bias for selection ONLY (no grad influence on bias)
        # (we allow Pb dtype to be low-precision; compute selection/weights in float32 for stability)
        Pb_f32 = Pb.float()
        bias = self.e_score_correction_bias.to(Pb_f32.dtype).view(1, 1, Rb)
        scores_for_choice = Pb_f32 + bias

        # Top-k selection along last dim (r_b)
        # (non-differentiable w.r.t. the excluded items; standard practice for Top-k MoE-style routing)
        topk_vals, topk_indices = torch.topk(scores_for_choice, k=self.topk, dim=-1, largest=True, sorted=False)

        # Weights for the chosen candidates come from the ORIGINAL logits (without bias)
        chosen_logits = torch.gather(Pb_f32, dim=-1, index=topk_indices)  # (B, r, topk)
        if self.temperature != 1.0:
            chosen_logits = chosen_logits / self.temperature
        weights = F.softmax(chosen_logits, dim=-1)  # (B, r, topk), sums to 1 over topk

        # Gather selected candidate vectors: use index_select to avoid huge broadcast
        # V_sel: (B, r, topk, f_out)
        f_out = V.shape[1]
        flat_idx = topk_indices.reshape(-1)  # (B*r*topk,)
        V_sel = V.index_select(dim=0, index=flat_idx).view(Bsz, R, self.topk, f_out)

        # Weighted sum over topk -> (B, r, f_out)
        B_out = torch.sum(V_sel * weights.unsqueeze(-1), dim=2)

        # Aux-Loss-Free bias update (step-wise, no grad). Default: only in training mode.
        if (not self.update_on_train_only) or self.training:
            with torch.no_grad():
                self._aux_free_bias_update(topk_indices.detach(), r_b=Rb)

        # Cast back to V dtype (common in LoRA codepaths)
        return B_out.to(V.dtype)