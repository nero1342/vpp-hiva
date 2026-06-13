"""HiVA B-spline coefficient codec for VPP (CALVIN).

This is the self-contained core that lets VPP's diffusion action head operate on
HiVA B-spline *coefficients* instead of raw action steps, without touching any
original VPP file.

Representation
--------------
A raw action chunk ``[B, H, 7]`` (CALVIN relative actions
``[dx, dy, dz, drx, dry, drz, gripper]``) is converted to ``k`` B-spline control
points per modality:

    translation : 3 channels, fit on the cumulative trajectory (cumsum of deltas)
    rotation    : 3 channels, fit on the cumulative trajectory
                  - "flat" (default): per-channel cumsum (Euler-delta friendly)
                  - "so3": cumulative SO(3) composition (axis-angle, Evo1-style)
    gripper     : 1 channel, fit directly on the command value

Stacked, the coefficients are ``[B, k, 7]`` -- the same rank as VPP's action
chunk, so VPP's diffusion transformer can denoise them unchanged.

The fit is a fixed linear projection ``theta = M @ y`` with
``M = (phi^T phi + ridge I)^-1 phi^T`` (precomputed buffer), and ``decode`` is its
exact inverse (see ``scripts/test_hiva_coeff_roundtrip.py``).

Nothing here requires grad: ``encode`` produces training *targets*, ``decode`` is
used only at inference/validation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# B-spline basis (pure-python, no scipy dependency).
# -----------------------------------------------------------------------------

def clamped_bspline_basis(d: int, *, n_ctrl: int, degree: int) -> torch.Tensor:
    """Open-uniform clamped B-spline basis evaluated at integer timesteps -> [d, n_ctrl]."""
    if d < 2:
        raise ValueError(f"horizon must be >= 2 for B-spline basis, got {d}.")
    if n_ctrl < degree + 1:
        raise ValueError(f"n_ctrl={n_ctrl} must be at least degree+1={degree + 1}.")

    x_values = [float(x) for x in range(d)]
    internal_count = n_ctrl - degree + 1
    if internal_count <= 1:
        t_internal = [0.0]
    else:
        t_internal = [(d - 1) * i / (internal_count - 1) for i in range(internal_count)]
    knots = [0.0] * degree + t_internal + [float(d - 1)] * degree

    def basis_one(i: int, p: int, x: float) -> float:
        if p == 0:
            left, right = knots[i], knots[i + 1]
            if left <= x < right:
                return 1.0
            if x == knots[-1] and left <= x <= right:
                return 1.0
            return 0.0
        left_den = knots[i + p] - knots[i]
        right_den = knots[i + p + 1] - knots[i + 1]
        left_term = 0.0
        right_term = 0.0
        if left_den > 0:
            left_term = (x - knots[i]) / left_den * basis_one(i, p - 1, x)
        if right_den > 0:
            right_term = (knots[i + p + 1] - x) / right_den * basis_one(i + 1, p - 1, x)
        return left_term + right_term

    rows = [[basis_one(i, degree, x) for i in range(n_ctrl)] for x in x_values]
    return torch.tensor(rows, dtype=torch.float32)


# -----------------------------------------------------------------------------
# SO(3) helpers (only used when rot_mode="so3").
# -----------------------------------------------------------------------------

def _skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    theta2 = theta * theta
    small = theta < 1e-4
    safe = theta.clamp_min(1e-6)
    safe2 = safe * safe
    a = torch.where(small, 1 - theta2 / 6 + theta2 * theta2 / 120, torch.sin(theta) / safe)
    b = torch.where(small, 0.5 - theta2 / 24 + theta2 * theta2 / 720, (1 - torch.cos(theta)) / safe2)
    k = _skew(rotvec)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(*rotvec.shape[:-1], 3, 3)
    return eye + a.unsqueeze(-1) * k + b.unsqueeze(-1) * (k @ k)


def matrix_to_rotvec(matrix: torch.Tensor) -> torch.Tensor:
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    omega = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    cos_theta = ((trace - 1) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    sin_theta = 0.5 * torch.linalg.norm(omega, dim=-1)
    theta = torch.atan2(sin_theta, cos_theta)
    theta2 = theta * theta
    small = theta < 1e-4
    scale = torch.where(small, 0.5 + theta2 / 12 + 7 * theta2 * theta2 / 720, theta / (2 * sin_theta.clamp_min(1e-7)))
    return scale.unsqueeze(-1) * omega


# -----------------------------------------------------------------------------
# Codec.
# -----------------------------------------------------------------------------

class HiVACoeffCodec(nn.Module):
    """Encode raw action chunks <-> HiVA B-spline coefficients.

    Args:
        action_dim:   raw action width (CALVIN = 7).
        fit_horizon:  number of action steps H the spline is fit over (= action chunk length).
        k:            number of B-spline control points per modality.
        degree:       B-spline degree (cubic = 3).
        rot_mode:     "flat" (per-channel cumsum) or "so3" (cumulative SO(3) compose).
        rot_eta:      scale used by so3 mode (R_t = R_{t-1} Exp(eta * a)); ignored when flat.
        ridge:        ridge for the lstsq fit (0 -> pseudo-inverse).
        grip_min/max: gripper clamp range on decode (CALVIN gripper in [-1, 1]).
        stats_mode:   coefficient normalization: "running" | "buffer" | "off".
        running_momentum: EMA momentum for "running" stats.
    """

    def __init__(
        self,
        action_dim: int = 7,
        fit_horizon: int = 10,
        k: int = 8,
        degree: int = 3,
        rot_mode: str = "flat",
        rot_eta: float = 1.0,
        ridge: float = 0.0,
        grip_min: float = -1.0,
        grip_max: float = 1.0,
        stats_mode: str = "running",
        running_momentum: float = 0.01,
        eps: float = 1e-6,
    ):
        super().__init__()
        if action_dim != 7:
            raise ValueError(f"HiVACoeffCodec assumes 7D [xyz, rot3, grip] actions; got action_dim={action_dim}.")
        if rot_mode not in {"flat", "so3"}:
            raise ValueError(f"rot_mode must be 'flat' or 'so3'; got {rot_mode!r}.")
        if stats_mode not in {"running", "buffer", "off"}:
            raise ValueError(f"stats_mode must be 'running', 'buffer', or 'off'; got {stats_mode!r}.")

        self.action_dim = action_dim
        self.fit_horizon = int(fit_horizon)
        self.k = int(k)
        self.degree = int(degree)
        self.rot_mode = rot_mode
        self.rot_eta = float(rot_eta)
        self.ridge = float(ridge)
        self.grip_min = float(grip_min)
        self.grip_max = float(grip_max)
        self.stats_mode = stats_mode
        self.running_momentum = float(running_momentum)
        self.eps = float(eps)

        phi = clamped_bspline_basis(self.fit_horizon, n_ctrl=self.k, degree=self.degree)  # [H, k]
        self.register_buffer("phi", phi, persistent=False)
        self.register_buffer("fit_matrix", self._make_fit_matrix(phi), persistent=False)  # [k, H]

        # Fixed-buffer normalization stats (default identity; "buffer" mode uses these).
        self.register_buffer("coeff_mean", torch.zeros(1, 1, action_dim), persistent=True)
        self.register_buffer("coeff_std", torch.ones(1, 1, action_dim), persistent=True)
        # Running normalization stats ("running" mode).
        self.register_buffer("run_mean", torch.zeros(1, 1, action_dim), persistent=True)
        self.register_buffer("run_var", torch.ones(1, 1, action_dim), persistent=True)
        self.register_buffer("run_count", torch.zeros((), dtype=torch.long), persistent=True)

    # -- fit matrix -----------------------------------------------------------

    def _make_fit_matrix(self, phi: torch.Tensor) -> torch.Tensor:
        phi = phi.to(dtype=torch.float32)
        if self.ridge > 0:
            k = phi.shape[1]
            lhs = phi.transpose(-1, -2) @ phi + self.ridge * torch.eye(k, dtype=phi.dtype)
            return torch.linalg.solve(lhs, phi.transpose(-1, -2))
        return torch.linalg.pinv(phi)

    # -- normalization --------------------------------------------------------

    @torch.no_grad()
    def _update_running_stats(self, coeff_raw: torch.Tensor) -> None:
        m = self.running_momentum
        batch_mean = coeff_raw.mean(dim=(0, 1), keepdim=True)
        batch_var = coeff_raw.var(dim=(0, 1), unbiased=False, keepdim=True)
        if self.run_count.item() == 0:
            self.run_mean.copy_(batch_mean)
            self.run_var.copy_(batch_var)
        else:
            self.run_mean.mul_(1.0 - m).add_(m * batch_mean)
            self.run_var.mul_(1.0 - m).add_(m * batch_var)
        self.run_count += 1

    def _mean_std(self, device: torch.device, dtype: torch.dtype):
        if self.stats_mode == "off":
            return None, None
        if self.stats_mode == "buffer":
            return self.coeff_mean.to(device=device, dtype=dtype), self.coeff_std.to(device=device, dtype=dtype)
        mean = self.run_mean.to(device=device, dtype=dtype)
        std = self.run_var.to(device=device, dtype=dtype).sqrt().clamp_min(self.eps)
        return mean, std

    def normalize(self, coeff_raw: torch.Tensor) -> torch.Tensor:
        mean, std = self._mean_std(coeff_raw.device, coeff_raw.dtype)
        if mean is None:
            return coeff_raw
        return (coeff_raw - mean) / std

    def unnormalize(self, coeff_norm: torch.Tensor) -> torch.Tensor:
        mean, std = self._mean_std(coeff_norm.device, coeff_norm.dtype)
        if mean is None:
            return coeff_norm
        return coeff_norm * std + mean

    # -- modality (de)composition --------------------------------------------

    def _prepare_window(self, actions_7d: torch.Tensor) -> torch.Tensor:
        """Slice (or terminal-hold pad) raw actions to ``fit_horizon`` steps."""
        B, T, C = actions_7d.shape
        H = self.fit_horizon
        if T == H:
            return actions_7d
        if T > H:
            return actions_7d[:, :H, :]
        pad = actions_7d.new_zeros(B, H - T, C)  # tr/rot deltas zero in synthetic tail
        pad[..., 6] = actions_7d[:, -1:, 6]       # hold gripper command
        return torch.cat([actions_7d, pad], dim=1)

    def _cumulative_rotation_so3(self, raw_rot: torch.Tensor) -> torch.Tensor:
        B, T, _ = raw_rot.shape
        deltas = rotvec_to_matrix((self.rot_eta * raw_rot).reshape(B * T, 3)).reshape(B, T, 3, 3)
        cur = torch.eye(3, dtype=raw_rot.dtype, device=raw_rot.device).unsqueeze(0).repeat(B, 1, 1)
        outs = []
        for t in range(T):
            cur = cur @ deltas[:, t]
            outs.append(matrix_to_rotvec(cur))
        return torch.stack(outs, dim=1)

    def _decode_rotation_so3(self, rho_hat: torch.Tensor) -> torch.Tensor:
        B, T, _ = rho_hat.shape
        rot_mats = rotvec_to_matrix(rho_hat.reshape(B * T, 3)).reshape(B, T, 3, 3)
        eye = torch.eye(3, dtype=rho_hat.dtype, device=rho_hat.device)
        prev = torch.cat([eye.expand(B, 1, 3, 3), rot_mats[:, :-1]], dim=1)
        delta = prev.transpose(-1, -2) @ rot_mats
        raw = matrix_to_rotvec(delta.reshape(B * T, 3, 3)).reshape(B, T, 3)
        return raw / self.rot_eta

    # -- public API -----------------------------------------------------------

    @torch.no_grad()
    def encode(self, actions: torch.Tensor, update_stats: bool = False) -> torch.Tensor:
        """raw actions [B, T>=H, 7] -> normalized coefficients [B, k, 7]."""
        a = self._prepare_window(actions[..., :7].to(dtype=torch.float32))
        m = self.fit_matrix.to(device=a.device, dtype=torch.float32)

        tr_cum = torch.cumsum(a[..., 0:3], dim=1)
        if self.rot_mode == "so3":
            rot_cum = self._cumulative_rotation_so3(a[..., 3:6])
        else:
            rot_cum = torch.cumsum(a[..., 3:6], dim=1)
        grip = a[..., 6:7]

        theta_tr = torch.einsum("kh,bhc->bkc", m, tr_cum)
        theta_rot = torch.einsum("kh,bhc->bkc", m, rot_cum)
        theta_grip = torch.einsum("kh,bhc->bkc", m, grip)
        coeff_raw = torch.cat([theta_tr, theta_rot, theta_grip], dim=-1)  # [B, k, 7]

        if update_stats and self.stats_mode == "running":
            self._update_running_stats(coeff_raw)
        return self.normalize(coeff_raw).to(dtype=actions.dtype)

    @torch.no_grad()
    def decode(self, coeff_norm: torch.Tensor) -> torch.Tensor:
        """normalized coefficients [B, k, 7] -> raw actions [B, H, 7]."""
        coeff_raw = self.unnormalize(coeff_norm[:, : self.k, :].to(dtype=torch.float32))
        phi = self.phi.to(device=coeff_raw.device, dtype=torch.float32)
        theta_tr, theta_rot, theta_grip = coeff_raw[..., 0:3], coeff_raw[..., 3:6], coeff_raw[..., 6:7]

        tr_cum_hat = torch.einsum("hk,bkc->bhc", phi, theta_tr)
        rot_cum_hat = torch.einsum("hk,bkc->bhc", phi, theta_rot)
        grip_hat = torch.einsum("hk,bkc->bhc", phi, theta_grip).clamp(self.grip_min, self.grip_max)

        B = coeff_raw.shape[0]
        zero3 = tr_cum_hat.new_zeros(B, 1, 3)
        tr_delta = torch.diff(torch.cat([zero3, tr_cum_hat], dim=1), dim=1)
        if self.rot_mode == "so3":
            rot_delta = self._decode_rotation_so3(rot_cum_hat)
        else:
            rot_delta = torch.diff(torch.cat([zero3, rot_cum_hat], dim=1), dim=1)

        actions = torch.cat([tr_delta, rot_delta, grip_hat], dim=-1)  # [B, H, 7]
        return actions.to(dtype=coeff_norm.dtype)

    def pad_to_tokens(self, coeff: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """Pad k coefficient tokens up to the denoiser's action_seq_len (zeros)."""
        if coeff.shape[1] >= n_tokens:
            return coeff[:, :n_tokens, :]
        return F.pad(coeff, (0, 0, 0, n_tokens - coeff.shape[1]))
