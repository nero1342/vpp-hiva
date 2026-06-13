"""VPP policy with HiVA B-spline coefficient action representation.

`VPP_HiVA_Policy` subclasses the original `VPP_Policy` and changes only the
*action representation*: the EDM diffusion head denoises HiVA B-spline
coefficients ``[B, k, 7]`` (padded to the denoiser's ``action_seq_len``) instead
of raw action steps. The entire video-conditioned diffusion backbone
(`TVP_encoder`, `Video_Former`, `GCDenoiser`) is reused unchanged, so VPP's
pretrained dp-calvin weights still load.

Boundaries that change:
  * training: ground-truth action chunk -> `codec.encode` -> coefficient target,
    then the existing EDM score-matching loss runs on coefficients.
  * inference/validation: diffusion samples coefficients -> `codec.decode` ->
    raw action chunk, then VPP's normal action chunking (`step`) executes them.

Nothing in the original `VPP_policy.py` is modified. Switch versions by changing
the config `_target_` (see `policy_conf/VPP_Calvinabc_train_hiva.yaml`).
"""

import logging
from typing import Dict, Optional, Tuple

import torch

from policy_models.VPP_policy import VPP_Policy
from policy_models.hiva_coeff import HiVACoeffCodec

logger = logging.getLogger(__name__)


class VPP_HiVA_Policy(VPP_Policy):
    def __init__(
        self,
        *args,
        hiva_k: int = 8,
        hiva_fit_horizon: int = 10,
        hiva_degree: int = 3,
        hiva_rot_mode: str = "flat",
        hiva_rot_eta: float = 1.0,
        hiva_ridge: float = 0.0,
        hiva_grip_min: float = -1.0,
        hiva_grip_max: float = 1.0,
        hiva_coeff_stats_mode: str = "running",
        hiva_running_momentum: float = 0.01,
        **kwargs,
    ):
        # The denoiser keeps its original action_seq_len / action_dim (from kwargs),
        # so pretrained weights load; we only reinterpret the tokens as coefficients.
        super().__init__(*args, **kwargs)

        self.hiva_k = int(hiva_k)
        # Number of coefficient tokens the diffusion model carries (= its action_seq_len).
        self.hiva_n_tokens = int(self.action_seq_len) if hasattr(self, "action_seq_len") else int(kwargs.get("action_seq_len", self.act_window_size))
        if self.hiva_k > self.hiva_n_tokens:
            raise ValueError(
                f"hiva_k={self.hiva_k} cannot exceed the denoiser action_seq_len={self.hiva_n_tokens}."
            )

        self.hiva = HiVACoeffCodec(
            action_dim=self.action_dim,
            fit_horizon=hiva_fit_horizon,
            k=hiva_k,
            degree=hiva_degree,
            rot_mode=hiva_rot_mode,
            rot_eta=hiva_rot_eta,
            ridge=hiva_ridge,
            grip_min=hiva_grip_min,
            grip_max=hiva_grip_max,
            stats_mode=hiva_coeff_stats_mode,
            running_momentum=hiva_running_momentum,
        )
        # Receding horizon: decode produces F (=fit_horizon) steps; step() executes
        # the first `multistep` (H) of them, then replans. Require H <= F.
        exec_horizon = int(getattr(self, "multistep", hiva_fit_horizon))
        if exec_horizon > hiva_fit_horizon:
            raise ValueError(
                f"multistep (execute={exec_horizon}) must be <= hiva_fit_horizon (F={hiva_fit_horizon}); "
                "you cannot execute more steps than the spline is fit/decoded over."
            )
        logger.info(
            "VPP_HiVA_Policy: F(fit)=%d k(ctrl)=%d H(exec/multistep)=%d n_tokens=%d rot_mode=%s stats_mode=%s",
            hiva_fit_horizon, self.hiva_k, exec_horizon, self.hiva_n_tokens, hiva_rot_mode, hiva_coeff_stats_mode,
        )

    # -- training: fit coefficient targets from raw actions -------------------

    def training_step(self, dataset_batch: Dict[str, Dict]) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.device)

        predictive_feature, latent_goal = self.extract_predictive_feature(dataset_batch)

        actions = dataset_batch["actions"].to(self.device)            # [B, H, 7]
        coeff = self.hiva.encode(actions, update_stats=self.training)  # [B, k, 7]
        coeff_target = self.hiva.pad_to_tokens(coeff, self.hiva_n_tokens)  # [B, n_tokens, 7]

        act_loss, _sigmas, _noise = self.diffusion_loss(predictive_feature, latent_goal, coeff_target)

        total_loss = total_loss + act_loss
        self._log_training_metrics(act_loss, total_loss, actions.shape[0])
        return total_loss

    # -- inference/validation: decode coefficients back to actions ------------

    def denoise_actions(
        self,
        latent_plan: torch.Tensor,
        perceptual_emb: torch.Tensor,
        latent_goal: torch.Tensor,
        inference: Optional[bool] = False,
        extra_args={},
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Parent samples a [B, act_window_size(=n_tokens), 7] coefficient tensor.
        coeff = super().denoise_actions(latent_plan, perceptual_emb, latent_goal, inference, extra_args)
        actions = self.hiva.decode(coeff)  # [B, H, 7]
        return actions
