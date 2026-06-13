"""Round-trip sanity check for the VPP HiVA coefficient codec.

Verifies decode is the exact inverse of encode (on the spline manifold) for both
rotation modes, with normalization off. Runs on CPU in a second, no data needed.

    python scripts/test_hiva_coeff_roundtrip.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from policy_models.hiva_coeff import HiVACoeffCodec


def check(rot_mode: str) -> bool:
    torch.manual_seed(0)
    codec = HiVACoeffCodec(
        action_dim=7, fit_horizon=10, k=8, degree=3,
        rot_mode=rot_mode, rot_eta=1.0, stats_mode="off",
        grip_min=-1.0, grip_max=1.0,
    ).eval()

    B, k = 4, codec.k
    # Random raw coefficients, small so decoded gripper stays inside [-1, 1].
    theta = torch.cat(
        [
            torch.randn(B, k, 3) * 0.05,   # tr
            torch.randn(B, k, 3) * 0.02,   # rot
            (torch.rand(B, k, 1) * 1.0 - 0.5),  # grip in ~[-0.5, 0.5]
        ],
        dim=-1,
    )

    # theta -> raw actions (decode) -> theta' (encode); should recover theta.
    actions = codec.decode(theta)                 # [B, H, 7]
    theta2 = codec.encode(actions)                # [B, k, 7]

    ok = True
    print(f"\n=== rot_mode={rot_mode} ===")
    print(f"{'block':>5} | {'rmse':>11} | {'max_abs':>11}")
    print("-" * 33)
    for name, sl in (("tr", slice(0, 3)), ("rot", slice(3, 6)), ("grip", slice(6, 7))):
        err = (theta[..., sl] - theta2[..., sl]).abs()
        rmse = err.pow(2).mean().sqrt().item()
        flag = "OK" if rmse < 1e-3 else "FAIL"
        ok = ok and rmse < 1e-3
        print(f"{name:>5} | {rmse:11.3e} | {err.max().item():11.3e}  {flag}")

    # action-space idempotency: decode(encode(actions)) == actions
    actions2 = codec.decode(theta2)
    act_rmse = (actions - actions2).pow(2).mean().sqrt().item()
    print(f"decoded action_rmse: {act_rmse:.3e}")
    return ok


def main() -> int:
    ok = check("flat") and check("so3")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
