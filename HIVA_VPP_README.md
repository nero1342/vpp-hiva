# HiVA in VPP (CALVIN) — coefficient action representation

Integrates HiVA into VPP by changing only the **action representation**: the EDM
diffusion head denoises HiVA B-spline coefficients `[B, k, 7]` instead of raw
action steps. VPP's video-conditioned diffusion backbone is reused unchanged, and
the offline "coefficient sidecar" is **not** needed — coefficients are fit in the
model from the raw action chunk.

All originals are untouched. Switch versions by changing the training config.

## New files

| File | Purpose |
|---|---|
| `policy_models/hiva_coeff.py` | `HiVACoeffCodec` — encode (actions→coeffs) / decode (coeffs→actions), B-spline basis + fixed fit matrix, coeff normalization. Self-contained, torch-only. |
| `policy_models/VPP_policy_hiva.py` | `VPP_HiVA_Policy(VPP_Policy)` — overrides only `training_step` (encode GT→coeff target) and `denoise_actions` (decode sampled coeff→actions). |
| `policy_conf/VPP_Calvinabc_train_hiva.yaml` | Training config; `_target_` → `VPP_HiVA_Policy`, adds `hiva_*` params. Action dims unchanged so pretrained dp-calvin weights load. |
| `step2_train_action_calvin_hiva.py` | Thin training launcher: imports the original `train(cfg)`, composes the `_hiva` config. |
| `policy_conf/calvin_evaluate_all_hiva.yaml` | Eval config; `_target_`→`VPP_HiVA_Policy`, `hiva_*` params **must match training**. |
| `policy_evaluation/calvin_evaluate_hiva.py` | Thin eval launcher: imports the original `calvin_evaluate.main`, composes the `_hiva` eval config. |
| `scripts/test_hiva_coeff_roundtrip.py` | CPU round-trip test (encode∘decode); passes at ~1e-8 for both rot modes. |

## How to switch / train

Same CLI as the original; just use the HiVA launcher (which composes
`VPP_Calvinabc_train_hiva` and reuses the original `train(cfg)`):

```bash
# original VPP:
python step2_train_action_calvin.py      --video_model_path ... --text_encoder_path ... --root_data_dir ...
# HiVA version (new launcher + config, originals untouched):
python step2_train_action_calvin_hiva.py --video_model_path ... --text_encoder_path ... --root_data_dir ...
```

`step2_train_action_calvin_hiva.py` imports `train` from the original script and
only swaps the composed config name, so there is no duplicated training logic and
nothing in the original files changes.

## Evaluation / inference

```bash
# original VPP eval:
python policy_evaluation/calvin_evaluate.py      --video_model_path ... --action_model_folder <ckpt> --clip_model_path ... --calvin_abc_dir ...
# HiVA eval (new launcher + config):
python policy_evaluation/calvin_evaluate_hiva.py --video_model_path ... --action_model_folder <hiva_ckpt> --clip_model_path ... --calvin_abc_dir ...
```

Inference path (all inherited except the decode step):
`model.step(obs, goal)` → `eval_forward` → `denoise_actions` *(overridden)* samples
coefficients `[B,10,7]` → `codec.decode` → raw action chunk `[B,10,7]` → normal
action chunking executes them. So `step()` returns ordinary CALVIN actions and the
rollout loop is unchanged.

- **The eval config's `hiva_*` params must match the training config.** The codec's
  basis/fit matrix and the coefficient normalization stats (`run_mean`/`run_var`,
  persistent buffers) are restored from the checkpoint via the eval script's
  existing `load_state_dict(..., strict=False)`.
- For multi-GPU eval, the same one-line config swap applies — import
  `calvin_evaluate_multi.main` in the launcher instead of `calvin_evaluate.main`.

## How it works

- **Training** (`training_step`): GT action chunk `[B,10,7]` → `codec.encode` →
  coefficients `[B,8,7]` → padded to the denoiser's `action_seq_len` (10) → used
  as the EDM score-matching **target**. The existing `diffusion_loss` runs
  unchanged on coefficients.
- **Inference/validation** (`denoise_actions`): diffusion samples coefficients
  `[B,10,7]` → take first `k` → `codec.decode` → action chunk `[B,10,7]` → VPP's
  normal `step()` chunking executes them.
- **Fit = inverse of decode.** `theta = M @ y`, `M = (φᵀφ+ridge·I)⁻¹φᵀ` (precomputed
  buffer); decode evaluates `φ @ theta` and differences it back to deltas. Round-trip
  is exact by construction (verified).

## Horizons: F (fit) vs k (control points) vs H (execute)

Three independent horizons (LP-MT style — fit/preview longer than you execute):

| symbol | meaning | config | default |
|---|---|---|---|
| **F** | spline fit / preview horizon (steps the B-spline spans) | `model.hiva_fit_horizon` **and** top-level `act_seq_len` | 15 |
| **k** | B-spline control points (≤ denoiser `action_seq_len`) | `model.hiva_k` | 8 |
| **H** | steps actually executed before replanning (receding horizon) | `model.multistep` | 10 |
| T | denoiser token count (coeffs padded to it; kept = pretrained) | `model.action_seq_len` | 10 |

**F=15, k=8, execute first H=10** is the shipped default:
- **Training** fits coefficients over **15-step** GT chunks (`act_seq_len=15` makes
  `ExtendedDiskDataset` load 15 future actions; `hiva_fit_horizon=15` fits over them).
- **Denoiser** stays at `action_seq_len=10` tokens (decoupled from F via an explicit
  literal, not `${act_seq_len}`), so the pretrained dp-calvin weights still load; the
  `k=8` coeff tokens are padded to 10.
- **Inference** decodes the full **15**-step spline, and `step()` executes only the
  **first `multistep=10`** before replanning — i.e. the last `F−H=5` preview steps are
  regenerated each chunk. Enforced by an assert `H ≤ F` in the policy.

**Hard constraint:** `hiva_fit_horizon` (F) must equal the dataset action window
(top-level `act_seq_len`). If they differ, `encode` would hold-pad/truncate the chunk
to F and the fit would be on synthetic data. Keep them in sync in both train and eval
configs.

## Config knobs (`model.hiva_*`)

| key | default | meaning |
|---|---|---|
| `hiva_k` | `8` | B-spline control points (≤ `action_seq_len`) |
| `hiva_fit_horizon` | `10` | action chunk length fit over (= dataset action window) |
| `hiva_degree` | `3` | cubic B-spline |
| `hiva_rot_mode` | `flat` | `flat` (per-channel cumsum; CALVIN Euler deltas) or `so3` (axis-angle composition) |
| `hiva_rot_eta` | `1.0` | scale for `so3` mode only |
| `hiva_ridge` | `0.0` | ridge for the fit (`0` → pseudo-inverse) |
| `hiva_grip_min/max` | `-1.0 / 1.0` | gripper clamp on decode |
| `hiva_coeff_stats_mode` | `running` | coeff normalization: `running` / `buffer` / `off` |
| `hiva_running_momentum` | `0.01` | EMA momentum for `running` |

## Weights & Biases logging

`train()` in `step2_train_action_calvin.py` now supports wandb, gated by
`cfg.use_wandb` (default **off** — original behavior is unchanged when absent).
The HiVA config enables it (`use_wandb: true`, `run_name`, `wandb_mode`); it logs
`train/total_loss`, `train/steps_per_sec`, `train/lr`, `val/validation_loss`, and
`hiva/coeff_std_mean` (coefficient normalization scale). Project/entity come from
the existing `logger` config block. Set `wandb_mode: offline|disabled` to skip
network/login. To log the *original* VPP run too, add `use_wandb: true` to
`VPP_Calvinabc_train.yaml`.

## Important notes

- **Keep coefficient normalization on.** Raw coefficients live on the *cumulative*
  trajectory, so their scale differs from raw actions; VPP's EDM (`sigma_data=0.5`)
  assumes ~unit-scaled targets. `hiva_coeff_stats_mode=running` rescales coeffs to
  unit variance. `off` is for the round-trip test, not training.
- **Rotation mode.** Default `flat` matches CALVIN's Euler-angle relative deltas and
  is exactly invertible. `so3` is provided for axis-angle envs / paper fidelity.
- **Spline fidelity sets the inference ceiling.** Even a perfect diffusion model can
  only reproduce actions up to `decode(encode(actions))` error — the B-spline fit of
  the real 10-step chunk with `k` control points. `decode∘encode` is exact for
  spline-representable signals (round-trip test ≈1e-8) but lossy for arbitrary ones.
  Measure this floor on **real CALVIN trajectories** (not random actions): if the
  reconstruction RMSE is too high, raise `hiva_k` (toward `fit_horizon`) or lower
  `hiva_degree`. This is the main quality knob to check before a full training run.
- **No duration head.** Fixed `fit_horizon` = the existing 10-step chunk; no duration
  segmentation needed (CALVIN has none).
- **Pretrained weights.** `action_seq_len`/`action_dim` are unchanged, so the
  dp-calvin checkpoint loads via the existing `strict=False` path; only the meaning
  of the action tokens shifts to coefficients. The denoiser's extra
  `n_tokens - k` tokens are trained to predict ~0 and ignored on decode.
- **Eval.** `policy_evaluation/calvin_evaluate*.py` call `model.step(obs, goal)`,
  which now returns decoded raw actions — no eval change needed.
```
