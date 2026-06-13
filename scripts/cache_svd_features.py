"""
Pre-compute and cache frozen SVD + CLIP features for faster policy training.

Each sample is stored as  {cache_dir}/{idx}.pt  containing:
  - 'svd_feat'    : float16 tensor of shape (T, 2*L, C)
                    T = Former_num_time_embeds (16)
                    L = spatial tokens per view (H_feat * W_feat)
                    C = feature channels (2560 with use_all_layer + extract_layer_idx=1)
                    The 2*L dimension is static-camera tokens cat with gripper tokens.
  - 'latent_goal' : float16 tensor of shape (D,) — frozen CLIP language embedding

Storage estimate: ~42 MB per unique sample (float16, T=16, L=256, C=2560).
Run once; the cached training script skips SVD entirely each epoch.

Usage:
  accelerate launch scripts/cache_svd_features.py \\
      --cache_dir /mnt/localssd/vpp/cache \\
      --root_data_dir /mnt/localssd/calvin/task_ABC_D \\
      --video_model_path /mnt/localssd/vpp/weights/svd-robot-calvin \\
      --text_encoder_path /mnt/localssd/vpp/weights/clip-vit-base-patch32
"""
import argparse
import os
import sys
from pathlib import Path
import torch
import einops
from tqdm import tqdm
from accelerate import Accelerator

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", type=str, required=True,
                   help="Root directory where .pt cache files will be written.")
    p.add_argument("--config_name", type=str, default="VPP_Calvinabc_train")
    p.add_argument("--split", type=str, default="both",
                   choices=["train", "val", "both"])
    p.add_argument("--root_data_dir", type=str, default="")
    p.add_argument("--video_model_path", type=str, default="")
    p.add_argument("--text_encoder_path", type=str, default="")
    p.add_argument("--noise_seed", type=int, default=42,
                   help="Fixed RNG seed for SVD latent noise, ensuring deterministic features.")
    return p.parse_args()


@torch.no_grad()
def extract_one_batch(model, batch, device, noise_seed: int):
    """
    Run the frozen TVP encoder + language goal for a single batch.

    Returns
    -------
    svd_feats   : (B, T, 2*L, C) float16 tensor on CPU
    latent_goals: (B, D)         float16 tensor on CPU
    """
    rgb_static  = batch["rgb_obs"]["rgb_static"].to(device)   # (B, obs_seq_len, 3, H, W)
    rgb_gripper = batch["rgb_obs"]["rgb_gripper"].to(device)
    lang_text   = batch["lang_text"]

    # Language goal (frozen CLIP)
    latent_goal = model.language_goal(lang_text).to(rgb_static.dtype)  # (B, D)

    # SVD features — fix the random seed so latents are deterministic
    B = rgb_static.shape[0]
    input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)   # (2B, obs_seq_len, 3, H, W)
    language_doubled = list(lang_text) + list(lang_text)

    torch.manual_seed(noise_seed)
    raw_feat = model.TVP_encoder(
        input_rgb, language_doubled,
        model.timestep, model.extract_layer_idx,
        all_layer=model.use_all_layer,
        step_time=1, max_length=model.max_length,
    )  # (2B, num_svd_frames, C, H_feat, W_feat)

    # Apply the same einops as extract_predictive_feature
    num_frames = model.Former_num_time_embeds
    feat = einops.rearrange(raw_feat, "b f c h w -> b f c (h w)")
    feat = einops.rearrange(feat, "b f c l -> b f l c")
    feat = feat[:, :num_frames]                             # (2B, T, L, C)

    static_feat, gripper_feat = torch.split(feat, [B, B], dim=0)  # each (B, T, L, C)
    feat = torch.cat([static_feat, gripper_feat], dim=2)           # (B, T, 2L, C)
    feat = feat.float()                                            # promote for accuracy

    return feat.half().cpu(), latent_goal.half().cpu()


def cache_split(model, loader, split_dir: Path, device, noise_seed: int, desc: str):
    split_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc=desc, unit="batch"):
        indices = batch["idx"]
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        # Skip entirely-cached batches early
        uncached = [i for i, idx in enumerate(indices)
                    if not (split_dir / f"{idx}.pt").exists()]
        if not uncached:
            continue

        svd_feats, latent_goals = extract_one_batch(model, batch, device, noise_seed)

        for i in uncached:
            idx = indices[i]
            torch.save(
                {"svd_feat": svd_feats[i], "latent_goal": latent_goals[i]},
                split_dir / f"{idx}.pt",
            )


def main():
    args = parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    import hydra
    from hydra import compose, initialize

    with initialize(config_path="../policy_conf", job_name="cache_features"):
        cfg = compose(config_name=args.config_name)

    if args.root_data_dir:
        cfg.root_data_dir = args.root_data_dir
        cfg.datamodule.root_data_dir = args.root_data_dir
    if args.video_model_path:
        cfg.model.pretrained_model_path = args.video_model_path
    if args.text_encoder_path:
        cfg.model.text_encoder_path = args.text_encoder_path

    if accelerator.is_main_process:
        print(f"Cache directory: {args.cache_dir}")
        print(f"Noise seed:      {args.noise_seed}")

    # Load the full model (SVD included) for feature extraction
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    model.process_device()
    model.eval()

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()

    cache_root = Path(args.cache_dir)

    if args.split in ("train", "both"):
        # Use non-shuffled loader for deterministic ordering
        train_loader = datamodule.train_dataloader()["lang"]
        cache_split(model, train_loader, cache_root / "train", device,
                    args.noise_seed, desc="train")

    if args.split in ("val", "both"):
        val_loader = datamodule.val_dataloader()["lang"]
        cache_split(model, val_loader, cache_root / "val", device,
                    args.noise_seed, desc="val")

    if accelerator.is_main_process:
        # Report cache size
        total_bytes = sum(f.stat().st_size for f in cache_root.rglob("*.pt"))
        print(f"Total cache size: {total_bytes / 1e9:.2f} GB")
        n_files = sum(1 for _ in cache_root.rglob("*.pt"))
        print(f"Total files cached: {n_files}")


if __name__ == "__main__":
    main()
