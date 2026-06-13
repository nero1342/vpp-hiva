"""HiVA training launcher for VPP on CALVIN.

Thin wrapper around ``step2_train_action_calvin.train`` that composes the HiVA
config (``VPP_Calvinabc_train_hiva``) instead of the default one, so the original
training script is left untouched. Use exactly like ``step2_train_action_calvin.py``:

    python step2_train_action_calvin_hiva.py \
        --video_model_path ... --text_encoder_path ... --root_data_dir ...
"""

import os

import torch

from step2_train_action_calvin import train

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    os.environ["TOKENIZERS_PARALLELISM"] = "True"

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_model_path", type=str, default="")
    parser.add_argument("--text_encoder_path", type=str, default="")
    parser.add_argument("--root_data_dir", type=str, default="")
    args = parser.parse_args()

    from hydra import compose, initialize

    with initialize(config_path="./policy_conf", job_name="VPP_Calvinabc_train_hiva"):
        cfg = compose(config_name="VPP_Calvinabc_train_hiva")
    if args.video_model_path:
        cfg.model.pretrained_model_path = args.video_model_path
    if args.text_encoder_path:
        cfg.model.text_encoder_path = args.text_encoder_path
    if args.root_data_dir:
        cfg.root_data_dir = args.root_data_dir
        cfg.datamodule.root_data_dir = args.root_data_dir
    train(cfg)
