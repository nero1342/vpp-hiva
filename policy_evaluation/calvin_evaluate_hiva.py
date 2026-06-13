"""HiVA CALVIN evaluation launcher.

Thin wrapper around ``calvin_evaluate.main`` that composes the HiVA eval config
(``calvin_evaluate_all_hiva.yaml`` -> instantiates ``VPP_HiVA_Policy``) instead of
the default one, so the original eval script is left untouched. Same CLI as
``calvin_evaluate.py``:

    python policy_evaluation/calvin_evaluate_hiva.py \
        --video_model_path ... --action_model_folder <hiva_ckpt_dir> \
        --clip_model_path ... --calvin_abc_dir ...

The HiVA action head decodes sampled coefficients back to raw actions inside
``VPP_HiVA_Policy.denoise_actions``, so ``model.step(obs, goal)`` returns ordinary
CALVIN actions and the rollout loop is unchanged. The codec's coefficient
normalization stats are restored from the checkpoint (persistent buffers).

For the multi-GPU version, import ``calvin_evaluate_multi.main`` instead of
``calvin_evaluate.main`` below — everything else is identical.
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_evaluation.calvin_evaluate_multi import main

if __name__ == "__main__":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    from hydra import compose, initialize
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_model_path", type=str, default="")
    parser.add_argument("--action_model_folder", type=str, default="")
    parser.add_argument("--clip_model_path", type=str, default="")
    parser.add_argument("--calvin_abc_dir", type=str, default="")
    args = parser.parse_args()

    with initialize(config_path="../policy_conf", job_name="calvin_evaluate_all_hiva"):
        cfg = compose(config_name="calvin_evaluate_all_hiva.yaml")
    if args.video_model_path:
        cfg.model.pretrained_model_path = args.video_model_path
    if args.action_model_folder:
        cfg.train_folder = args.action_model_folder
    if args.clip_model_path:
        cfg.model.text_encoder_path = args.clip_model_path
    if args.calvin_abc_dir:
        cfg.root_data_dir = args.calvin_abc_dir
    main(cfg)
