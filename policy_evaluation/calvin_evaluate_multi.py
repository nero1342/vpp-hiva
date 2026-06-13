"""Multi-GPU CALVIN evaluation.

This is a parallelized variant of ``calvin_evaluate.py``. It keeps the exact
CALVIN model / env / language-embedding interface used by ``calvin_evaluate.py``
(``get_default_beso_and_env``, ``model.step(obs, goal)``, ``model.reset()``,
``lang_embeddings.get_lang_goal(...)``) but borrows the parallel-inference and
progress-reporting machinery from ``dawn_evaluate.py``:

  * ``accelerate`` shards the evaluation sequences across all available GPUs
    (one independent model + env per process; no DDP wrapping needed since this
    is data-parallel *inference*, not training).
  * ``rich.progress`` shows a global "sequences" bar plus a live per-subtask
    rollout bar for every sequence being evaluated.
  * results are collected back to the main process with ``gather_object``.

Launch with ``accelerate``, e.g.::

    accelerate launch --num_processes 4 policy_evaluation/calvin_evaluate_multi.py \
        --video_model_path ... --action_model_folder ... \
        --clip_model_path ... --calvin_abc_dir ...
"""

from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import hydra
import numpy as np
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from torch.utils.data import DataLoader
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import accelerate
from accelerate.utils import gather_object

from policy_evaluation.multistep_sequences import get_sequences
from policy_evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition, join_vis_lang
from policy_models.utils.utils import get_last_checkpoint
from policy_models.rollout.rollout_video import RolloutVideo

logger = logging.getLogger(__name__)

SEQ_LEN = 5


class ListDataset(torch.utils.data.Dataset):
    """Wraps the eval sequences so the accelerate-prepared DataLoader can shard
    them across processes while preserving each sequence's global index."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (idx, self.data[idx])


def get_video_tag(i):
    # ``i`` is the global sequence index (preserved through ListDataset), so it
    # is already unique across processes -- no rank offset needed.
    return f"_long_horizon/sequence_{i}"


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")

    log_dir = log_dir / "logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir, exist_ok=True)
    print(f"logging to {log_dir}")
    return log_dir


def count_success(results, seq_len=SEQ_LEN):
    count = Counter(results)
    step_success = []
    for i in range(1, seq_len + 1):
        n_success = sum(count[j] for j in reversed(range(i, seq_len + 1)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(results, sequences, cfg, log_dir=None):
    """Aggregate the gathered ``results`` / ``sequences`` and write results.json.

    Unlike the single-GPU script which re-derives the sequences via
    ``get_sequences``, the multi-GPU path gathers the sequences alongside the
    results so the zip stays correct regardless of cross-process ordering.
    """
    if log_dir is None:
        log_dir = get_log_dir(cfg.train_folder)

    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results, seq_len=SEQ_LEN))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, sequence in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    data = {"avg_seq_len": float(avg_seq_len), "chain_sr": chain_sr, "task_info": task_info}

    # if cfg.log_wandb:
    #     import wandb
    #     wandb.log({
    #         "avrg_performance/avg_seq_len": avg_seq_len,
    #         "avrg_performance/chain_sr": chain_sr,
    #         "detailed_metrics/task_info": task_info,
    #     })

    with open(os.path.join(log_dir, "results.json"), "w") as file:
        json.dump(data, file, indent=2)
    print(f"Saved results to {os.path.join(log_dir, 'results.json')}")


def evaluate_policy(model, env, lang_embeddings, cfg, accelerator, num_videos=0, save_dir=None):
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations

    # video stuff
    if num_videos > 0:
        rollout_video = RolloutVideo(
            logger=logger,
            empty_cache=False,
            log_to_file=True,
            save_dir=save_dir,
            resolution_scale=1,
        )
    else:
        rollout_video = None

    eval_sequences = get_sequences(cfg.num_sequences)

    # Shard the sequences across processes. accelerator.prepare splits the
    # DataLoader so each GPU only iterates over its own slice.
    dataset = ListDataset(eval_sequences)
    eval_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
    )
    dataloader = accelerator.prepare(eval_loader)

    device = next(model.parameters()).device

    progress = Progress(
        TextColumn("{task.description}"),
        SpinnerColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        disable=not accelerator.is_main_process,
    )
    progress.start()

    results = []
    sequences = []
    plans = defaultdict(list)

    eval_task = progress.add_task(
        f"[{device}] Evaluating sequences", total=len(dataloader)
    )

    for i, data in enumerate(dataloader):
        idx, (initial_state, eval_sequence) = data[0]
        start_time = time.time()
        record = idx < num_videos

        task_sequence = " -> ".join(eval_sequence)
        logger.info(f"[{device}] sequence {idx} ({i + 1}/{len(dataloader)}): {task_sequence}")

        result = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, lang_embeddings,
            val_annotations, cfg, progress, record, rollout_video, idx,
        )
        end_time = time.time()

        results.append(result)
        sequences.append(list(eval_sequence))

        if record:
            rollout_video.write_to_tmp()
            # if result < 4:
            rollout_video._log_currentvideos_to_file2(idx, result, save_as_video=True)

        # Per-process running success summary.
        success_rates = count_success(results)
        average_rate = sum(success_rates) / len(success_rates) * SEQ_LEN
        description = f"[{device}] " + " ".join(
            [f"{j + 1}/{SEQ_LEN} : {v * 100:.1f}% |" for j, v in enumerate(success_rates)]
        )
        description += f" Average: {average_rate:.2f} | Time: {end_time - start_time:.1f}s |"
        logger.info(description)

        progress.update(eval_task, advance=1)

    progress.stop()

    logger.info(f"[{device}] local evaluation done; waiting for all processes...")
    accelerator.wait_for_everyone()
    results = gather_object(results)
    sequences = gather_object(sequences)
    return results, sequences, plans


def evaluate_sequence(
    env, model, task_checker, initial_state, eval_sequence, lang_embeddings,
    val_annotations, cfg, progress, record, rollout_video, i,
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if record:
        caption = " | ".join(eval_sequence)
        rollout_video.new_video(tag=get_video_tag(i), caption=caption)
    success_counter = 0
    if cfg.debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for sub_idx, subtask in enumerate(eval_sequence):
        if record:
            rollout_video.new_subtask()
        success = rollout(
            env, model, task_checker, cfg, sub_idx, subtask, lang_embeddings,
            val_annotations, progress, i, record, rollout_video,
        )
        if record:
            rollout_video.draw_outcome(success)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(
    env, model, task_oracle, cfg, sub_idx, subtask, lang_embeddings,
    val_annotations, progress, seq_idx, record=False, rollout_video=None,
):
    if cfg.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    # get language goal embedding
    goal = lang_embeddings.get_lang_goal(lang_annotation)
    goal['lang_text'] = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    # Per-subtask progress bar: shows step-by-step progress of this rollout.
    bar = progress.add_task(
        f"  seq {seq_idx} | subtask {sub_idx}. {subtask}", total=cfg.ep_len
    )
    try:
        for step in range(cfg.ep_len):
            action = model.step(obs, goal)
            obs, _, _, current_info = env.step(action)
            if cfg.debug:
                img = env.render(mode="rgb_array")
                join_vis_lang(img, lang_annotation)
            if record:
                # update video
                rollout_video.update(obs["rgb_obs"]["rgb_static"])
            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                if cfg.debug:
                    print(colored("success", "green"), end=" ")
                if record:
                    rollout_video.add_language_instruction(lang_annotation)
                progress.update(bar, completed=cfg.ep_len)
                return True
            progress.update(bar, advance=1)
    finally:
        progress.remove_task(bar)

    if cfg.debug:
        print(colored("fail", "red"), end=" ")
    if record:
        rollout_video.add_language_instruction(lang_annotation)
    return False


def main(cfg):
    from datetime import timedelta
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=864000))
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    # One GPU per process; build an independent model + env on each.
    device_id = accelerator.local_process_index
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    seed_everything(0, workers=True)  # type:ignore

    logging.basicConfig(
        level=logging.INFO if accelerator.is_main_process else logging.WARNING,
        format=f"[rank{accelerator.process_index}] %(asctime)s %(message)s",
    )

    log_wandb = cfg.log_wandb

    checkpoints = [get_last_checkpoint(Path(cfg.train_folder))]
    lang_embeddings = None
    env = None
    results = {}
    sequences = {}

    if accelerator.is_main_process:
        print('train_folder', cfg.train_folder)

    # Single shared log dir across all processes.
    log_dir = None
    if accelerator.is_main_process:
        log_dir = get_log_dir(cfg.train_folder)
        if log_wandb:
            os.makedirs(log_dir / "wandb", exist_ok=True)
    log_dir = broadcast_log_dir(accelerator, log_dir)

    for checkpoint in checkpoints:
        logger.info(f"[{device}] building env / lang embeddings")
        env, _, lang_embeddings = get_default_beso_and_env(
            cfg.train_folder,
            cfg.root_data_dir,
            checkpoint,
            env=env,
            lang_embeddings=lang_embeddings,
            eval_cfg_overwrite=cfg.eval_cfg_overwrite,
            device_id=device_id,
            cfg=cfg,
        )
        if cfg.train_folder[-3:] == ".pt":
            ckpt = cfg.train_folder
            ckpt_path = ckpt
        else:
            ckpt_path = os.path.join(cfg.train_folder)
            ckpt = None
            for file in os.listdir(ckpt_path):
                print(file)
                if file[-3:] == ".pt":
                    ckpt = os.path.join(ckpt_path, file)
        print("??", ckpt_path, ckpt)
        logger.info(f"[{device}] Loading model from {ckpt}")
        # exit(0)
        state_dict = torch.load(ckpt, map_location='cpu', weights_only=False)

        model = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(state_dict['model'], strict=False)
        model.freeze()
        model = model.cuda(device)

        logger.info(
            f"sampling steps={cfg.num_sampling_steps} sampler={cfg.sampler_type} "
            f"multistep={cfg.multistep} sigma_min={cfg.sigma_min} "
            f"sigma_max={cfg.sigma_max} noise_scheduler={cfg.noise_scheduler}"
        )
        model.num_sampling_steps = cfg.num_sampling_steps
        model.sampler_type = cfg.sampler_type
        model.multistep = cfg.multistep
        if cfg.sigma_min is not None:
            model.sigma_min = cfg.sigma_min
        if cfg.sigma_max is not None:
            model.sigma_max = cfg.sigma_max
        if cfg.noise_scheduler is not None:
            model.noise_scheduler = cfg.noise_scheduler

        if cfg.cfg_value != 1:
            raise NotImplementedError("cfg_value != 1 not implemented yet")
        model.process_device()
        model.eval()

        ckpt_results, ckpt_sequences, _ = evaluate_policy(
            model, env, lang_embeddings, cfg, accelerator,
            num_videos=cfg.num_videos, save_dir=Path(log_dir),
        )
        results[checkpoint] = ckpt_results
        sequences[checkpoint] = ckpt_sequences

    if accelerator.is_main_process:
        for checkpoint in checkpoints:
            print(f"Results for {checkpoint}:")
            print_and_save(results[checkpoint], sequences[checkpoint], cfg, log_dir=log_dir)


def broadcast_log_dir(accelerator, log_dir):
    """Share the main process's log dir string with all other processes."""
    payload = [str(log_dir) if log_dir is not None else None]
    payload = broadcast_object_list(payload, accelerator)
    return payload[0]


def broadcast_object_list(obj_list, accelerator):
    from accelerate.utils import broadcast_object_list as _bcast
    _bcast(obj_list, from_process=0)
    return obj_list


if __name__ == "__main__":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # Set CUDA device IDs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_model_path", type=str, default="")
    parser.add_argument("--action_model_folder", type=str, default="")
    parser.add_argument("--clip_model_path", type=str, default="")
    parser.add_argument("--calvin_abc_dir", type=str, default="")

    args = parser.parse_args()

    with initialize(config_path="../policy_conf", job_name="calvin_evaluate_all.yaml"):
        cfg = compose(config_name="calvin_evaluate_all.yaml")
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.train_folder = args.action_model_folder
    cfg.model.text_encoder_path = args.clip_model_path
    cfg.root_data_dir = args.calvin_abc_dir
    main(cfg)

    # accelerate launch --num_processes 4 policy_evaluation/calvin_evaluate_multi.py \
    #   --video_model_path /home/disk2/gyj/hyc_ckpt/svd_2camera/checkpoint-100000 \
    #   --action_model_folder /home/disk2/gyj/hyccode/Video-Prediction-Policy/checkpoint/alllayer1 \
    #   --clip_model_path /home/disk2/gyj/hyc_ckpt/llm/clip-vit-base-patch32 \
    #   --calvin_abc_dir /home/disk2/gyj/task_ABC_D
