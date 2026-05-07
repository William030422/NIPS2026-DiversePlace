#!/usr/bin/env python3
"""Run inference for a saved EfficientPlace PPO checkpoint."""

import argparse
import json
import logging
import pickle
import random
import re
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import torch
from omegaconf import OmegaConf

from src import Agent, Environment

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt


Placement = Dict[str, Tuple[int, int, int, int]]


@dataclass
class ConstraintConfig:
    enabled: bool
    protected_macro_count: int
    radius_ratio: float
    min_radius: int


@dataclass
class RolloutResult:
    index: int
    seed: int
    hpwl: float
    hpwl_breakdown: Dict[str, float]
    dead_end: bool
    placement: Placement


class NullSummaryWriter:
    def close(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer full macro placements from a saved EfficientPlace checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path. Accepts either a model directory or an actor checkpoint file.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Benchmark config name or absolute YAML path. Auto-resolved from the run directory when omitted.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Checkpoint episode used to resolve the curriculum window. Defaults to the saved best episode.",
    )
    parser.add_argument(
        "--ppo-macros",
        type=int,
        default=None,
        help="Override the number of macros placed by PPO before greedy completion.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=20,
        help="Number of stochastic inference rollouts to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed. Rollout i uses seed+ i.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for inference outputs. Defaults to <run_root>/inference/ep<episode>_m<ppo_macros>.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_omegaconf_resolvers() -> None:
    if not OmegaConf.has_resolver("now"):
        OmegaConf.register_new_resolver(
            "now",
            lambda pattern: datetime.now().strftime(pattern),
        )


def resolve_checkpoint_paths(checkpoint_arg: str) -> Tuple[Path, Path, Path, Path]:
    checkpoint_path = Path(checkpoint_arg).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    if checkpoint_path.is_dir():
        model_dir = checkpoint_path
        actor_path = model_dir / "actor_net.pth"
        critic_path = model_dir / "critic_net.pth"
    else:
        actor_path = checkpoint_path
        model_dir = actor_path.parent
        critic_path = model_dir / "critic_net.pth"

    if not actor_path.exists():
        raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")
    if not critic_path.exists():
        raise FileNotFoundError(f"Critic checkpoint not found: {critic_path}")

    run_root = model_dir.parent
    return checkpoint_path, model_dir, actor_path, run_root


def resolve_config_path(repo_root: Path, run_root: Path, config_arg: Optional[str]) -> Path:
    if config_arg:
        explicit = Path(config_arg).expanduser()
        if explicit.is_file():
            return explicit.resolve()
        if explicit.suffix == ".yaml":
            candidate = repo_root / "conf" / "benchmark" / explicit.name
        else:
            candidate = repo_root / "conf" / "benchmark" / f"{config_arg}.yaml"
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"Benchmark config not found for --config={config_arg}")

    workspace_root = repo_root / "workspace"
    config_name = None
    try:
        relative_run_root = run_root.relative_to(workspace_root)
        if relative_run_root.parts:
            config_name = relative_run_root.parts[0]
    except ValueError:
        match = re.search(r"/workspace/([^/]+)/", str(run_root))
        if match:
            config_name = match.group(1)

    if not config_name:
        raise ValueError(
            "Could not infer benchmark config from checkpoint path. Pass --config explicitly."
        )

    config_path = repo_root / "conf" / "benchmark" / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Auto-resolved config does not exist: {config_path}. Pass --config explicitly."
        )
    return config_path.resolve()


def load_run_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_torch_state(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_checkpoint_episode(run_root: Path, episode_override: Optional[int]) -> int:
    if episode_override is not None:
        return int(episode_override)

    run_summary = load_run_json(run_root / "run_summary.json")
    if run_summary and run_summary.get("best_episode") is not None:
        return int(run_summary["best_episode"])

    topk_summary = load_run_json(run_root / "sol" / "global_topk" / "topk_summary.json")
    if topk_summary:
        entries = topk_summary.get("entries", [])
        if entries:
            return int(entries[0]["episode"])

    episode_from_name = re.search(r"_ep_(\d+)", str(run_root))
    if episode_from_name:
        return int(episode_from_name.group(1))

    raise ValueError(
        "Could not determine which episode produced this checkpoint. Pass --episode explicitly."
    )


def infer_curriculum_window(
    episode: int,
    initial_macros: int,
    window_size: int,
    increment: int,
) -> Tuple[int, int]:
    window_index = max(0, episode // window_size)
    num_macros = initial_macros + window_index * increment
    return window_index, num_macros


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def maybe_load_anchor_placement(run_root: Path, window_index: int) -> Optional[Placement]:
    if window_index <= 0:
        return None
    anchor_path = run_root / "curriculum" / f"window_{window_index - 1}" / "best_placement.pkl"
    if not anchor_path.exists():
        logging.warning("Previous-window anchor placement missing: %s", anchor_path)
        return None
    return load_pickle(anchor_path)


def build_constrained_position_mask(
    position_mask: torch.Tensor,
    anchor_x: int,
    anchor_y: int,
    radius_x: int,
    radius_y: int,
    grid_size: int,
    macro_name: str,
    macro_idx: int,
) -> torch.Tensor:
    min_x = max(0, anchor_x - radius_x)
    max_x = min(grid_size - 1, anchor_x + radius_x)
    min_y = max(0, anchor_y - radius_y)
    max_y = min(grid_size - 1, anchor_y + radius_y)

    while True:
        box_valid = torch.zeros_like(position_mask, dtype=torch.bool)
        box_valid[min_x:max_x + 1, min_y:max_y + 1] = True
        allowed_valid = torch.logical_and(position_mask < 0.5, box_valid)

        if torch.any(allowed_valid):
            constrained_mask = torch.ones_like(position_mask)
            constrained_mask[allowed_valid] = 0.0
            return constrained_mask

        if min_x == 0 and max_x == grid_size - 1 and min_y == 0 and max_y == grid_size - 1:
            logging.warning(
                "Hard constraint fallback to original mask for macro %s (index %d).",
                macro_name,
                macro_idx,
            )
            return position_mask.clone()

        min_x = max(0, min_x - 1)
        max_x = min(grid_size - 1, max_x + 1)
        min_y = max(0, min_y - 1)
        max_y = min(grid_size - 1, max_y + 1)


def apply_hard_constraint(
    env: Environment,
    state: torch.Tensor,
    constraint_cfg: ConstraintConfig,
    anchor_placement: Optional[Placement],
) -> torch.Tensor:
    if not constraint_cfg.enabled or anchor_placement is None:
        return state
    if env.t >= env.num_macros:
        return state

    macro_idx = env.t
    if macro_idx >= min(constraint_cfg.protected_macro_count, env.num_macros):
        return state

    macro_name = env.macro_name_list[macro_idx]
    anchor = anchor_placement.get(macro_name)
    if anchor is None:
        return state

    radius_x = max(
        constraint_cfg.min_radius,
        int(ceil(constraint_cfg.radius_ratio * env.size_x[macro_idx])),
    )
    radius_y = max(
        constraint_cfg.min_radius,
        int(ceil(constraint_cfg.radius_ratio * env.size_y[macro_idx])),
    )

    constrained_state = state.clone()
    constrained_state[2] = build_constrained_position_mask(
        position_mask=state[2],
        anchor_x=int(anchor[0]),
        anchor_y=int(anchor[1]),
        radius_x=radius_x,
        radius_y=radius_y,
        grid_size=env.grid,
        macro_name=macro_name,
        macro_idx=macro_idx,
    )
    return constrained_state


def state_has_no_valid_actions(state: torch.Tensor) -> bool:
    return bool(torch.all(state[2] > 0.5).item())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serialize_placement(placement: Placement) -> Dict[str, List[int]]:
    return {
        macro_name: [int(x), int(y), int(w), int(h)]
        for macro_name, (x, y, w, h) in placement.items()
    }


def run_rollout(
    env: Environment,
    agent: Agent,
    rollout_index: int,
    rollout_seed: int,
    constraint_cfg: ConstraintConfig,
    anchor_placement: Optional[Placement],
    dead_end_hpwl_penalty: float,
) -> RolloutResult:
    seed_everything(rollout_seed)
    raw_state = env.reset()
    state = apply_hard_constraint(env, raw_state, constraint_cfg, anchor_placement)
    ppo_t = 0
    dead_end = False

    while True:
        if state_has_no_valid_actions(state):
            dead_end = True
            logging.warning(
                "Rollout %d dead-ended before PPO step %d (macro index %d).",
                rollout_index,
                ppo_t,
                env.t,
            )
            break

        action, _ = agent.select_action(state, ppo_t)
        raw_next_state, _, done, _ = env.step(action)
        next_state = apply_hard_constraint(env, raw_next_state, constraint_cfg, anchor_placement)

        if done and env.num_macros > env.num_macros_to_place:
            greedy_state = raw_next_state
            for _ in range(env.num_macros - env.num_macros_to_place):
                greedy_action = agent.act_greedy(greedy_state)
                greedy_state, _, _, _ = env.step(greedy_action)
            break

        if done:
            break

        state = next_state
        ppo_t += 1

    hpwl_breakdown = env.calculate_hpwl_by_groups()
    if dead_end:
        hpwl_breakdown["ppo_window"] += dead_end_hpwl_penalty
        hpwl_breakdown["total"] += dead_end_hpwl_penalty

    return RolloutResult(
        index=rollout_index,
        seed=rollout_seed,
        hpwl=float(hpwl_breakdown["total"]),
        hpwl_breakdown={k: float(v) for k, v in hpwl_breakdown.items()},
        dead_end=dead_end,
        placement=env.macro_pos.copy(),
    )


def visualize_placements(
    placements: List[Placement],
    hpwls: List[float],
    grid_size: int,
    save_path: Path,
) -> None:
    num_plots = len(placements)
    cols = min(5, max(1, num_plots))
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.0))
    axes = np.atleast_1d(axes).reshape(-1)

    best_hpwl = min(hpwls) if hpwls else float("nan")
    for idx, (macro_pos, hpwl) in enumerate(zip(placements, hpwls)):
        ax = axes[idx]
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        if abs(hpwl - best_hpwl) <= 1e-12:
            ax.set_title(f"Run {idx + 1}\nHPWL: {hpwl:.4f}", color="darkred")
        else:
            ax.set_title(f"Run {idx + 1}\nHPWL: {hpwl:.4f}")

        for _, (x, y, w, h) in macro_pos.items():
            rect = patches.Rectangle(
                (y, x),
                h,
                w,
                linewidth=0.5,
                edgecolor="navy",
                facecolor="deepskyblue",
                alpha=0.45,
            )
            ax.add_patch(rect)

    for idx in range(num_plots, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"EfficientPlace Inference Rollouts ({num_plots} runs)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_omegaconf_resolvers()

    repo_root = Path(__file__).resolve().parent
    _, model_dir, actor_path, run_root = resolve_checkpoint_paths(args.checkpoint)
    config_path = resolve_config_path(repo_root, run_root, args.config)
    config = OmegaConf.load(config_path)

    run_summary = load_run_json(run_root / "run_summary.json")
    checkpoint_episode = resolve_checkpoint_episode(run_root, args.episode)

    env_cfg = OmegaConf.to_container(config.env, resolve=True)
    agent_cfg = OmegaConf.to_container(config.agent, resolve=True)
    trainer_cfg = OmegaConf.to_container(config.trainer, resolve=True)

    env_cfg.pop("_target_", None)
    agent_cfg.pop("_target_", None)
    trainer_cfg.pop("_target_", None)

    actor_state = load_torch_state(actor_path)
    critic_state = load_torch_state(model_dir / "critic_net.pth")
    actor_horizon = int(actor_state["time_embedding.weight"].shape[0])
    agent_cfg["num_macros_to_place"] = actor_horizon

    enable_curriculum = bool(trainer_cfg.get("enable_curriculum", False))
    initial_num_macros = int(config.get("num_macros_to_place", env_cfg["num_macros_to_place"]))
    if enable_curriculum:
        window_size = int(trainer_cfg.get("curriculum_window_size", 2000))
        increment = int(trainer_cfg.get("curriculum_increment", 5))
        window_index, resolved_ppo_macros = infer_curriculum_window(
            checkpoint_episode,
            initial_macros=initial_num_macros,
            window_size=window_size,
            increment=increment,
        )
    else:
        window_size = None
        increment = None
        window_index = 0
        resolved_ppo_macros = int(env_cfg["num_macros_to_place"])

    if args.ppo_macros is not None:
        resolved_ppo_macros = int(args.ppo_macros)

    if resolved_ppo_macros > actor_horizon:
        raise ValueError(
            f"Resolved PPO macro count {resolved_ppo_macros} exceeds actor horizon {actor_horizon}."
        )

    env_cfg["num_macros_to_place"] = resolved_ppo_macros
    env = Environment(**env_cfg)
    if resolved_ppo_macros > env.num_macros:
        resolved_ppo_macros = env.num_macros
        env.update_num_macros_to_place(resolved_ppo_macros)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = Agent(**agent_cfg)
    dummy_writer = NullSummaryWriter()
    agent.set_up(device=device, tb_writer=dummy_writer)
    agent.actor_net.load_state_dict(actor_state)
    agent.critic_net.load_state_dict(critic_state)
    agent.actor_net.eval()
    agent.critic_net.eval()

    constraint_cfg = ConstraintConfig(
        enabled=bool(enable_curriculum and trainer_cfg.get("enable_hard_constraint", True)),
        protected_macro_count=int(trainer_cfg.get("protected_macro_count", 5)),
        radius_ratio=float(trainer_cfg.get("constraint_radius_ratio", 0.25)),
        min_radius=int(trainer_cfg.get("constraint_min_radius", 1)),
    )
    anchor_placement = maybe_load_anchor_placement(run_root, window_index) if constraint_cfg.enabled else None
    dead_end_hpwl_penalty = float(trainer_cfg.get("dead_end_hpwl_penalty", 1000.0))

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (run_root / "inference" / f"ep{checkpoint_episode}_m{resolved_ppo_macros}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Repo root: %s", repo_root)
    logging.info("Run root: %s", run_root)
    logging.info("Config: %s", config_path)
    logging.info("Actor checkpoint: %s", actor_path)
    logging.info("Device: %s", device)
    logging.info("Checkpoint episode: %d", checkpoint_episode)
    if enable_curriculum:
        logging.info(
            "Curriculum window: %d (window_size=%d, increment=%d, PPO macros=%d)",
            window_index,
            window_size,
            increment,
            resolved_ppo_macros,
        )
    else:
        logging.info("Non-curriculum run with PPO macros=%d", resolved_ppo_macros)
    logging.info("Total macros in benchmark: %d", env.num_macros)
    logging.info(
        "Hard constraint: %s",
        "enabled" if constraint_cfg.enabled and anchor_placement is not None else "disabled",
    )

    rollout_results: List[RolloutResult] = []
    for rollout_idx in range(args.num_runs):
        rollout_seed = args.seed + rollout_idx
        rollout_result = run_rollout(
            env=env,
            agent=agent,
            rollout_index=rollout_idx,
            rollout_seed=rollout_seed,
            constraint_cfg=constraint_cfg,
            anchor_placement=anchor_placement,
            dead_end_hpwl_penalty=dead_end_hpwl_penalty,
        )
        rollout_results.append(rollout_result)
        logging.info(
            "Run %02d | seed=%d | HPWL=%.4f | dead_end=%s",
            rollout_idx + 1,
            rollout_seed,
            rollout_result.hpwl,
            rollout_result.dead_end,
        )

    hpwls = np.asarray([result.hpwl for result in rollout_results], dtype=float)
    best_idx = int(np.argmin(hpwls)) if len(hpwls) else -1
    image_path = output_dir / f"rollouts_{args.num_runs}runs_ep{checkpoint_episode}_m{resolved_ppo_macros}.png"
    visualize_placements(
        placements=[result.placement for result in rollout_results],
        hpwls=[result.hpwl for result in rollout_results],
        grid_size=env.grid,
        save_path=image_path,
    )

    summary = {
        "checkpoint": str(actor_path),
        "run_root": str(run_root),
        "config": str(config_path),
        "device": device,
        "checkpoint_episode": checkpoint_episode,
        "resolved_window_index": window_index,
        "resolved_ppo_macros": resolved_ppo_macros,
        "actor_horizon": actor_horizon,
        "total_macros": int(env.num_macros),
        "run_summary_best_episode": None if not run_summary else run_summary.get("best_episode"),
        "run_summary_completed_episodes": None if not run_summary else run_summary.get("completed_episodes"),
        "hard_constraint_enabled": bool(constraint_cfg.enabled and anchor_placement is not None),
        "anchor_window_index": window_index - 1 if anchor_placement is not None else None,
        "image_path": str(image_path),
        "num_runs": args.num_runs,
        "base_seed": args.seed,
        "hpwl_stats": {
            "best": float(np.min(hpwls)) if len(hpwls) else None,
            "mean": float(np.mean(hpwls)) if len(hpwls) else None,
            "std": float(np.std(hpwls)) if len(hpwls) else None,
            "worst": float(np.max(hpwls)) if len(hpwls) else None,
            "best_run_index": best_idx,
        },
        "runs": [
            {
                "index": result.index,
                "seed": result.seed,
                "hpwl": result.hpwl,
                "hpwl_breakdown": result.hpwl_breakdown,
                "dead_end": result.dead_end,
                "placement": serialize_placement(result.placement),
            }
            for result in rollout_results
        ],
    }

    summary_path = output_dir / "summary.json"
    placements_path = output_dir / "placements.pkl"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with placements_path.open("wb") as f:
        pickle.dump([result.placement for result in rollout_results], f)

    dummy_writer.close()
    logging.info("Saved rollout image to: %s", image_path)
    logging.info("Saved summary to: %s", summary_path)
    logging.info("Saved placements to: %s", placements_path)
    if best_idx >= 0:
        logging.info(
            "Best rollout: run %d with HPWL %.4f",
            best_idx + 1,
            rollout_results[best_idx].hpwl,
        )


if __name__ == "__main__":
    main()
