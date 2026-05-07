from math import ceil, log
import csv
import torch
import pickle
import numpy as np
import os
import json
import hydra
from omegaconf import DictConfig
import logging
import treelib
from tensorboardX import SummaryWriter
import random
from src import Agent, Environment, ReplayBuffer
from src.curriculum_manager import CurriculumManager
from src.diversity_manager import WindowEliteMap

import wandb
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.linalg
import copy

# def calculate_macro_iou(macro_pos_1, macro_pos_2):
#     """
#     1. Macro-wise IOU: Mean IoU across matched macros.
#     """
#     total_iou = 0.0
#     num_macros = len(macro_pos_1)
#
#     for node_name in macro_pos_1:
#         if node_name not in macro_pos_2:
#             continue
#
#         x1, y1, w1, h1 = macro_pos_1[node_name]
#         x2, y2, w2, h2 = macro_pos_2[node_name]
#
#         inter_x1 = max(x1, x2)
#         inter_y1 = max(y1, y2)
#         inter_x2 = min(x1 + w1, x2 + w2)
#         inter_y2 = min(y1 + h1, y2 + h2)
#
#         inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
#         area1 = w1 * h1
#         area2 = w2 * h2
#         union_area = area1 + area2 - inter_area
#
#         if union_area > 0:
#             total_iou += inter_area / union_area
#
#     return total_iou / num_macros if num_macros > 0 else 0.0
#
# def calculate_canvas_iou(macro_pos_1, macro_pos_2, grid_size):
#     """
#     2. Canvas-wise IOU: IoU of the binary occupancy maps.
#     Treats all macros as 'occupied space' without distinguishing identity.
#     """
#     grid1 = np.zeros((grid_size, grid_size), dtype=np.bool_)
#     grid2 = np.zeros((grid_size, grid_size), dtype=np.bool_)
#
#     # Fill Grid 1
#     for _, (x, y, w, h) in macro_pos_1.items():
#         # Clip to grid boundaries to be safe
#         x_end = min(x + w, grid_size)
#         y_end = min(y + h, grid_size)
#         grid1[x:x_end, y:y_end] = True
#
#     # Fill Grid 2
#     for _, (x, y, w, h) in macro_pos_2.items():
#         x_end = min(x + w, grid_size)
#         y_end = min(y + h, grid_size)
#         grid2[x:x_end, y:y_end] = True
#
#     intersection = np.logical_and(grid1, grid2).sum()
#     union = np.logical_or(grid1, grid2).sum()
#
#     return intersection / union if union > 0 else 0.0
#
# def calculate_l1_distance(macro_pos_1, macro_pos_2):
#     """
#     3. L1 Distance: Sum of L1 distances of all macro top-left corners.
#     """
#     dist_sum = 0.0
#
#     for node_name in macro_pos_1:
#         if node_name in macro_pos_2:
#             x1, y1, _, _ = macro_pos_1[node_name]
#             x2, y2, _, _ = macro_pos_2[node_name]
#             dist_sum += abs(x1 - x2) + abs(y1 - y2)
#
#     return dist_sum
#
# def compute_vendi_score_from_matrix(K):
#     """
#     Computes Vendi Score given a Similarity Matrix K.
#     """
#     n = K.shape[0]
#     if n == 0: return 0.0
#
#     # Normalize Kernel: K/n
#     K_norm = K / n
#
#     # Compute Eigenvalues
#     eigvals = np.linalg.eigvalsh(K_norm)
#
#     # Filter small numerical errors and normalize
#     eigvals = eigvals[eigvals > 1e-10]
#     p = eigvals / np.sum(eigvals)
#
#     entropy = -np.sum(p * np.log(p))
#     return np.exp(entropy)

def visualize_placements(placements, hpwls, grid_size, save_path, step):
    """
    Visualizes 10 placements.
    """
    num_plots = len(placements)
    cols = 5
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
    axes = axes.flatten()

    for i, macro_pos in enumerate(placements):
        ax = axes[i]
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        hpwl_val = hpwls[i] if hpwls else 0.0
        ax.set_title(f"Rollout {i+1}\nHPWL: {hpwl_val:.4f}")
        ax.invert_yaxis()

        for node_name, (x, y, w, h) in macro_pos.items():
            # matplotlib rectangle: (y, x) because EDA usually maps array[x][y]
            rect = patches.Rectangle((y, x), h, w, linewidth=1, edgecolor='blue', facecolor='cyan', alpha=0.5)
            ax.add_patch(rect)

    plt.suptitle(f"Placements at Step {step}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _default_wandb_group(config: DictConfig) -> str:
    if config.trainer.get('enable_curriculum', False):
        if config.trainer.get('enable_diversity_phase', False):
            return "VanillaPPO-Curriculum-Unfixed-Hard-Constraint-Diversity"
        return "VanillaPPO-Curriculum-Unfixed-Hard-Constraint"
    return "VanillaPPO-Baseline"


class Trainer:

    def __init__(self, num_loops, num_episodes_in_loop, num_update_epochs, update_batch_size,
        num_macros_to_place, solution_pool_size, alpha, update_frontiers_begin, update_frontiers_freq,
        model_dir, solution_dir, enable_curriculum=False, curriculum_window_size=2000,
        curriculum_increment=5, curriculum_selection_strategy="best-1", curriculum_top_k=3,
        curriculum_fixed_macro_count=0,
        curriculum_prefix_fixed_phase_ratio=1.0,
        curriculum_prefix_release_constraint_ratio=None,
        curriculum_prefix_transfer_mode="auto",
        enable_hard_constraint=True, protected_macro_count=5,
        constraint_radius_ratio=0.25, constraint_min_radius=1,
        enable_adaptive_hard_constraint=False,
        adaptive_large_macro_count=0,
        visualize_every_episodes=500,
        enable_prefix_diversity_window0=False,
        enable_diversity_phase=False, diversity_phase_ratio=0.5,
        diversity_quality_eps=0.05, diversity_reference_k=10,
        diversity_output_k=5, diversity_bonus_lambda=0.01,
        diversity_bonus_cap_ratio=0.10, diversity_min_refs=3,
        qd_map_bins=8,
        global_topk_size=None,
        dead_end_reward_penalty=-10.0,
        dead_end_hpwl_penalty=1000.0,
        target_budget_episodes=None,
        placement_metrics_path=None,
        run_summary_path=None,
        autoexp_run_id=None,
        checkpoint_path=None,
        resume_from=None,
        checkpoint_every_episodes=None,
    ):

        self.num_loops = num_loops
        self.num_episodes_in_loop = num_episodes_in_loop

        self.num_update_epochs = num_update_epochs
        self.update_batch_size = update_batch_size
        self.num_macros_to_place = num_macros_to_place

        self.buffer_capacity = num_episodes_in_loop * num_macros_to_place
        self.solution_pool_size = solution_pool_size
        self.alpha = alpha

        self.update_frontiers_begin = update_frontiers_begin
        self.update_frontiers_freq = update_frontiers_freq

        self.best_hpwl = float('inf')
        self.best_placement = None
        self.best_hpwl_episode = None

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.solution_dir = solution_dir
        os.makedirs(self.solution_dir, exist_ok=True)
        self.run_root = Path(self.model_dir).resolve().parent
        self.placement_metrics_path = str(Path(placement_metrics_path).resolve()) if placement_metrics_path else str((self.run_root / "placement_metrics.csv").resolve())
        self.run_summary_path = str(Path(run_summary_path).resolve()) if run_summary_path else str((self.run_root / "run_summary.json").resolve())
        self.autoexp_run_id = autoexp_run_id
        self.checkpoint_path = checkpoint_path
        self.resume_from = resume_from
        self.checkpoint_every_episodes = checkpoint_every_episodes
        self.pending_wandb_metrics = {}
        self.target_budget_episodes = (
            None if target_budget_episodes is None else int(target_budget_episodes)
        )
        if (
            self.target_budget_episodes is not None
            and self.target_budget_episodes % self.num_episodes_in_loop != 0
        ):
            raise ValueError(
                "target_budget_episodes must be divisible by num_episodes_in_loop "
                f"(got budget={self.target_budget_episodes}, num_episodes_in_loop={self.num_episodes_in_loop})"
            )
        self.enable_hard_constraint = enable_hard_constraint
        self.protected_macro_count = protected_macro_count
        self.constraint_radius_ratio = constraint_radius_ratio
        self.constraint_min_radius = constraint_min_radius
        self.enable_adaptive_hard_constraint = bool(enable_adaptive_hard_constraint)
        self.adaptive_large_macro_count = max(0, int(adaptive_large_macro_count))
        self.visualize_every_episodes = max(1, int(visualize_every_episodes))
        self.enable_prefix_diversity_window0 = bool(enable_prefix_diversity_window0)
        self.enable_diversity_phase = enable_diversity_phase
        self.diversity_phase_ratio = diversity_phase_ratio
        self.diversity_quality_eps = diversity_quality_eps
        self.diversity_reference_k = diversity_reference_k
        self.diversity_output_k = diversity_output_k
        self.diversity_bonus_lambda = diversity_bonus_lambda
        self.diversity_bonus_cap_ratio = diversity_bonus_cap_ratio
        self.diversity_min_refs = diversity_min_refs
        self.qd_map_bins = max(2, int(qd_map_bins))
        self.last_window_best_placement = None
        self.adaptive_large_macro_names = None
        self.last_window_num_macros = None
        self.warmup_episode_pool = []
        self.frozen_reference_pool = []
        self.diversity_episode_pool = []
        self.warmup_best_hpwl = float('inf')
        self.diversity_window_index = 0
        self.frozen_refs_ready = False
        self.window_qd_map = WindowEliteMap(self.qd_map_bins, self.qd_map_bins)
        self.global_topk_size = (
            solution_pool_size if global_topk_size is None else global_topk_size
        )
        self.global_topk_size = max(1, int(self.global_topk_size))
        self.global_topk_archive = []
        self.global_topk_dir = os.path.join(self.solution_dir, "global_topk")
        os.makedirs(self.global_topk_dir, exist_ok=True)
        logging.info(f"Global top-K archive ENABLED (K={self.global_topk_size})")
        self.dead_end_reward_penalty = float(dead_end_reward_penalty)
        self.dead_end_hpwl_penalty = float(dead_end_hpwl_penalty)

        # Curriculum learning
        self.enable_curriculum = enable_curriculum
        self.curriculum_selection_strategy = curriculum_selection_strategy
        self.curriculum_top_k = max(1, int(curriculum_top_k))
        self.curriculum_fixed_macro_count = max(0, int(curriculum_fixed_macro_count))
        self.curriculum_prefix_fixed_phase_ratio = float(
            np.clip(curriculum_prefix_fixed_phase_ratio, 0.0, 1.0)
        )
        self.curriculum_prefix_release_constraint_ratio = (
            None
            if curriculum_prefix_release_constraint_ratio is None
            else float(curriculum_prefix_release_constraint_ratio)
        )
        self.curriculum_prefix_transfer_mode = str(curriculum_prefix_transfer_mode).lower()
        self.curriculum_prefix_pool = []
        self.last_prefix_choice_idx = -1
        self.active_episode_prefix_mode = "none"
        self.active_episode_prefix_anchors = {}
        self.active_episode_prefix_boxes = {}
        self.active_episode_prefix_count = 0
        self.active_episode_prefix_radius_ratio = None
        if self.enable_curriculum:
            curriculum_dir = os.path.join(model_dir, "../curriculum")
            self.curriculum_manager = CurriculumManager(
                window_size=curriculum_window_size,
                increment_size=curriculum_increment,
                initial_num_macros=num_macros_to_place,
                selection_strategy=curriculum_selection_strategy,
                top_k=curriculum_top_k,
                save_dir=curriculum_dir
            )
            logging.info("Curriculum learning ENABLED")
        else:
            self.curriculum_manager = None
            logging.info("Curriculum learning DISABLED")

    def setup_tb_writer(self, tb_writer: SummaryWriter):
        self.tb_writer = tb_writer

    def _queue_wandb_metrics(self, metrics):
        if metrics:
            self.pending_wandb_metrics.update(metrics)

    def _consume_wandb_metrics(self):
        metrics = self.pending_wandb_metrics.copy()
        self.pending_wandb_metrics.clear()
        return metrics

    def _clear_episode_prefix_plan(self):
        self.active_episode_prefix_mode = "none"
        self.active_episode_prefix_anchors = {}
        self.active_episode_prefix_boxes = {}
        self.active_episode_prefix_count = 0
        self.active_episode_prefix_radius_ratio = None

    def _get_prefix_fix_phase_length(self):
        if not self.enable_curriculum or self.curriculum_manager is None:
            return 0
        return int(
            np.floor(
                self.curriculum_manager.window_size * self.curriculum_prefix_fixed_phase_ratio
            )
        )

    def _get_prefix_mode_code(self):
        if self.active_episode_prefix_mode == "fixed":
            return 1.0
        if self.active_episode_prefix_mode == "released":
            return 2.0
        if self.active_episode_prefix_mode == "consensus_soft":
            return 3.0
        return 0.0

    def _get_effective_prefix_transfer_mode(self):
        mode = self.curriculum_prefix_transfer_mode
        if mode in {"", "auto", "legacy"}:
            if self.curriculum_prefix_fixed_phase_ratio >= 1.0:
                return "hard_fixing"
            return "bridge"
        return mode

    def _build_consensus_soft_prefix_boxes(self, env: Environment, fix_count: int):
        boxes = {}
        for macro_idx, macro_name in enumerate(env.macro_name_list[:fix_count]):
            elite_positions = []
            for placement in self.curriculum_prefix_pool:
                anchor = placement.get(macro_name)
                if anchor is not None:
                    elite_positions.append(anchor)

            if len(elite_positions) == 0:
                continue

            x_coords = np.asarray([anchor[0] for anchor in elite_positions], dtype=np.float64)
            y_coords = np.asarray([anchor[1] for anchor in elite_positions], dtype=np.float64)
            anchor_x = int(np.round(np.median(x_coords)))
            anchor_y = int(np.round(np.median(y_coords)))

            # Heuristic, parameter-free consensus softness:
            # keep the usual size-aware local box, then widen it enough to cover
            # the disagreement among the top-k elite placements.
            base_radius_x = max(
                self.constraint_min_radius,
                int(ceil(self.constraint_radius_ratio * env.size_x[macro_idx])),
            )
            base_radius_y = max(
                self.constraint_min_radius,
                int(ceil(self.constraint_radius_ratio * env.size_y[macro_idx])),
            )
            spread_radius_x = int(ceil(np.max(np.abs(x_coords - anchor_x))))
            spread_radius_y = int(ceil(np.max(np.abs(y_coords - anchor_y))))

            boxes[macro_name] = {
                "anchor_x": anchor_x,
                "anchor_y": anchor_y,
                "radius_x": max(base_radius_x, spread_radius_x),
                "radius_y": max(base_radius_y, spread_radius_y),
            }

        return boxes

    def _refresh_curriculum_prefix_pool(self):
        self.curriculum_prefix_pool = []
        if not self.enable_curriculum or self.curriculum_fixed_macro_count <= 0:
            return

        ranked_window = sorted(
            self.curriculum_manager.window_placements,
            key=lambda item: item[0],
        )
        if len(ranked_window) == 0:
            return

        target_k = min(self.curriculum_top_k, len(ranked_window))
        self.curriculum_prefix_pool = [
            copy.deepcopy(placement)
            for _, placement in ranked_window[:target_k]
        ]
        logging.info(
            "Prepared curriculum prefix pool with %d candidate placement(s) for the next window.",
            len(self.curriculum_prefix_pool),
        )

    def _select_curriculum_episode_prefix(self, env: Environment):
        self._clear_episode_prefix_plan()
        self.last_prefix_choice_idx = -1
        if not self.enable_curriculum:
            return {}
        if self.curriculum_fixed_macro_count <= 0:
            return {}
        if len(self.curriculum_prefix_pool) == 0:
            return {}
        if self.last_window_num_macros is None or self.last_window_num_macros <= 0:
            return {}

        fix_count = min(
            self.curriculum_fixed_macro_count,
            self.last_window_num_macros,
            env.num_macros_to_place,
            env.num_macros,
        )
        if fix_count <= 0:
            return {}

        transfer_mode = self._get_effective_prefix_transfer_mode()
        if transfer_mode == "consensus_soft":
            self.active_episode_prefix_boxes = self._build_consensus_soft_prefix_boxes(
                env=env,
                fix_count=fix_count,
            )
            self.active_episode_prefix_mode = "consensus_soft"
            self.active_episode_prefix_count = len(self.active_episode_prefix_boxes)
            if self.active_episode_prefix_count > 0:
                logging.info(
                    "Consensus-soft prefix active: constraining the first %d macro(s) "
                    "with median anchors derived from %d elite placement(s).",
                    self.active_episode_prefix_count,
                    len(self.curriculum_prefix_pool),
                )
            return {}

        strategy = str(self.curriculum_selection_strategy).lower()
        if strategy in {"best-1", "best", "top-1"}:
            prefix_idx = 0
        elif strategy in {"top-k", "sample-topk", "sample", "random-topk"}:
            prefix_idx = int(np.random.randint(len(self.curriculum_prefix_pool)))
        else:
            logging.warning(
                "Unknown curriculum_selection_strategy '%s'; falling back to best-1.",
                self.curriculum_selection_strategy,
            )
            prefix_idx = 0

        selected = self.curriculum_prefix_pool[prefix_idx]
        fixed_positions = {}
        for macro_name in env.macro_name_list[:fix_count]:
            anchor = selected.get(macro_name)
            if anchor is not None:
                fixed_positions[macro_name] = anchor

        self.last_prefix_choice_idx = prefix_idx
        self.active_episode_prefix_anchors = copy.deepcopy(fixed_positions)
        self.active_episode_prefix_count = len(fixed_positions)

        if transfer_mode in {"hard_fixing", "fixed"}:
            self.active_episode_prefix_mode = "fixed"
            return fixed_positions

        fix_phase_len = self._get_prefix_fix_phase_length()
        local_episode_idx = self.curriculum_manager.episodes_in_current_window
        if local_episode_idx < fix_phase_len:
            self.active_episode_prefix_mode = "fixed"
            return fixed_positions

        self.active_episode_prefix_mode = "released"
        self.active_episode_prefix_radius_ratio = (
            self.curriculum_prefix_release_constraint_ratio
            if self.curriculum_prefix_release_constraint_ratio is not None
            else self.constraint_radius_ratio
        )
        return {}

    def _append_placement_metrics_row(self, row):
        fieldnames = [
            "timestamp",
            "global_episode",
            "window_index",
            "local_episode_idx",
            "phase",
            "hpwl",
            "hpwl_from_fixed",
            "hpwl_ppo_window",
            "hpwl_greedy",
            "num_macros_placed",
            "dead_end",
            "qd_quality_eligible",
            "qd_novelty",
            "qd_axis_movement_x",
            "qd_axis_movement_y",
            "qd_reference_index",
            "qd_descriptor_x",
            "qd_descriptor_y",
            "qd_cell_x",
            "qd_cell_y",
            "qd_map_updated",
            "qd_map_occupied_cells",
            "qd_map_coverage",
            "ref_pool_updated",
            "ref_pool_best_hpwl",
            "ref_pool_worst_hpwl",
            "topk_inserted",
            "topk_rank",
            "topk_best_hpwl",
            "topk_worst_hpwl",
        ]
        metrics_path = Path(self.placement_metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = metrics_path.exists()
        with metrics_path.open("a", newline="", encoding="utf8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _write_run_summary(
        self,
        status,
        global_episode,
        loop_index,
        episodes_completed_in_loop,
        next_visualize_episode,
        total_train_time,
        error=None,
    ):
        summary_path = Path(self.run_summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        topk_summary_path = Path(self.global_topk_dir) / "topk_summary.json"
        payload = {
            "run_id": self.autoexp_run_id,
            "status": status,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "target_budget_episodes": self.target_budget_episodes,
            "completed_episodes": int(global_episode),
            "best_hpwl": None if not np.isfinite(self.best_hpwl) else float(self.best_hpwl),
            "best_episode": self.best_hpwl_episode,
            "next_loop_index": int(loop_index),
            "episodes_completed_in_loop": int(episodes_completed_in_loop),
            "next_visualize_episode": int(next_visualize_episode),
            "total_train_time_sec": float(total_train_time),
            "placement_metrics_path": self.placement_metrics_path,
            "model_dir": str(Path(self.model_dir).resolve()),
            "solution_dir": str(Path(self.solution_dir).resolve()),
            "global_topk_dir": str(Path(self.global_topk_dir).resolve()),
            "global_topk_summary_path": str(topk_summary_path.resolve()) if topk_summary_path.exists() else None,
            "curriculum": self.curriculum_manager.get_current_state() if self.enable_curriculum else None,
            "error": error,
        }
        summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")

    def _reset_diversity_window_state(self, window_index):
        self.diversity_window_index = window_index
        self.warmup_episode_pool = []
        self.frozen_reference_pool = []
        self.diversity_episode_pool = []
        self.warmup_best_hpwl = float('inf')
        self.frozen_refs_ready = False
        self.window_qd_map.reset()

    def _get_window_phase_info(self):
        if not (self.enable_curriculum and self.enable_diversity_phase):
            return "disabled", 0, 0
        local_episode_idx = self.curriculum_manager.episodes_in_current_window
        warmup_len = int(np.floor(self.curriculum_manager.window_size * self.diversity_phase_ratio))
        phase = "warmup" if local_episode_idx < warmup_len else "diversity"
        return phase, local_episode_idx, warmup_len

    def _make_default_diversity_stats(self, env: Environment):
        phase, local_episode_idx, warmup_len = self._get_window_phase_info()
        macro_names_scope, scope_is_prefix = self._get_diversity_scope_macro_names(env)
        return {
            "phase": phase,
            "local_episode_idx": local_episode_idx,
            "warmup_len": warmup_len,
            "reference_pool_size": len(self.frozen_reference_pool),
            "quality_eligible": 0.0,
            "novelty": 0.0,
            "bonus_raw": 0.0,
            "bonus_capped": 0.0,
            "archive_pool_size": len(self.diversity_episode_pool),
            "pairwise_archive_diversity": 0.0,
            "axis_movement_x": 0.0,
            "axis_movement_y": 0.0,
            "reference_index": -1.0,
            "descriptor_x": float("nan"),
            "descriptor_y": float("nan"),
            "cell_x": -1.0,
            "cell_y": -1.0,
            "map_updated": 0.0,
            "map_occupied_cells": float(self.window_qd_map.occupied_cells),
            "map_total_cells": float(self.window_qd_map.num_cells),
            "map_coverage": self.window_qd_map.coverage_ratio(),
            "ref_pool_updated": 0.0,
            "ref_pool_best_hpwl": (
                float(self.frozen_reference_pool[0][0]) if len(self.frozen_reference_pool) > 0 else float("nan")
            ),
            "ref_pool_worst_hpwl": (
                float(self.frozen_reference_pool[-1][0]) if len(self.frozen_reference_pool) > 0 else float("nan")
            ),
            "scope_is_prefix": 1.0 if scope_is_prefix else 0.0,
            "scope_macro_count": float(len(macro_names_scope)),
            "dead_end": 0.0,
        }

    def _state_has_no_valid_actions(self, state: torch.Tensor) -> bool:
        return bool(torch.all(state[2] > 0.5).item())

    def _get_diversity_scope_macro_names(self, env: Environment):
        if not self.enable_curriculum:
            return [], False
        current_limit = min(env.num_macros_to_place, env.num_macros)
        if current_limit <= 0:
            return [], False
        if (
            self.enable_prefix_diversity_window0
            and self.curriculum_manager.current_window == 0
            and env.num_fixed_macros == 0
        ):
            scope_end = min(self.protected_macro_count, current_limit)
            return env.macro_name_list[:scope_end], True
        scope_begin = max(
            min(self.protected_macro_count, current_limit),
            min(env.num_fixed_macros, current_limit),
        )
        if scope_begin >= current_limit:
            return [], False
        return env.macro_name_list[scope_begin:current_limit], False

    def _compute_macro_ndd(self, placement_a, placement_b):
        ax, ay, aw, ah = placement_a
        bx, by, _, _ = placement_b
        diag = max(1.0, float(np.sqrt(aw ** 2 + ah ** 2)))
        return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2) / diag)

    def _compute_weighted_ndd_components(self, placement_a, placement_b, macro_names_scope):
        if len(macro_names_scope) == 0:
            return {
                "novelty": 0.0,
                "axis_movement_x": 0.0,
                "axis_movement_y": 0.0,
            }

        macro_data = []
        total_area = 0.0
        for macro_name in macro_names_scope:
            if macro_name not in placement_a or macro_name not in placement_b:
                continue
            ax, ay, w, h = placement_a[macro_name]
            bx, by, _, _ = placement_b[macro_name]
            area = float(max(1, w * h))
            diag = max(1.0, float(np.sqrt(w ** 2 + h ** 2)))
            displacement = np.asarray(
                [
                    (float(ax) - float(bx)) / diag,
                    (float(ay) - float(by)) / diag,
                ],
                dtype=np.float64,
            )
            macro_data.append((area, displacement))
            total_area += area

        if total_area <= 0.0:
            return {
                "novelty": 0.0,
                "axis_movement_x": 0.0,
                "axis_movement_y": 0.0,
            }

        novelty = 0.0
        axis_movement_x = 0.0
        axis_movement_y = 0.0
        for area, displacement in macro_data:
            weight = area / total_area
            novelty += weight * float(np.linalg.norm(displacement))
            axis_movement_x += weight * abs(float(displacement[0]))
            axis_movement_y += weight * abs(float(displacement[1]))
        return {
            "novelty": float(novelty),
            "axis_movement_x": float(axis_movement_x),
            "axis_movement_y": float(axis_movement_y),
        }

    def _compute_weighted_ndd_distance(self, placement_a, placement_b, macro_names_scope):
        return self._compute_weighted_ndd_components(
            placement_a,
            placement_b,
            macro_names_scope,
        )["novelty"]

    def _normalize_qd_axis(self, value: float) -> float:
        value = max(0.0, float(value))
        return float(value / (1.0 + value))

    def _find_nearest_reference_for_qd(self, final_placement, macro_names_scope):
        if len(self.frozen_reference_pool) == 0:
            return None

        best_match = None
        for reference_index, (reference_hpwl, reference_placement) in enumerate(self.frozen_reference_pool):
            components = self._compute_weighted_ndd_components(
                final_placement,
                reference_placement,
                macro_names_scope,
            )
            candidate = {
                **components,
                "descriptor": (
                    self._normalize_qd_axis(components["axis_movement_x"]),
                    self._normalize_qd_axis(components["axis_movement_y"]),
                ),
                "reference_index": int(reference_index),
                "reference_hpwl": float(reference_hpwl),
                "reference_placement": reference_placement,
            }
            if best_match is None:
                best_match = candidate
                continue
            novelty_delta = candidate["novelty"] - best_match["novelty"]
            if novelty_delta < -1e-12 or (
                abs(novelty_delta) <= 1e-12
                and candidate["reference_hpwl"] < best_match["reference_hpwl"]
            ):
                best_match = candidate

        return best_match

    def _build_adaptive_qd_axis_edges(self, descriptor_values, num_bins):
        descriptor_values = np.asarray(descriptor_values, dtype=np.float64)
        if descriptor_values.size < 2:
            return np.linspace(0.0, 1.0, int(num_bins) + 1, dtype=np.float64)

        clipped_values = np.clip(descriptor_values, 0.0, 1.0)
        quantiles = np.linspace(0.0, 1.0, int(num_bins) + 1, dtype=np.float64)
        edges = np.quantile(clipped_values, quantiles)
        edges[0] = 0.0
        edges[-1] = 1.0

        # Quantiles can collapse when descriptors cluster tightly. Spread those
        # degenerate cuts just enough to preserve a valid monotone partition
        # while keeping the empirical shape of the axis.
        min_step = 1e-6
        for idx in range(1, len(edges)):
            remaining = len(edges) - 1 - idx
            max_allowed = 1.0 - remaining * min_step
            edges[idx] = min(max(float(edges[idx]), float(edges[idx - 1] + min_step)), max_allowed)

        edges[0] = 0.0
        edges[-1] = 1.0
        return edges.astype(np.float64, copy=False)

    def _fit_window_qd_bins(self, episode_pool, macro_names_scope, window_index):
        if len(episode_pool) == 0 or len(macro_names_scope) == 0 or len(self.frozen_reference_pool) == 0:
            return

        descriptors = []
        for _, placement in episode_pool:
            nearest_reference = self._find_nearest_reference_for_qd(
                placement,
                macro_names_scope,
            )
            if nearest_reference is None:
                continue
            descriptors.append(nearest_reference["descriptor"])

        if len(descriptors) == 0:
            return

        descriptor_array = np.asarray(descriptors, dtype=np.float64)
        bin_edges_x = self._build_adaptive_qd_axis_edges(descriptor_array[:, 0], self.qd_map_bins)
        bin_edges_y = self._build_adaptive_qd_axis_edges(descriptor_array[:, 1], self.qd_map_bins)
        self.window_qd_map.set_bin_edges(bin_edges_x, bin_edges_y)

        occupied_preview = {
            self.window_qd_map.descriptor_to_cell((float(descriptor[0]), float(descriptor[1])))
            for descriptor in descriptor_array
        }
        logging.info(
            f"Window {window_index}: fitted adaptive QD bins from {len(descriptor_array)} warmup descriptors "
            f"(preview coverage={len(occupied_preview)}/{self.window_qd_map.num_cells})."
        )

    def _compute_qd_reward_scale(self, descriptor, final_hpwl):
        cell_probe = self.window_qd_map.probe(descriptor)
        incumbent = cell_probe["entry"]
        if incumbent is None:
            return 1.0, cell_probe
        if final_hpwl < incumbent.hpwl - 1e-12:
            return 1.0, cell_probe
        return float(1.0 / np.sqrt(1.0 + cell_probe["visit_count"])), cell_probe

    def _compute_pairwise_mean_distance(self, episode_pool, macro_names_scope):
        if len(episode_pool) < 2:
            return 0.0
        dist_sum = 0.0
        pair_count = 0
        for i in range(len(episode_pool)):
            for j in range(i + 1, len(episode_pool)):
                dist_sum += self._compute_weighted_ndd_distance(
                    episode_pool[i][1], episode_pool[j][1], macro_names_scope
                )
                pair_count += 1
        return dist_sum / max(1, pair_count)

    def _select_diverse_archive(self, episode_pool, macro_names_scope, top_k):
        if len(episode_pool) == 0 or top_k <= 0:
            return []

        sorted_pool = sorted(episode_pool, key=lambda item: item[0])
        target_k = min(top_k, len(sorted_pool))
        selected = [sorted_pool[0]]

        while len(selected) < target_k:
            best_candidate = None
            best_score = -1.0
            best_hpwl = float('inf')

            for candidate in sorted_pool:
                if candidate in selected:
                    continue
                min_distance = min(
                    self._compute_weighted_ndd_distance(
                        candidate[1], chosen[1], macro_names_scope
                    )
                    for chosen in selected
                )
                if min_distance > best_score or (
                    abs(min_distance - best_score) <= 1e-12 and candidate[0] < best_hpwl
                ):
                    best_candidate = candidate
                    best_score = min_distance
                    best_hpwl = candidate[0]

            if best_candidate is None:
                break
            selected.append(best_candidate)

        return selected

    def _ensure_diversity_state_ready(self, env: Environment):
        if not (self.enable_curriculum and self.enable_diversity_phase):
            return

        current_window = self.curriculum_manager.current_window
        if self.diversity_window_index != current_window:
            self._reset_diversity_window_state(current_window)

        phase, _, warmup_len = self._get_window_phase_info()
        if phase != "diversity" or self.frozen_refs_ready:
            return

        macro_names_scope, scope_is_prefix = self._get_diversity_scope_macro_names(env)
        scope_mode = "prefix" if scope_is_prefix else "suffix"

        if warmup_len <= 0:
            self.frozen_reference_pool = []
            self.frozen_refs_ready = True
            logging.info(
                f"Window {current_window}: diversity phase starts without warmup references "
                f"(scope={scope_mode}, macros={len(macro_names_scope)})."
            )
            return

        if len(self.warmup_episode_pool) == 0 or not np.isfinite(self.warmup_best_hpwl):
            self.frozen_reference_pool = []
            self.frozen_refs_ready = True
            logging.warning(
                f"Window {current_window}: no warmup placements available for diversity references "
                f"(scope={scope_mode}, macros={len(macro_names_scope)})."
            )
            return

        gate_threshold = self.warmup_best_hpwl * (1.0 + self.diversity_quality_eps)
        gated_pool = [item for item in self.warmup_episode_pool if item[0] <= gate_threshold]
        gated_pool.sort(key=lambda item: item[0])
        self.frozen_reference_pool = gated_pool[:self.diversity_reference_k]
        self._fit_window_qd_bins(
            episode_pool=gated_pool,
            macro_names_scope=macro_names_scope,
            window_index=current_window,
        )
        self.frozen_refs_ready = True
        logging.info(
            f"Window {current_window}: built initial diversity reference pool with "
            f"{len(self.frozen_reference_pool)} placements (warmup best HPWL {self.warmup_best_hpwl:.4f}, "
            f"scope={scope_mode}, macros={len(macro_names_scope)})."
        )

    def _compute_diversity_bonus(self, env: Environment, final_hpwl, final_placement, terminal_reward):
        diversity_stats = self._make_default_diversity_stats(env)
        phase = diversity_stats["phase"]

        if not (self.enable_curriculum and self.enable_diversity_phase):
            return 0.0, diversity_stats
        if phase != "diversity":
            return 0.0, diversity_stats
        if len(self.frozen_reference_pool) < self.diversity_min_refs:
            return 0.0, diversity_stats
        if not np.isfinite(self.warmup_best_hpwl):
            return 0.0, diversity_stats

        gate_threshold = self.warmup_best_hpwl * (1.0 + self.diversity_quality_eps)
        if final_hpwl > gate_threshold:
            return 0.0, diversity_stats

        macro_names_scope, _ = self._get_diversity_scope_macro_names(env)
        if len(macro_names_scope) == 0:
            return 0.0, diversity_stats

        nearest_reference = self._find_nearest_reference_for_qd(
            final_placement,
            macro_names_scope,
        )
        if nearest_reference is None:
            return 0.0, diversity_stats

        qd_reward_scale, cell_probe = self._compute_qd_reward_scale(
            descriptor=nearest_reference["descriptor"],
            final_hpwl=final_hpwl,
        )
        novelty = nearest_reference["novelty"]
        bonus_raw = self.diversity_bonus_lambda * novelty * qd_reward_scale
        bonus_cap = self.diversity_bonus_cap_ratio * abs(float(terminal_reward))
        bonus_capped = min(bonus_raw, bonus_cap)

        diversity_stats.update({
            "quality_eligible": 1.0,
            "novelty": novelty,
            "bonus_raw": bonus_raw,
            "bonus_capped": bonus_capped,
            "axis_movement_x": nearest_reference["axis_movement_x"],
            "axis_movement_y": nearest_reference["axis_movement_y"],
            "reference_index": float(nearest_reference["reference_index"]),
            "descriptor_x": nearest_reference["descriptor"][0],
            "descriptor_y": nearest_reference["descriptor"][1],
            "cell_x": float(cell_probe["cell"][0]),
            "cell_y": float(cell_probe["cell"][1]),
        })
        return bonus_capped, diversity_stats

    def _update_posthoc_qd_map(
        self,
        env: Environment,
        final_hpwl,
        final_placement,
        global_episode,
        diversity_stats,
    ):
        # Reward probes the current QD cell before this update. This post-episode
        # update persists the elite and visit count so later episodes can use it.
        if diversity_stats["quality_eligible"] <= 0.5:
            return diversity_stats

        macro_names_scope, _ = self._get_diversity_scope_macro_names(env)
        if len(macro_names_scope) == 0:
            return diversity_stats

        descriptor = (
            float(diversity_stats["descriptor_x"]),
            float(diversity_stats["descriptor_y"]),
        )
        reference_index = int(diversity_stats["reference_index"])
        reference_hpwl = None
        if (
            not np.isfinite(descriptor[0])
            or not np.isfinite(descriptor[1])
            or reference_index < 0
        ):
            nearest_reference = self._find_nearest_reference_for_qd(
                final_placement,
                macro_names_scope,
            )
            if nearest_reference is None:
                return diversity_stats
            descriptor = nearest_reference["descriptor"]
            reference_index = nearest_reference["reference_index"]
            reference_hpwl = nearest_reference["reference_hpwl"]
            diversity_stats.update({
                "axis_movement_x": nearest_reference["axis_movement_x"],
                "axis_movement_y": nearest_reference["axis_movement_y"],
                "reference_index": float(reference_index),
                "descriptor_x": descriptor[0],
                "descriptor_y": descriptor[1],
            })
        elif 0 <= reference_index < len(self.frozen_reference_pool):
            reference_hpwl = float(self.frozen_reference_pool[reference_index][0])

        map_update = self.window_qd_map.add_or_update(
            hpwl=final_hpwl,
            placement=final_placement,
            descriptor=descriptor,
            novelty=diversity_stats["novelty"],
            reference_index=reference_index,
            reference_hpwl=reference_hpwl,
            episode=global_episode,
            local_episode_idx=diversity_stats["local_episode_idx"],
        )
        diversity_stats.update({
            "cell_x": float(map_update["cell"][0]),
            "cell_y": float(map_update["cell"][1]),
            "map_updated": map_update["updated"],
            "map_occupied_cells": float(map_update["occupied_cells"]),
            "map_total_cells": float(self.window_qd_map.num_cells),
            "map_coverage": map_update["coverage"],
        })
        return diversity_stats

    def _maybe_update_reference_pool(self, hpwl: float, placement: dict, diversity_stats: dict):
        if diversity_stats["quality_eligible"] <= 0.5:
            return diversity_stats

        placement_copy = placement.copy()
        placement_signature = self._make_placement_signature(placement_copy)

        existing_idx = None
        for idx, (_, reference_placement) in enumerate(self.frozen_reference_pool):
            if self._make_placement_signature(reference_placement) == placement_signature:
                existing_idx = idx
                break

        updated = 0.0
        if existing_idx is not None:
            if hpwl < self.frozen_reference_pool[existing_idx][0] - 1e-12:
                self.frozen_reference_pool[existing_idx] = (float(hpwl), placement_copy)
                updated = 1.0
        elif len(self.frozen_reference_pool) < self.diversity_reference_k:
            self.frozen_reference_pool.append((float(hpwl), placement_copy))
            updated = 1.0
        elif len(self.frozen_reference_pool) > 0 and hpwl < self.frozen_reference_pool[-1][0] - 1e-12:
            self.frozen_reference_pool[-1] = (float(hpwl), placement_copy)
            updated = 1.0

        if updated > 0.5:
            self.frozen_reference_pool.sort(key=lambda item: item[0])

        diversity_stats.update({
            "ref_pool_updated": updated,
            "reference_pool_size": float(len(self.frozen_reference_pool)),
            "ref_pool_best_hpwl": (
                float(self.frozen_reference_pool[0][0]) if len(self.frozen_reference_pool) > 0 else float("nan")
            ),
            "ref_pool_worst_hpwl": (
                float(self.frozen_reference_pool[-1][0]) if len(self.frozen_reference_pool) > 0 else float("nan")
            ),
        })
        return diversity_stats

    def _record_episode_for_diversity(self, env: Environment, hpwl: float, placement: dict):
        if not (self.enable_curriculum and self.enable_diversity_phase):
            return

        phase, _, warmup_len = self._get_window_phase_info()
        placement_copy = placement.copy()

        if phase == "warmup":
            self.warmup_episode_pool.append((hpwl, placement_copy))
            self.warmup_best_hpwl = min(self.warmup_best_hpwl, hpwl)
            return

        if warmup_len <= 0 or not np.isfinite(self.warmup_best_hpwl):
            return

        gate_threshold = self.warmup_best_hpwl * (1.0 + self.diversity_quality_eps)
        if hpwl <= gate_threshold:
            self.diversity_episode_pool.append((hpwl, placement_copy))

    def _save_window_diversity_archive(self, env: Environment):
        if not (self.enable_curriculum and self.enable_diversity_phase):
            return

        current_window = self.curriculum_manager.current_window
        macro_names_scope, scope_is_prefix = self._get_diversity_scope_macro_names(env)
        scope_mode = "prefix" if scope_is_prefix else "suffix"
        archive = self._select_diverse_archive(
            self.diversity_episode_pool,
            macro_names_scope,
            self.diversity_output_k,
        )
        archive_pairwise_diversity = self._compute_pairwise_mean_distance(archive, macro_names_scope)

        archive_dir = os.path.join(self.solution_dir, "diversity_archives")
        os.makedirs(archive_dir, exist_ok=True)
        archive_path = os.path.join(archive_dir, f"window_{current_window}_archive.pkl")
        archive_payload = {
            "window_index": current_window,
            "warmup_best_hpwl": self.warmup_best_hpwl,
            "reference_pool_size": len(self.frozen_reference_pool),
            "scope_mode": scope_mode,
            "scope_macro_count": len(macro_names_scope),
            "archive_pairwise_diversity": archive_pairwise_diversity,
            "entries": [
                {"hpwl": hpwl, "placement": placement}
                for hpwl, placement in archive
            ],
        }
        with open(archive_path, 'wb') as f:
            pickle.dump(archive_payload, f)

        elite_map_path = os.path.join(archive_dir, f"window_{current_window}_elite_map.pkl")
        elite_map_payload = self.window_qd_map.export_payload(top_k=self.diversity_output_k)
        elite_map_payload.update({
            "window_index": current_window,
            "warmup_best_hpwl": self.warmup_best_hpwl,
            "reference_pool_size": len(self.frozen_reference_pool),
            "scope_mode": scope_mode,
            "scope_macro_count": len(macro_names_scope),
            "binning_mode": "adaptive" if self.window_qd_map.uses_adaptive_bins else "uniform",
            "descriptor_axes": {
                "x": "axis_movement_x",
                "y": "axis_movement_y",
                "normalization": "value/(1+value)",
            },
        })
        with open(elite_map_path, "wb") as f:
            pickle.dump(elite_map_payload, f)
        self._visualize_window_elite_map(
            current_window=current_window,
            save_path=os.path.join(archive_dir, f"window_{current_window}_elite_map.png"),
        )

        self._queue_wandb_metrics({
            "diversity/window_archive_size": len(archive),
            "diversity/window_reference_pool_size": len(self.frozen_reference_pool),
            "diversity/window_archive_pairwise_diversity": archive_pairwise_diversity,
            "diversity/window_scope_is_prefix": 1.0 if scope_is_prefix else 0.0,
            "diversity/window_scope_macro_count": float(len(macro_names_scope)),
            "diversity/window_index": current_window,
            "diversity/window_qd_map_occupied_cells": float(self.window_qd_map.occupied_cells),
            "diversity/window_qd_map_coverage": self.window_qd_map.coverage_ratio(),
        })
        logging.info(
            f"Window {current_window}: saved diversity archive with {len(archive)} placements "
            f"(scope={scope_mode}, macros={len(macro_names_scope)}) to {archive_path}"
        )
        logging.info(
            f"Window {current_window}: saved 2D axis-movement elite map with {self.window_qd_map.occupied_cells}/"
            f"{self.window_qd_map.num_cells} occupied cells to {elite_map_path}"
        )

    def _visualize_window_elite_map(self, current_window: int, save_path: str):
        bins_x = self.window_qd_map.bins_x
        bins_y = self.window_qd_map.bins_y
        bin_edges_x = self.window_qd_map.bin_edges_x
        bin_edges_y = self.window_qd_map.bin_edges_y
        hpwl_grid = np.full((bins_y, bins_x), np.nan, dtype=np.float64)
        for (cell_x, cell_y), entry in self.window_qd_map.iter_entries():
            hpwl_grid[cell_y, cell_x] = entry.hpwl

        masked_grid = np.ma.masked_invalid(hpwl_grid)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="#f0f0f0")

        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        if masked_grid.count() > 0:
            heatmap = ax.pcolormesh(
                bin_edges_x,
                bin_edges_y,
                masked_grid,
                cmap=cmap,
                shading="flat",
                edgecolors="white",
                linewidth=0.6,
            )
        else:
            heatmap = ax.pcolormesh(
                bin_edges_x,
                bin_edges_y,
                np.zeros((bins_y, bins_x), dtype=np.float64),
                cmap=cmap,
                shading="flat",
                edgecolors="white",
                linewidth=0.6,
                vmin=0.0,
                vmax=1.0,
            )

        binning_label = "Adaptive" if self.window_qd_map.uses_adaptive_bins else "Uniform"
        ax.set_title(
            f"Window {current_window} 2D Axis-Movement Elite Map ({binning_label} Bins)\n"
            f"occupied={self.window_qd_map.occupied_cells}/{self.window_qd_map.num_cells}"
        )
        ax.set_xlabel("X-axis Movement")
        ax.set_ylabel("Y-axis Movement")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(np.linspace(0.0, 1.0, 5))
        ax.set_yticks(np.linspace(0.0, 1.0, 5))

        for cell_y in range(bins_y):
            for cell_x in range(bins_x):
                value = hpwl_grid[cell_y, cell_x]
                if np.isnan(value):
                    continue
                center_x = 0.5 * (bin_edges_x[cell_x] + bin_edges_x[cell_x + 1])
                center_y = 0.5 * (bin_edges_y[cell_y] + bin_edges_y[cell_y + 1])
                ax.text(
                    center_x,
                    center_y,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )

        cbar = fig.colorbar(heatmap, ax=ax, shrink=0.9)
        cbar.set_label("Elite HPWL")
        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

    def _make_placement_signature(self, placement):
        return tuple(
            (macro_name, int(x), int(y), int(w), int(h))
            for macro_name, (x, y, w, h) in sorted(placement.items())
        )

    def _serialize_placement_for_json(self, placement):
        return {
            macro_name: [int(x), int(y), int(w), int(h)]
            for macro_name, (x, y, w, h) in sorted(placement.items())
        }

    def _persist_global_topk_archive(self):
        for filename in os.listdir(self.global_topk_dir):
            if filename.startswith("rank_") and filename.endswith(".pkl"):
                os.remove(os.path.join(self.global_topk_dir, filename))

        metadata_entries = []
        for rank, entry in enumerate(self.global_topk_archive, start=1):
            payload = {
                "rank": rank,
                "hpwl": entry["hpwl"],
                "episode": entry["episode"],
                "window_index": entry["window_index"],
                "local_episode_idx": entry["local_episode_idx"],
                "phase": entry["phase"],
                "num_macros_placed": entry["num_macros_placed"],
                "placement": entry["placement"],
            }
            rank_path = os.path.join(
                self.global_topk_dir,
                f"rank_{rank:02d}_hpwl_{entry['hpwl']:.4f}_ep_{entry['episode']}.pkl",
            )
            with open(rank_path, "wb") as f:
                pickle.dump(payload, f)

            metadata_entries.append({
                "rank": rank,
                "hpwl": entry["hpwl"],
                "episode": entry["episode"],
                "window_index": entry["window_index"],
                "local_episode_idx": entry["local_episode_idx"],
                "phase": entry["phase"],
                "num_macros_placed": entry["num_macros_placed"],
                "path": os.path.basename(rank_path),
                "placement": self._serialize_placement_for_json(entry["placement"]),
            })

        archive_payload = {
            "topk_size": self.global_topk_size,
            "entries": [
                {
                    "rank": idx + 1,
                    "hpwl": entry["hpwl"],
                    "episode": entry["episode"],
                    "window_index": entry["window_index"],
                    "local_episode_idx": entry["local_episode_idx"],
                    "phase": entry["phase"],
                    "num_macros_placed": entry["num_macros_placed"],
                    "placement": entry["placement"],
                }
                for idx, entry in enumerate(self.global_topk_archive)
            ],
        }
        with open(os.path.join(self.global_topk_dir, "topk_archive.pkl"), "wb") as f:
            pickle.dump(archive_payload, f)
        with open(os.path.join(self.global_topk_dir, "topk_summary.json"), "w") as f:
            json.dump(
                {
                    "topk_size": self.global_topk_size,
                    "entries": metadata_entries,
                },
                f,
                indent=2,
            )

    def _update_global_topk_archive(
        self,
        hpwl,
        placement,
        global_episode,
        window_index,
        local_episode_idx,
        phase,
        num_macros_placed,
    ):
        stats = {
            "topk/archive_size": float(len(self.global_topk_archive)),
            "topk/inserted": 0.0,
            "topk/rank": 0.0,
            "topk/best_hpwl": float(self.global_topk_archive[0]["hpwl"]) if self.global_topk_archive else float("nan"),
            "topk/worst_hpwl": float(self.global_topk_archive[-1]["hpwl"]) if self.global_topk_archive else float("nan"),
        }

        if not np.isfinite(hpwl):
            return stats

        placement_copy = copy.deepcopy(placement)
        candidate_signature = self._make_placement_signature(placement_copy)

        existing_idx = None
        for idx, entry in enumerate(self.global_topk_archive):
            if entry["signature"] == candidate_signature:
                existing_idx = idx
                break

        changed = False
        inserted_rank = None

        if existing_idx is not None:
            if hpwl < self.global_topk_archive[existing_idx]["hpwl"] - 1e-12:
                self.global_topk_archive[existing_idx].update({
                    "hpwl": float(hpwl),
                    "episode": int(global_episode),
                    "window_index": None if window_index is None else int(window_index),
                    "local_episode_idx": None if local_episode_idx is None else int(local_episode_idx),
                    "phase": phase,
                    "num_macros_placed": int(num_macros_placed),
                    "placement": placement_copy,
                })
                changed = True
        else:
            cutoff_hpwl = (
                self.global_topk_archive[-1]["hpwl"]
                if len(self.global_topk_archive) >= self.global_topk_size
                else float("inf")
            )
            if len(self.global_topk_archive) < self.global_topk_size or hpwl < cutoff_hpwl - 1e-12:
                self.global_topk_archive.append({
                    "hpwl": float(hpwl),
                    "episode": int(global_episode),
                    "window_index": None if window_index is None else int(window_index),
                    "local_episode_idx": None if local_episode_idx is None else int(local_episode_idx),
                    "phase": phase,
                    "num_macros_placed": int(num_macros_placed),
                    "placement": placement_copy,
                    "signature": candidate_signature,
                })
                changed = True

        if changed:
            self.global_topk_archive.sort(key=lambda entry: (entry["hpwl"], entry["episode"]))
            self.global_topk_archive = self.global_topk_archive[:self.global_topk_size]

            for idx, entry in enumerate(self.global_topk_archive):
                if entry["signature"] == candidate_signature:
                    inserted_rank = idx + 1
                    break

            self._persist_global_topk_archive()
            if inserted_rank is not None:
                logging.info(
                    f"  [TopK] Added/updated placement at rank {inserted_rank}/{self.global_topk_size} "
                    f"with HPWL {hpwl:.4f}"
                )

        if self.global_topk_archive:
            stats.update({
                "topk/archive_size": float(len(self.global_topk_archive)),
                "topk/inserted": 1.0 if inserted_rank is not None else 0.0,
                "topk/rank": float(inserted_rank) if inserted_rank is not None else 0.0,
                "topk/best_hpwl": float(self.global_topk_archive[0]["hpwl"]),
                "topk/worst_hpwl": float(self.global_topk_archive[-1]["hpwl"]),
            })
        return stats

    def _build_constrained_position_mask(self, position_mask, anchor_x, anchor_y, radius_x, radius_y, grid_size, macro_name, macro_idx):
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
                    f"Hard constraint fallback to original mask for macro {macro_name} "
                    f"(index {macro_idx}) because no valid location exists in the expanded local box."
                )
                return position_mask.clone()

            min_x = max(0, min_x - 1)
            max_x = min(grid_size - 1, max_x + 1)
            min_y = max(0, min_y - 1)
            max_y = min(grid_size - 1, max_y + 1)

    def _initialize_adaptive_large_macro_cache(self, env: Environment):
        if not self.enable_adaptive_hard_constraint:
            return
        if self.adaptive_large_macro_names is not None:
            return
        if self.adaptive_large_macro_count <= 0:
            self.adaptive_large_macro_names = set()
            return

        macro_names = list(env.placedb.node_info.keys())
        macro_names.sort(
            key=lambda macro_name: (
                env.placedb.node_info[macro_name]["x"] * env.placedb.node_info[macro_name]["y"]
            ),
            reverse=True,
        )
        selected = macro_names[:self.adaptive_large_macro_count]
        self.adaptive_large_macro_names = set(selected)
        logging.info(
            "Adaptive hard constraint: protecting %d largest macro(s): %s",
            len(selected),
            ", ".join(selected),
        )

    def _should_apply_adaptive_large_macro_constraint(self, macro_idx: int):
        if not self.enable_adaptive_hard_constraint:
            return False
        if self.last_window_num_macros is None:
            return False
        return macro_idx < self.last_window_num_macros

    def _resolve_constraint_radius_ratio(self, env: Environment, macro_idx: int):
        prefix_limit = min(self.protected_macro_count, env.num_macros)
        macro_name = env.macro_name_list[macro_idx]
        is_prefix_macro = macro_idx < prefix_limit

        is_large_macro = False
        if self.enable_adaptive_hard_constraint:
            self._initialize_adaptive_large_macro_cache(env)
            is_large_macro = macro_name in self.adaptive_large_macro_names

        if is_prefix_macro:
            return self.constraint_radius_ratio

        if not is_large_macro:
            return None

        if self._should_apply_adaptive_large_macro_constraint(macro_idx):
            return self.constraint_radius_ratio
        return None

    def _apply_hard_constraint(self, env: Environment, state: torch.Tensor):
        if not (self.enable_curriculum and self.enable_hard_constraint):
            return state
        constrained_state = state.clone()
        if env.t >= env.num_macros:
            return constrained_state

        macro_idx = env.t
        macro_name = env.macro_name_list[macro_idx]
        anchor = None
        radius_ratio = None
        radius_x = None
        radius_y = None

        if (
            self.active_episode_prefix_mode == "consensus_soft"
            and macro_idx < self.active_episode_prefix_count
            and len(self.active_episode_prefix_boxes) > 0
        ):
            box = self.active_episode_prefix_boxes.get(macro_name)
            if box is not None:
                anchor = (box["anchor_x"], box["anchor_y"])
                radius_x = int(box["radius_x"])
                radius_y = int(box["radius_y"])

        if (
            self.active_episode_prefix_mode == "released"
            and macro_idx < self.active_episode_prefix_count
            and len(self.active_episode_prefix_anchors) > 0
            and anchor is None
        ):
            anchor = self.active_episode_prefix_anchors.get(macro_name)
            radius_ratio = self.active_episode_prefix_radius_ratio

        if anchor is None:
            if self.last_window_best_placement is None:
                return constrained_state
            radius_ratio = self._resolve_constraint_radius_ratio(env, macro_idx)
            if radius_ratio is None:
                return constrained_state
            anchor = self.last_window_best_placement.get(macro_name)
            if anchor is None:
                return constrained_state

        if radius_x is None or radius_y is None:
            if radius_ratio is None:
                return constrained_state
            radius_x = max(self.constraint_min_radius, int(ceil(radius_ratio * env.size_x[macro_idx])))
            radius_y = max(self.constraint_min_radius, int(ceil(radius_ratio * env.size_y[macro_idx])))
        anchor_x, anchor_y = anchor[0], anchor[1]

        constrained_state[2] = self._build_constrained_position_mask(
            position_mask=state[2],
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            radius_x=radius_x,
            radius_y=radius_y,
            grid_size=env.grid,
            macro_name=macro_name,
            macro_idx=macro_idx,
        )
        return constrained_state

    def _visualize_window_transition(self, placement, hpwl, grid_size, window_idx, episode_idx, num_ppo_macros):
        """
        Visualize the best placement at window transition (EXPANDING WINDOW - NO FREEZING).

        Args:
            placement: Dictionary of macro positions
            hpwl: HPWL value
            grid_size: Grid size
            window_idx: Current window index
            episode_idx: Current episode index
            num_ppo_macros: Number of macros PPO will place in the next window
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        ax.set_title(f"Best Placement - Window {window_idx} → Window {window_idx+1}\n"
                     f"Total HPWL: {hpwl:.4f} | Episode: {episode_idx}\n"
                     f"Total Macros: {len(placement)} | PPO Window: 1-{num_ppo_macros} (NO FREEZING)",
                     fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        # Highlight macros inside the current PPO window versus the greedy remainder.
        for i, (macro_name, (x, y, w, h)) in enumerate(placement.items()):
            if i < num_ppo_macros:
                color = 'lightgreen'
                edgecolor = 'darkgreen'
                linewidth = 2
            else:
                color = 'lightblue'
                edgecolor = 'darkblue'
                linewidth = 1

            rect = patches.Rectangle((y, x), h, w, linewidth=linewidth,
                                    edgecolor=edgecolor, facecolor=color, alpha=0.7)
            ax.add_patch(rect)

            # Add macro index label for first 30 macros
            if i < 30:
                ax.text(y + h/2, x + w/2, str(i+1),
                       ha='center', va='center', fontsize=6, fontweight='bold')

        # Add legend
        from matplotlib.patches import Patch
        greedy_begin = num_ppo_macros + 1
        greedy_label = (
            f'Greedy remainder ({greedy_begin}-{len(placement)})'
            if num_ppo_macros < len(placement)
            else 'Greedy remainder (none)'
        )
        legend_elements = [
            Patch(facecolor='lightgreen', edgecolor='darkgreen', label=f'In PPO Window (1-{num_ppo_macros})'),
            Patch(facecolor='lightblue', edgecolor='darkblue', label=greedy_label),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Y', fontsize=12)
        ax.set_ylabel('X', fontsize=12)

        plt.tight_layout()

        # Save to curriculum directory
        vis_dir = os.path.join(self.model_dir, "../curriculum_visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        save_path = os.path.join(vis_dir, f"window_{window_idx}_transition_ep{episode_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logging.info(f"  Saved window transition visualization to: {save_path}")

    def evaluate_and_visualize(self, agent: Agent, env: Environment, loop_idx: int):
        logging.info(f"--- Starting Evaluation Rollouts at Step {loop_idx} ---")

        collected_placements = []
        collected_hpwls = []
        n_rollouts = 10

        # 1. Perform Rollouts
        for i in range(n_rollouts):
            env.set_fixed_macros(self._select_curriculum_episode_prefix(env))
            raw_s = env.reset()
            s = self._apply_hard_constraint(env, raw_s)
            t = env.num_fixed_macros  # Start from fixed macros
            done = False

            while not done:
                action, _ = agent.select_action(s, t - env.num_fixed_macros)  # Adjust time step
                raw_s, r, done, info = env.step(action)
                s = self._apply_hard_constraint(env, raw_s)
                t += 1

            if env.num_macros > env.num_macros_to_place:
                for _ in range(env.num_macros - env.num_macros_to_place):
                    action = agent.act_greedy(s)
                    raw_s, _, _, _ = env.step(action)
                    s = self._apply_hard_constraint(env, raw_s)

            collected_placements.append(copy.deepcopy(env.macro_pos))
            final_metric = env.calculate_hpwl_by_groups()["total"]
            collected_hpwls.append(final_metric)

        # 2. Compute Metrics Matrices
        # K_macro = np.zeros((n_rollouts, n_rollouts))
        # K_canvas = np.zeros((n_rollouts, n_rollouts))
        # dist_l1 = []
        #
        # for i in range(n_rollouts):
        #     for j in range(n_rollouts):
        #         # Macro-wise IoU (Similarity)
        #         K_macro[i, j] = calculate_macro_iou(collected_placements[i], collected_placements[j])
        #
        #         # Canvas-wise IoU (Similarity)
        #         K_canvas[i, j] = calculate_canvas_iou(collected_placements[i], collected_placements[j], env.grid)
        #
        #         # L1 Distance (Distance) - collect off-diagonals
        #         if i < j:
        #             d = calculate_l1_distance(collected_placements[i], collected_placements[j])
        #             dist_l1.append(d)
        #
        # # 3. Compute Scores
        # vendi_macro = compute_vendi_score_from_matrix(K_macro)
        # vendi_canvas = compute_vendi_score_from_matrix(K_canvas)
        # avg_l1_distance = np.mean(dist_l1) if len(dist_l1) > 0 else 0.0
        #
        # logging.info(f"Step {loop_idx} Stats:")
        # logging.info(f"  Vendi (Macro IoU):  {vendi_macro:.4f}")
        # logging.info(f"  Vendi (Canvas IoU): {vendi_canvas:.4f}")
        # logging.info(f"  Avg L1 Distance:    {avg_l1_distance:.4f}")
        #
        # # 4. Log to WandB and Tensorboard
        # log_dict = {
        #     "eval/vendi_macro_iou": vendi_macro,
        #     "eval/vendi_canvas_iou": vendi_canvas,
        #     "eval/avg_l1_distance": avg_l1_distance,
        #     "eval/step": loop_idx
        # }
        # self.tb_writer.add_scalar('Eval/Vendi_Macro_IoU', vendi_macro, loop_idx)
        # self.tb_writer.add_scalar('Eval/Vendi_Canvas_IoU', vendi_canvas, loop_idx)
        # self.tb_writer.add_scalar('Eval/Avg_L1_Distance', avg_l1_distance, loop_idx)

        # 5. Visualize
        vis_dir = os.path.join(self.model_dir, "../visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        visualize_placements(collected_placements, collected_hpwls, env.grid,
                           os.path.join(vis_dir, f"rollouts_step_{loop_idx}.png"), loop_idx)

        logging.info(f"Saved visualization to: {os.path.join(vis_dir, f'rollouts_step_{loop_idx}.png')}")
        return {}

    def _execute_training_episode(
        self,
        agent: Agent,
        env: Environment,
        replay_buffer: ReplayBuffer,
        global_episode: int,
        loop: int,
    ):
        hpwl, placement, hpwl_breakdown, diversity_stats = self.run_episode(
            env,
            agent,
            replay_buffer,
            global_episode,
        )

        num_macros_in_placement = len(placement)
        logging.info(f"Episode {global_episode}, Total HPWL = {hpwl:.4f} (Macros: {num_macros_in_placement})")
        logging.info(f"  ├─ HPWL from Fixed Macros:  {hpwl_breakdown['from_fixed']:.4f}")
        logging.info(f"  ├─ HPWL from PPO Window:    {hpwl_breakdown['ppo_window']:.4f}")
        logging.info(f"  └─ HPWL from Greedy:        {hpwl_breakdown['greedy']:.4f}")
        print(
            f"    -> Total: {hpwl:.4f} | Fixed: {hpwl_breakdown['from_fixed']:.4f} | "
            f"PPO: {hpwl_breakdown['ppo_window']:.4f} | Greedy: {hpwl_breakdown['greedy']:.4f}"
        )

        curriculum_log = {}
        if self.enable_curriculum:
            current_window = self.curriculum_manager.current_window
            if diversity_stats.get("dead_end", 0.0) < 0.5:
                self._record_episode_for_diversity(env, hpwl, placement)
            self.curriculum_manager.record_episode(hpwl, placement)
            current_episodes_in_window = self.curriculum_manager.episodes_in_current_window
            curriculum_log = {
                "curriculum/window": current_window,
                "curriculum/num_macros_to_place": self.curriculum_manager.current_num_macros,
                "curriculum/num_fixed_macros": env.num_fixed_macros,
                "curriculum/episodes_in_window": current_episodes_in_window,
                "curriculum/prefix_pool_size": len(self.curriculum_prefix_pool),
                "curriculum/prefix_choice": self.last_prefix_choice_idx,
                "curriculum/prefix_mode": self._get_prefix_mode_code(),
                "curriculum/prefix_active_count": float(self.active_episode_prefix_count),
                "curriculum/prefix_fix_phase_len": float(self._get_prefix_fix_phase_length()),
            }
        else:
            current_window = None
            current_episodes_in_window = None

        topk_stats = self._update_global_topk_archive(
            hpwl=hpwl,
            placement=placement,
            global_episode=global_episode,
            window_index=current_window,
            local_episode_idx=current_episodes_in_window,
            phase=diversity_stats["phase"],
            num_macros_placed=num_macros_in_placement,
        )

        if hpwl < self.best_hpwl:
            self.best_hpwl = hpwl
            self.best_placement = placement
            self.best_hpwl_episode = int(global_episode)
            logging.info(f"  [New Best] Global Best HPWL: {self.best_hpwl:.4f}")

            torch.save(agent.actor_net.state_dict(), os.path.join(self.model_dir, "actor_net.pth"))
            torch.save(agent.critic_net.state_dict(), os.path.join(self.model_dir, "critic_net.pth"))

            os.makedirs(self.solution_dir, exist_ok=True)
            with open(os.path.join(self.solution_dir, f"best_placement_{self.best_hpwl:.4f}.pkl"), 'wb') as f:
                pickle.dump(self.best_placement, f)
        else:
            logging.info(f"  [Best] Current Global Best HPWL: {self.best_hpwl:.4f}")

        self._append_placement_metrics_row(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "global_episode": int(global_episode),
                "window_index": curriculum_log.get("curriculum/window"),
                "local_episode_idx": diversity_stats["local_episode_idx"],
                "phase": diversity_stats["phase"],
                "hpwl": float(hpwl),
                "hpwl_from_fixed": float(hpwl_breakdown["from_fixed"]),
                "hpwl_ppo_window": float(hpwl_breakdown["ppo_window"]),
                "hpwl_greedy": float(hpwl_breakdown["greedy"]),
                "num_macros_placed": int(num_macros_in_placement),
                "dead_end": float(diversity_stats["dead_end"]),
                "qd_quality_eligible": float(diversity_stats["quality_eligible"]),
                "qd_novelty": float(diversity_stats["novelty"]),
                "qd_axis_movement_x": float(diversity_stats["axis_movement_x"]),
                "qd_axis_movement_y": float(diversity_stats["axis_movement_y"]),
                "qd_reference_index": float(diversity_stats["reference_index"]),
                "qd_descriptor_x": float(diversity_stats["descriptor_x"]),
                "qd_descriptor_y": float(diversity_stats["descriptor_y"]),
                "qd_cell_x": float(diversity_stats["cell_x"]),
                "qd_cell_y": float(diversity_stats["cell_y"]),
                "qd_map_updated": float(diversity_stats["map_updated"]),
                "qd_map_occupied_cells": float(diversity_stats["map_occupied_cells"]),
                "qd_map_coverage": float(diversity_stats["map_coverage"]),
                "ref_pool_updated": float(diversity_stats["ref_pool_updated"]),
                "ref_pool_best_hpwl": float(diversity_stats["ref_pool_best_hpwl"]),
                "ref_pool_worst_hpwl": float(diversity_stats["ref_pool_worst_hpwl"]),
                "topk_inserted": float(topk_stats["topk/inserted"]),
                "topk_rank": float(topk_stats["topk/rank"]),
                "topk_best_hpwl": float(topk_stats["topk/best_hpwl"]),
                "topk_worst_hpwl": float(topk_stats["topk/worst_hpwl"]),
            }
        )

        next_global_episode = global_episode + 1

        if self.enable_curriculum and self.curriculum_manager.should_advance_window():
            logging.info("=" * 60)
            logging.info("ADVANCING CURRICULUM WINDOW (EXPANDING WINDOW - NO FREEZING)")
            logging.info("=" * 60)

            best_hpwl, best_placement = self.curriculum_manager.get_best_placement_in_window()
            self.last_window_best_placement = best_placement.copy()
            self.last_window_num_macros = int(self.curriculum_manager.current_num_macros)
            self._refresh_curriculum_prefix_pool()
            self._save_window_diversity_archive(env)

            self._visualize_window_transition(
                best_placement,
                best_hpwl,
                env.grid,
                self.curriculum_manager.current_window,
                next_global_episode,
                self.curriculum_manager.current_num_macros,
            )

            new_num_macros, fixed_positions = self.curriculum_manager.advance_window()
            env.set_fixed_macros(fixed_positions)
            env.update_num_macros_to_place(new_num_macros)
            effective_fixed_count = min(
                self.curriculum_fixed_macro_count,
                self.last_window_num_macros if self.last_window_num_macros is not None else 0,
                new_num_macros,
                env.num_macros,
            )
            if len(self.curriculum_prefix_pool) > 0 and effective_fixed_count > 0:
                logging.info(
                    "Sequential-fix bridge active: next window will sample from %d prefix candidate(s) "
                    "and fix the first %d macro(s) per episode.",
                    len(self.curriculum_prefix_pool),
                    effective_fixed_count,
                )
            if self.enable_diversity_phase:
                self._reset_diversity_window_state(self.curriculum_manager.current_window)

            if new_num_macros > agent.actor_net.time_embedding.num_embeddings:
                logging.warning(
                    f"Agent's time embedding size ({agent.actor_net.time_embedding.num_embeddings}) "
                    f"is smaller than new num_macros ({new_num_macros}). "
                    f"Consider reinitializing agent or using larger initial embedding size."
                )

            logging.info("=" * 60)

        wandb.log(
            {
                "episode/hpwl": hpwl,
                "episode/hpwl_from_fixed": hpwl_breakdown['from_fixed'],
                "episode/hpwl_ppo_window": hpwl_breakdown['ppo_window'],
                "episode/hpwl_greedy": hpwl_breakdown['greedy'],
                "episode/num_macros_placed": num_macros_in_placement,
                "episode/global_episode": global_episode,
                "episode/loop": loop,
                "episode/best_hpwl": self.best_hpwl,
                "episode/dead_end": diversity_stats["dead_end"],
                "diversity/phase_is_diversity": 1.0 if diversity_stats["phase"] == "diversity" else 0.0,
                "diversity/local_episode_idx": diversity_stats["local_episode_idx"],
                "diversity/warmup_len": diversity_stats["warmup_len"],
                "diversity/reference_pool_size": diversity_stats["reference_pool_size"],
                "diversity/quality_eligible": diversity_stats["quality_eligible"],
                "diversity/novelty": diversity_stats["novelty"],
                "diversity/bonus_raw": diversity_stats["bonus_raw"],
                "diversity/bonus_capped": diversity_stats["bonus_capped"],
                "diversity/axis_movement_x": diversity_stats["axis_movement_x"],
                "diversity/axis_movement_y": diversity_stats["axis_movement_y"],
                "diversity/descriptor_x": diversity_stats["descriptor_x"],
                "diversity/descriptor_y": diversity_stats["descriptor_y"],
                "diversity/cell_x": diversity_stats["cell_x"],
                "diversity/cell_y": diversity_stats["cell_y"],
                "diversity/map_updated": diversity_stats["map_updated"],
                "diversity/map_occupied_cells": diversity_stats["map_occupied_cells"],
                "diversity/map_coverage": diversity_stats["map_coverage"],
                "diversity/ref_pool_updated": diversity_stats["ref_pool_updated"],
                "diversity/ref_pool_best_hpwl": diversity_stats["ref_pool_best_hpwl"],
                "diversity/ref_pool_worst_hpwl": diversity_stats["ref_pool_worst_hpwl"],
                "diversity/archive_pool_size": diversity_stats["archive_pool_size"],
                "diversity/archive_pairwise_diversity": diversity_stats["pairwise_archive_diversity"],
                "diversity/scope_is_prefix": diversity_stats["scope_is_prefix"],
                "diversity/scope_macro_count": diversity_stats["scope_macro_count"],
                **topk_stats,
                **curriculum_log,
                **self._consume_wandb_metrics(),
            },
            step=int(global_episode),
        )

        return next_global_episode

    def train(self, agent: Agent, env: Environment):
        logging.info(f"==================== Training ====================")
        if self.enable_curriculum and self.enable_diversity_phase:
            self._reset_diversity_window_state(self.curriculum_manager.current_window)

        loop_index = 0
        global_episode = 0
        next_visualize_episode = self.visualize_every_episodes
        total_train_time = 0.0
        episodes_completed_in_loop = 0

        self._write_run_summary(
            status="running",
            global_episode=global_episode,
            loop_index=loop_index,
            episodes_completed_in_loop=episodes_completed_in_loop,
            next_visualize_episode=next_visualize_episode,
            total_train_time=total_train_time,
        )

        try:
            while loop_index < self.num_loops:
                if self.target_budget_episodes is not None and global_episode >= self.target_budget_episodes:
                    break

                current_episode = loop_index * self.num_episodes_in_loop
                if loop_index > 0 and current_episode >= next_visualize_episode:
                    self._queue_wandb_metrics(
                        self.evaluate_and_visualize(agent, env, next_visualize_episode)
                    )
                    while next_visualize_episode <= current_episode:
                        next_visualize_episode += self.visualize_every_episodes

                current_buffer_capacity = self.num_episodes_in_loop * env.num_macros_to_place
                replay_buffer = ReplayBuffer(current_buffer_capacity, env.grid)

                loop_start_time = time.time()
                episodes_completed_in_loop = 0
                for _ in range(self.num_episodes_in_loop):
                    if self.target_budget_episodes is not None and global_episode >= self.target_budget_episodes:
                        break

                    global_episode = self._execute_training_episode(
                        agent,
                        env,
                        replay_buffer,
                        global_episode,
                        loop_index,
                    )
                    episodes_completed_in_loop += 1
                    self._write_run_summary(
                        status="running",
                        global_episode=global_episode,
                        loop_index=loop_index,
                        episodes_completed_in_loop=episodes_completed_in_loop,
                        next_visualize_episode=next_visualize_episode,
                        total_train_time=total_train_time,
                    )

                if episodes_completed_in_loop == 0:
                    break

                agent.update(replay_buffer, self.num_update_epochs, self.update_batch_size)

                total_train_time += time.time() - loop_start_time

                loop_index += 1
                episodes_completed_in_loop = 0
                self._write_run_summary(
                    status="running",
                    global_episode=global_episode,
                    loop_index=loop_index,
                    episodes_completed_in_loop=episodes_completed_in_loop,
                    next_visualize_episode=next_visualize_episode,
                    total_train_time=total_train_time,
                )

            if (
                self.target_budget_episodes is not None
                and global_episode < self.target_budget_episodes
            ):
                raise RuntimeError(
                    "Training exited before reaching target budget "
                    f"({global_episode}/{self.target_budget_episodes} episodes)."
                )

            self._write_run_summary(
                status="completed",
                global_episode=global_episode,
                loop_index=loop_index,
                episodes_completed_in_loop=episodes_completed_in_loop,
                next_visualize_episode=next_visualize_episode,
                total_train_time=total_train_time,
            )
        except Exception as exc:
            self._write_run_summary(
                status="failed",
                global_episode=global_episode,
                loop_index=loop_index,
                episodes_completed_in_loop=episodes_completed_in_loop,
                next_visualize_episode=next_visualize_episode,
                total_train_time=total_train_time,
                error=str(exc),
            )
            raise

    def run_episode(
        self,
        env: Environment,
        agent: Agent,
        replay_buffer: ReplayBuffer,
        global_episode: int,
    ):
        # initialize the environment
        hpwl = 0
        self._ensure_diversity_state_ready(env)
        env.set_fixed_macros(self._select_curriculum_episode_prefix(env))
        raw_s = env.reset()
        s = self._apply_hard_constraint(env, raw_s)
        done = False

        ppo_t = 0  # Time step for PPO (relative to macros it's placing)
        terminal_transition = None
        terminal_reward = 0.0
        dead_end_triggered = False
        episode_buffer_start = replay_buffer.count

        while True:
            if self._state_has_no_valid_actions(s):
                dead_end_triggered = True
                terminal_reward = float(self.dead_end_reward_penalty)
                macro_name = env.macro_name_list[env.t] if env.t < env.num_macros else "<done>"
                logging.warning(
                    "Episode %d dead-ended before sampling at PPO step %d (macro %s, env index %d). "
                    "Terminating episode with reward penalty %.4f.",
                    global_episode,
                    ppo_t,
                    macro_name,
                    env.t,
                    terminal_reward,
                )
                if replay_buffer.count > episode_buffer_start:
                    replay_buffer.r[replay_buffer.count - 1] += terminal_reward
                    replay_buffer.done[replay_buffer.count - 1] = 1.0
                break

            a, a_logp = agent.select_action(s, ppo_t)

            raw_s_, r, done, info = env.step(a)
            s_ = self._apply_hard_constraint(env, raw_s_)
            hpwl += info["delta_hpwl"]

            if not done:
                replay_buffer.store(s, ppo_t, a, a_logp, r, 0.0)

                ppo_t += 1
            else:
                terminal_transition = (s, ppo_t, a, a_logp)
                terminal_reward = float(r)

            if done and env.num_macros > env.num_macros_to_place:
                # if the episode is done, place the remaining macros greedily
                s = raw_s_
                for _ in range(env.num_macros - env.num_macros_to_place):
                    a = agent.act_greedy(s)
                    s_, r_, done, info = env.step(a)
                    hpwl += info["delta_hpwl"]
                    terminal_reward += float(r_)
                    s = s_
                break
            if done:
                break

            s = s_

        # Calculate HPWL breakdown by groups (proper calculation)
        hpwl_breakdown = env.calculate_hpwl_by_groups()

        # Use the total from breakdown (not incremental calculation)
        # This ensures: total = fixed + ppo + greedy
        total_hpwl = hpwl_breakdown['total']
        final_placement = env.macro_pos.copy()
        if dead_end_triggered:
            hpwl_breakdown['ppo_window'] += self.dead_end_hpwl_penalty
            hpwl_breakdown['total'] += self.dead_end_hpwl_penalty
            total_hpwl = hpwl_breakdown['total']
            diversity_bonus = 0.0
            diversity_stats = self._make_default_diversity_stats(env)
            diversity_stats["dead_end"] = 1.0
        else:
            diversity_bonus, diversity_stats = self._compute_diversity_bonus(
                env=env,
                final_hpwl=total_hpwl,
                final_placement=final_placement,
                terminal_reward=terminal_reward,
            )
            diversity_stats = self._update_posthoc_qd_map(
                env=env,
                final_hpwl=total_hpwl,
                final_placement=final_placement,
                global_episode=global_episode,
                diversity_stats=diversity_stats,
            )
            diversity_stats = self._maybe_update_reference_pool(
                hpwl=total_hpwl,
                placement=final_placement,
                diversity_stats=diversity_stats,
            )

        if terminal_transition is not None:
            s_done, t_done, a_done, a_logp_done = terminal_transition
            replay_buffer.store(
                s_done,
                t_done,
                a_done,
                a_logp_done,
                terminal_reward + diversity_bonus,
                1.0,
            )

        if (
            self.enable_curriculum
            and self.enable_diversity_phase
            and diversity_stats["phase"] == "diversity"
            and diversity_stats["dead_end"] < 0.5
        ):
            candidate_pool = list(self.diversity_episode_pool)
            if diversity_stats["quality_eligible"] > 0.5:
                candidate_pool.append((total_hpwl, final_placement.copy()))
            diversity_stats["archive_pool_size"] = len(candidate_pool)
            macro_names_scope, _ = self._get_diversity_scope_macro_names(env)
            diversity_stats["pairwise_archive_diversity"] = self._compute_pairwise_mean_distance(
                candidate_pool,
                macro_names_scope,
            )

        return total_hpwl, final_placement, hpwl_breakdown, diversity_stats


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(config: DictConfig):
    # set random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # wandb init
    curriculum_suffix = "-curriculum" if config.trainer.get('enable_curriculum', False) else ""
    wandb_group = os.environ.get("WANDB_GROUP") or _default_wandb_group(config)
    wandb_name_suffix = os.environ.get("WANDB_NAME_SUFFIX", "")
    wandb_mode = os.environ.get("WANDB_MODE", "online")
    wandb_run_id = os.environ.get("WANDB_RUN_ID")
    wandb_resume = os.environ.get("WANDB_RESUME")
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "Macro-Placement-Ablation"),
        name=f"PPO-{config.benchmark}-seed{config.seed}-horizon{config.num_macros_to_place}{curriculum_suffix}{wandb_name_suffix}",
        group=wandb_group,
        config=dict(config),
        mode=wandb_mode,
        id=wandb_run_id,
        resume=wandb_resume,
    )

    # set cpu num
    cpu_num = 10
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    # set device
    if(torch.cuda.is_available()):
        device = torch.device(f'cuda:{config.cuda}')
        torch.cuda.empty_cache()
        logging.info(f'Using GPU: {torch.cuda.get_device_name()}, cuda: {config.cuda}.')
    else:
        device = torch.device('cpu')
        logging.info('Using CPU.')

    # set up tensorboard writer
    tb_writer = SummaryWriter(config.tb_dir)

    # instantiate agent, environment and trainer
    agent: Agent = hydra.utils.instantiate(config.agent)
    agent.set_up(device, tb_writer)
    env: Environment = hydra.utils.instantiate(config.env)
    trainer: Trainer = hydra.utils.instantiate(config.trainer)
    trainer.setup_tb_writer(tb_writer)

    # train the model
    trainer.train(agent, env)

if __name__ == '__main__':
    main()
