"""
Curriculum Manager for Progressive Macro Placement Training - EXPANDING WINDOW (NO FREEZING)

This module manages the expanding window curriculum where:
- Training is divided into windows (e.g., 2000 episodes each)
- Each window increases num_macros_to_place by a fixed increment (e.g., +5)
- NO MACROS ARE FROZEN - PPO can refine all previously learned placements
- Remaining macros (outside the window) are placed greedily
- PPO weights and optimizer state remain intact across windows
"""

import logging
import pickle
import os
from typing import Dict, List, Tuple, Optional
import numpy as np


class CurriculumManager:
    """Manages expanding window curriculum for macro placement (no freezing)."""

    def __init__(
        self,
        window_size: int = 2000,
        increment_size: int = 5,
        initial_num_macros: int = 5,
        selection_strategy: str = "best-1",
        top_k: int = 3,
        save_dir: Optional[str] = None
    ):
        """
        Initialize the curriculum manager.

        Args:
            window_size: Number of episodes per training window
            increment_size: Number of additional macros to place per window
            initial_num_macros: Initial number of macros to place with PPO
            selection_strategy: Strategy for tracking best placement (for logging only)
            top_k: Number of top placements to track (for logging only)
            save_dir: Directory to save curriculum state and best placements
        """
        self.window_size = window_size
        self.increment_size = increment_size
        self.initial_num_macros = initial_num_macros
        self.selection_strategy = selection_strategy
        self.top_k = top_k
        self.save_dir = save_dir

        # Current curriculum state
        self.current_window = 0
        self.current_num_macros = initial_num_macros
        self.episodes_in_current_window = 0
        self.global_episode = 0

        # Track placements in current window
        self.window_placements = []  # List of (hpwl, placement) tuples

        # NO FREEZING - these remain empty/zero
        self.fixed_macro_positions = {}  # Always empty in expanding window mode
        self.num_fixed_macros = 0  # Always 0 in expanding window mode

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        logging.info(f"Initialized CurriculumManager (EXPANDING WINDOW - NO FREEZING):")
        logging.info(f"  Window size: {window_size} episodes")
        logging.info(f"  Increment size: +{increment_size} macros per window")
        logging.info(f"  Initial macros: {initial_num_macros}")
        logging.info(f"  Mode: Expanding window (no freezing)")
        logging.info(f"  PPO will place macros 1-{initial_num_macros} in window 0")

    def record_episode(self, hpwl: float, placement: Dict[str, Tuple[int, int, int, int]]):
        """
        Record an episode's result in the current window.

        Args:
            hpwl: Final HPWL of the episode
            placement: Dictionary mapping macro names to (x, y, w, h) positions
        """
        self.window_placements.append((hpwl, placement.copy()))
        self.episodes_in_current_window += 1
        self.global_episode += 1

    def should_advance_window(self) -> bool:
        """Check if we should advance to the next training window."""
        return self.episodes_in_current_window >= self.window_size

    def get_best_placement_in_window(self) -> Tuple[float, Dict[str, Tuple[int, int, int, int]]]:
        """
        Get the best placement from the current window without advancing.

        Returns:
            Tuple of (best_hpwl, best_placement)
        """
        if not self.window_placements:
            return float('inf'), {}

        best_idx = np.argmin([hpwl for hpwl, _ in self.window_placements])
        best_hpwl, best_placement = self.window_placements[best_idx]
        return best_hpwl, best_placement

    def advance_window(self) -> Tuple[int, Dict[str, Tuple[int, int, int, int]]]:
        """
        Advance to the next training window.

        CRITICAL: In expanding window mode, we do NOT freeze any macros.
        We only expand num_macros_to_place.

        Returns:
            Tuple of (new_num_macros_to_place, fixed_macro_positions)
            Note: fixed_macro_positions will always be empty dict in this mode
        """
        if not self.window_placements:
            logging.warning("No placements recorded in current window. Cannot advance.")
            return self.current_num_macros, self.fixed_macro_positions

        # Get best placement for logging/visualization only
        best_hpwl, best_placement = self.get_best_placement_in_window()

        # Save window results
        if self.save_dir:
            self._save_window_results(best_placement, best_hpwl)

        # Advance to next window
        old_num_macros = self.current_num_macros
        self.current_window += 1
        self.current_num_macros += self.increment_size
        self.episodes_in_current_window = 0
        self.window_placements = []

        # NO FREEZING - fixed positions remain empty
        # num_fixed_macros remains 0

        logging.info(f"=" * 60)
        logging.info(f"Advanced to Window {self.current_window}")
        logging.info(f"  Previous window: PPO placed macros 1-{old_num_macros}")
        logging.info(f"  New window: PPO will place macros 1-{self.current_num_macros}")
        logging.info(f"  Best HPWL in previous window: {best_hpwl:.4f}")
        logging.info(f"  NO MACROS FROZEN - PPO can refine all {self.current_num_macros} macros")
        logging.info(f"  Remaining macros ({self.current_num_macros + 1}+) placed greedily")
        logging.info(f"=" * 60)

        # Return empty fixed positions (no freezing)
        return self.current_num_macros, {}

    def _save_window_results(self, best_placement: Dict[str, Tuple[int, int, int, int]], best_hpwl: float):
        """Save window results to disk."""
        window_dir = os.path.join(self.save_dir, f"window_{self.current_window}")
        os.makedirs(window_dir, exist_ok=True)

        # Save best placement
        with open(os.path.join(window_dir, "best_placement.pkl"), 'wb') as f:
            pickle.dump(best_placement, f)

        # Save all placements in window
        with open(os.path.join(window_dir, "all_placements.pkl"), 'wb') as f:
            pickle.dump(self.window_placements, f)

        # Save curriculum state
        state = {
            'window': self.current_window,
            'num_macros': self.current_num_macros,
            'num_fixed_macros': 0,  # Always 0 in expanding window mode
            'fixed_positions': {},  # Always empty in expanding window mode
            'global_episode': self.global_episode,
            'best_hpwl': best_hpwl,
            'mode': 'expanding_window_no_freeze'
        }
        with open(os.path.join(window_dir, "curriculum_state.pkl"), 'wb') as f:
            pickle.dump(state, f)

        logging.info(f"  Saved window results to {window_dir}")

    def get_current_state(self) -> Dict:
        """Get current curriculum state."""
        return {
            'window': self.current_window,
            'num_macros_to_place': self.current_num_macros,
            'num_fixed_macros': 0,  # Always 0 in expanding window mode
            'episodes_in_window': self.episodes_in_current_window,
            'global_episode': self.global_episode,
            'fixed_positions': {},  # Always empty in expanding window mode
            'mode': 'expanding_window_no_freeze'
        }

    def load_state(self, state_path: str):
        """Load curriculum state from disk."""
        with open(state_path, 'rb') as f:
            state = pickle.load(f)

        self.current_window = state['window']
        self.current_num_macros = state['num_macros']
        self.global_episode = state['global_episode']

        # In expanding window mode, these are always 0/empty
        self.num_fixed_macros = 0
        self.fixed_macro_positions = {}

        logging.info(f"Loaded curriculum state from {state_path}")
        logging.info(f"  Window: {self.current_window}")
        logging.info(f"  Num macros to place: {self.current_num_macros}")
        logging.info(f"  Mode: Expanding window (no freezing)")
