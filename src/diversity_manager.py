import copy
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


Placement = Dict[str, Tuple[int, int, int, int]]


@dataclass
class EliteEntry:
    hpwl: float
    placement: Placement
    descriptor: Tuple[float, float]
    novelty: float
    reference_index: Optional[int] = None
    reference_hpwl: Optional[float] = None
    episode: Optional[int] = None
    local_episode_idx: Optional[int] = None


class WindowEliteMap:
    def __init__(self, bins_x: int = 8, bins_y: int = 8):
        self.bins_x = max(2, int(bins_x))
        self.bins_y = max(2, int(bins_y))
        self.entries: Dict[Tuple[int, int], EliteEntry] = {}
        self.visit_counts: Dict[Tuple[int, int], int] = {}
        self.bin_edges_x = self._make_uniform_edges(self.bins_x)
        self.bin_edges_y = self._make_uniform_edges(self.bins_y)

    def _make_uniform_edges(self, num_bins: int) -> np.ndarray:
        return np.linspace(0.0, 1.0, int(num_bins) + 1, dtype=np.float64)

    def reset(self) -> None:
        self.entries = {}
        self.visit_counts = {}
        self.bin_edges_x = self._make_uniform_edges(self.bins_x)
        self.bin_edges_y = self._make_uniform_edges(self.bins_y)

    @property
    def occupied_cells(self) -> int:
        return len(self.entries)

    @property
    def num_cells(self) -> int:
        return self.bins_x * self.bins_y

    def coverage_ratio(self) -> float:
        return float(self.occupied_cells / max(1, self.num_cells))

    @property
    def uses_adaptive_bins(self) -> bool:
        return not (
            np.allclose(self.bin_edges_x, self._make_uniform_edges(self.bins_x))
            and np.allclose(self.bin_edges_y, self._make_uniform_edges(self.bins_y))
        )

    def set_bin_edges(self, bin_edges_x, bin_edges_y) -> None:
        bin_edges_x = np.asarray(bin_edges_x, dtype=np.float64)
        bin_edges_y = np.asarray(bin_edges_y, dtype=np.float64)
        if bin_edges_x.shape != (self.bins_x + 1,) or bin_edges_y.shape != (self.bins_y + 1,):
            raise ValueError(
                "Adaptive bin edges must match the configured QD grid size "
                f"(expected {(self.bins_x + 1, self.bins_y + 1)}, "
                f"got {(bin_edges_x.shape, bin_edges_y.shape)})."
            )
        if (
            np.any(np.diff(bin_edges_x) <= 0.0)
            or np.any(np.diff(bin_edges_y) <= 0.0)
        ):
            raise ValueError("Adaptive bin edges must be strictly increasing on both axes.")
        if abs(float(bin_edges_x[0])) > 1e-12 or abs(float(bin_edges_y[0])) > 1e-12:
            raise ValueError("Adaptive bin edges must start at 0.0.")
        if abs(float(bin_edges_x[-1]) - 1.0) > 1e-12 or abs(float(bin_edges_y[-1]) - 1.0) > 1e-12:
            raise ValueError("Adaptive bin edges must end at 1.0.")
        self.bin_edges_x = bin_edges_x.copy()
        self.bin_edges_y = bin_edges_y.copy()

    def descriptor_to_cell(self, descriptor: Tuple[float, float]) -> Tuple[int, int]:
        x, y = descriptor
        x = float(np.clip(x, 0.0, np.nextafter(1.0, 0.0)))
        y = float(np.clip(y, 0.0, np.nextafter(1.0, 0.0)))
        return (
            min(
                self.bins_x - 1,
                max(0, int(np.searchsorted(self.bin_edges_x, x, side="right") - 1)),
            ),
            min(
                self.bins_y - 1,
                max(0, int(np.searchsorted(self.bin_edges_y, y, side="right") - 1)),
            ),
        )

    def probe(self, descriptor: Tuple[float, float]) -> dict:
        cell = self.descriptor_to_cell(descriptor)
        return {
            "cell": cell,
            "entry": self.entries.get(cell),
            "visit_count": int(self.visit_counts.get(cell, 0)),
        }

    def add_or_update(
        self,
        hpwl: float,
        placement: Placement,
        descriptor: Tuple[float, float],
        novelty: float,
        reference_index: Optional[int] = None,
        reference_hpwl: Optional[float] = None,
        episode: Optional[int] = None,
        local_episode_idx: Optional[int] = None,
    ) -> dict:
        cell = self.descriptor_to_cell(descriptor)
        incumbent = self.entries.get(cell)
        self.visit_counts[cell] = self.visit_counts.get(cell, 0) + 1
        updated = 0.0

        if incumbent is None or hpwl < incumbent.hpwl - 1e-12:
            self.entries[cell] = EliteEntry(
                hpwl=float(hpwl),
                placement=copy.deepcopy(placement),
                descriptor=(float(descriptor[0]), float(descriptor[1])),
                novelty=float(novelty),
                reference_index=None if reference_index is None else int(reference_index),
                reference_hpwl=None if reference_hpwl is None else float(reference_hpwl),
                episode=None if episode is None else int(episode),
                local_episode_idx=None if local_episode_idx is None else int(local_episode_idx),
            )
            updated = 1.0

        return {
            "cell": cell,
            "updated": updated,
            "occupied_cells": self.occupied_cells,
            "coverage": self.coverage_ratio(),
            "visit_count": int(self.visit_counts.get(cell, 0)),
        }

    def iter_entries(self) -> Iterable[Tuple[Tuple[int, int], EliteEntry]]:
        return self.entries.items()

    def export_payload(self, top_k: Optional[int] = None) -> dict:
        sorted_entries = sorted(
            self.entries.items(),
            key=lambda item: (item[1].hpwl, item[0][0], item[0][1]),
        )
        entries_payload = [
            {
                "cell_x": cell[0],
                "cell_y": cell[1],
                "hpwl": entry.hpwl,
                "novelty": entry.novelty,
                "visit_count": int(self.visit_counts.get(cell, 0)),
                "descriptor_x": entry.descriptor[0],
                "descriptor_y": entry.descriptor[1],
                "reference_index": entry.reference_index,
                "reference_hpwl": entry.reference_hpwl,
                "episode": entry.episode,
                "local_episode_idx": entry.local_episode_idx,
                "placement": entry.placement,
            }
            for cell, entry in sorted_entries
        ]
        payload = {
            "bins_x": self.bins_x,
            "bins_y": self.bins_y,
            "bin_edges_x": self.bin_edges_x.tolist(),
            "bin_edges_y": self.bin_edges_y.tolist(),
            "binning_mode": "adaptive" if self.uses_adaptive_bins else "uniform",
            "occupied_cells": self.occupied_cells,
            "num_cells": self.num_cells,
            "coverage_ratio": self.coverage_ratio(),
            "entries": entries_payload,
        }
        if top_k is not None and top_k > 0:
            payload["best_entries"] = entries_payload[: min(int(top_k), len(entries_payload))]
        return payload
