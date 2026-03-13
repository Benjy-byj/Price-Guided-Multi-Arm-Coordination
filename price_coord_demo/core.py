from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .array_backend import get_array_module, scalar_to_float, to_numpy

Cell = Tuple[int, int, int]
SpaceTimeSlot = Tuple[int, int, int, int]


@dataclass
class GridWorkspace:
    """Discrete 3D workspace with time-expanded prices over (x, y, z, t) slots."""

    dims: Tuple[int, int, int] = (10, 10, 10)
    world_min: np.ndarray = field(default_factory=lambda: np.array([-0.35, -0.35, 0.72], dtype=float))
    world_max: np.ndarray = field(default_factory=lambda: np.array([0.35, 0.35, 1.12], dtype=float))
    default_capacity: int = 2
    time_horizon: int = 24
    time_bin_size: float = 0.35
    array_backend: str = "numpy"

    def __post_init__(self) -> None:
        self.world_min = np.asarray(self.world_min, dtype=float)
        self.world_max = np.asarray(self.world_max, dtype=float)
        self.xp = get_array_module(self.array_backend)
        self._cell_size = (self.world_max - self.world_min) / np.array(self.dims, dtype=float)
        slot_shape = self.dims + (self.time_horizon,)
        self.prices = self.xp.zeros(slot_shape, dtype=float)
        self.capacity = self.xp.full(slot_shape, self.default_capacity, dtype=float)
        self.bottleneck_mask = np.zeros(self.dims, dtype=bool)

    @property
    def cell_size(self) -> np.ndarray:
        return self._cell_size

    def clone(self) -> "GridWorkspace":
        clone_ws = GridWorkspace(
            dims=self.dims,
            world_min=self.world_min.copy(),
            world_max=self.world_max.copy(),
            default_capacity=self.default_capacity,
            time_horizon=self.time_horizon,
            time_bin_size=self.time_bin_size,
            array_backend=self.array_backend,
        )
        clone_ws.prices = self.xp.array(self.prices, copy=True)
        clone_ws.capacity = self.xp.array(self.capacity, copy=True)
        clone_ws.bottleneck_mask = self.bottleneck_mask.copy()
        return clone_ws

    def reset_prices(self) -> None:
        self.prices.fill(0.0)

    def to_numpy(self, array) -> np.ndarray:
        return to_numpy(array)

    def zeros_like_prices(self, dtype=float):
        return self.xp.zeros_like(self.prices, dtype=dtype)

    def max_price_value(self) -> float:
        return scalar_to_float(self.xp.max(self.prices))

    def configure_center_bottleneck(
        self,
        width_xy: int = 2,
        capacity_value: int = 1,
        z_slice: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Create a center bottleneck band to induce shared-resource contention."""

        cx, cy = self.dims[0] // 2, self.dims[1] // 2
        half = max(1, width_xy // 2)
        x0, x1 = max(0, cx - half), min(self.dims[0], cx + half)
        y0, y1 = max(0, cy - half), min(self.dims[1], cy + half)
        if z_slice is None:
            z0, z1 = self.dims[2] // 4, self.dims[2] - self.dims[2] // 4
        else:
            z0 = max(0, int(z_slice[0]))
            z1 = min(self.dims[2], int(z_slice[1]))
        self.capacity[x0:x1, y0:y1, z0:z1, :] = float(capacity_value)
        self.bottleneck_mask[x0:x1, y0:y1, z0:z1] = True

    def world_to_grid(self, point: Sequence[float], clamp: bool = False) -> Optional[Cell]:
        p = np.asarray(point, dtype=float)
        if clamp:
            p = np.minimum(np.maximum(p, self.world_min), self.world_max - 1e-9)
        elif np.any(p < self.world_min) or np.any(p >= self.world_max):
            return None

        ratio = (p - self.world_min) / self._cell_size
        idx = np.floor(ratio).astype(int)
        idx = np.minimum(np.maximum(idx, 0), np.array(self.dims) - 1)
        return int(idx[0]), int(idx[1]), int(idx[2])

    def grid_to_world(self, cell: Cell) -> np.ndarray:
        c = np.asarray(cell, dtype=float)
        return self.world_min + (c + 0.5) * self._cell_size

    def _sample_segment(self, p0: np.ndarray, p1: np.ndarray, step: float) -> List[np.ndarray]:
        length = float(np.linalg.norm(p1 - p0))
        if length < 1e-9:
            return [p0.copy()]
        n = max(2, int(np.ceil(length / step)) + 1)
        alphas = np.linspace(0.0, 1.0, n)
        return [p0 * (1.0 - a) + p1 * a for a in alphas]

    def path_to_cells(self, waypoints: Sequence[np.ndarray], sample_step: Optional[float] = None) -> List[Cell]:
        if len(waypoints) < 2:
            return []

        step = sample_step if sample_step is not None else float(np.min(self._cell_size) * 0.5)
        ordered: List[Cell] = []
        seen = set()

        for i in range(len(waypoints) - 1):
            p0, p1 = np.asarray(waypoints[i], dtype=float), np.asarray(waypoints[i + 1], dtype=float)
            for pt in self._sample_segment(p0, p1, step):
                cell = self.world_to_grid(pt, clamp=True)
                if cell is None:
                    continue
                if cell not in seen:
                    ordered.append(cell)
                    seen.add(cell)
        return ordered

    def path_to_spacetime_slots(
        self,
        waypoints: Sequence[np.ndarray],
        nominal_speed: float,
        start_time_offset: float = 0.0,
        sample_step: Optional[float] = None,
    ) -> List[SpaceTimeSlot]:
        if len(waypoints) < 2:
            return []

        step = sample_step if sample_step is not None else float(np.min(self._cell_size) * 0.5)
        ordered: List[SpaceTimeSlot] = []
        seen = set()
        elapsed_length = 0.0
        speed = max(float(nominal_speed), 1e-6)

        for i in range(len(waypoints) - 1):
            p0 = np.asarray(waypoints[i], dtype=float)
            p1 = np.asarray(waypoints[i + 1], dtype=float)
            segment_length = float(np.linalg.norm(p1 - p0))
            samples = self._sample_segment(p0, p1, step)
            denom = max(1, len(samples) - 1)
            for j, pt in enumerate(samples):
                cell = self.world_to_grid(pt, clamp=True)
                if cell is None:
                    continue
                alpha = float(j / denom)
                travel_time = (elapsed_length + alpha * segment_length) / speed
                absolute_time = max(0.0, start_time_offset + travel_time)
                time_bin = min(self.time_horizon - 1, int(np.floor(absolute_time / self.time_bin_size)))
                slot = (cell[0], cell[1], cell[2], time_bin)
                if slot not in seen:
                    ordered.append(slot)
                    seen.add(slot)
            elapsed_length += segment_length
        return ordered

    def slot_time_range(self, slot: SpaceTimeSlot) -> Tuple[float, float]:
        start = float(slot[3]) * self.time_bin_size
        return start, start + self.time_bin_size

    def top_priced_slots(self, k: int = 8) -> List[Tuple[SpaceTimeSlot, float]]:
        prices_np = self.to_numpy(self.prices)
        flat = prices_np.ravel()
        if flat.size == 0:
            return []
        topk = min(k, flat.size)
        idxs = np.argpartition(flat, -topk)[-topk:]
        idxs = sorted(idxs, key=lambda i: flat[i], reverse=True)
        result: List[Tuple[Cell, float]] = []
        for i in idxs:
            price = float(flat[i])
            if price <= 1e-8:
                continue
            cell = np.unravel_index(i, prices_np.shape)
            result.append(((int(cell[0]), int(cell[1]), int(cell[2]), int(cell[3])), price))
        return result

    def top_priced_cells(self, k: int = 8) -> List[Tuple[Cell, float]]:
        prices_np = self.to_numpy(self.prices)
        flat = np.max(prices_np, axis=3).ravel()
        if flat.size == 0:
            return []
        topk = min(k, flat.size)
        idxs = np.argpartition(flat, -topk)[-topk:]
        idxs = sorted(idxs, key=lambda i: flat[i], reverse=True)
        result: List[Tuple[Cell, float]] = []
        for i in idxs:
            price = float(flat[i])
            if price <= 1e-8:
                continue
            cell = np.unravel_index(i, self.dims)
            result.append(((int(cell[0]), int(cell[1]), int(cell[2])), price))
        return result


@dataclass
class TargetPoint:
    target_id: int
    position: np.ndarray
    utility: float


@dataclass
class CandidatePath:
    path_id: str
    robot_id: int
    target_id: int
    style: str
    waypoints: List[np.ndarray]
    cells: List[Cell]
    resource_cells: List[Cell]
    resource_slots: List[SpaceTimeSlot]
    path_cost: float
    target_utility: float
    nominal_duration: float

    def congestion_penalty(self, workspace: GridWorkspace, price_weight: float = 1.0) -> float:
        return float(price_weight * sum(float(workspace.prices[s]) for s in self.resource_slots))

    def score(self, workspace: GridWorkspace, price_weight: float = 1.0, use_pricing: bool = True) -> float:
        penalty = self.congestion_penalty(workspace, price_weight) if use_pricing else 0.0
        return float(self.target_utility - self.path_cost - penalty)


@dataclass
class PathSelection:
    robot_id: int
    path: CandidatePath
    score: float
    price_penalty: float


@dataclass
class RobotAgent:
    robot_id: int
    base_position: np.ndarray
    start_ee_position: np.ndarray
    candidate_paths: List[CandidatePath] = field(default_factory=list)
    chosen_path: Optional[CandidatePath] = None

    def choose_best_path(
        self,
        workspace: GridWorkspace,
        price_weight: float = 1.0,
        use_pricing: bool = True,
        switch_threshold: float = 0.0,
    ) -> PathSelection:
        best: Optional[PathSelection] = None
        incumbent: Optional[PathSelection] = None
        for path in self.candidate_paths:
            penalty = path.congestion_penalty(workspace, price_weight) if use_pricing else 0.0
            score = path.target_utility - path.path_cost - penalty
            candidate = PathSelection(robot_id=self.robot_id, path=path, score=float(score), price_penalty=float(penalty))
            if self.chosen_path is not None and path.path_id == self.chosen_path.path_id:
                incumbent = candidate
            if best is None:
                best = candidate
            elif candidate.score > best.score + 1e-9:
                best = candidate
            elif abs(candidate.score - best.score) <= 1e-9 and candidate.path.path_cost < best.path.path_cost:
                best = candidate

        if best is None:
            raise ValueError(f"Robot {self.robot_id} has no candidate paths")
        if use_pricing and incumbent is not None and incumbent.score >= best.score - switch_threshold:
            best = incumbent
        self.chosen_path = best.path
        return best


@dataclass
class IterationRecord:
    iteration: int
    selections: List[PathSelection]
    demand: np.ndarray
    conflict_cells: List[Tuple[SpaceTimeSlot, int, float, float]]
    total_utility: float
    total_path_cost: float
    total_price_penalty: float
    total_score: float
    conflict_degree: float
    utilization_ratio: float
    top_price_cells: List[Tuple[SpaceTimeSlot, float]]
    top_price_changes: List[Tuple[SpaceTimeSlot, float]]


@dataclass
class CoordinatorResult:
    records: List[IterationRecord]
    price_snapshots: List[np.ndarray]


@dataclass
class PricingCoordinator:
    workspace: GridWorkspace
    agents: List[RobotAgent]
    alpha: float = 0.8
    decay: float = 0.15
    max_price: float = 30.0
    price_weight: float = 1.0
    stable_rounds: int = 2
    min_iters: int = 4
    switch_threshold: float = 0.0

    def _compute_demand(self, selections: List[PathSelection]) -> np.ndarray:
        demand = self.workspace.zeros_like_prices(dtype=float)
        for sel in selections:
            for slot in sel.path.resource_slots:
                demand[slot] += 1.0
        return demand

    def _update_price(self, demand: np.ndarray) -> np.ndarray:
        # Price decay prevents runaway escalation and helps previously expensive cells become
        # affordable again, mitigating under-utilization of shared corridors.
        xp = self.workspace.xp
        delta = demand - self.workspace.capacity
        updated = self.workspace.prices * (1.0 - self.decay) + self.alpha * delta
        updated = xp.clip(updated, 0.0, self.max_price)
        change = updated - self.workspace.prices
        self.workspace.prices = updated
        return change

    def run(self, max_iters: int = 12, use_pricing: bool = True) -> CoordinatorResult:
        records: List[IterationRecord] = []
        price_snapshots: List[np.ndarray] = [self.workspace.to_numpy(self.workspace.prices)]

        prev_choice_ids: Dict[int, str] = {}
        stable_count = 0

        for it in range(max_iters):
            selections = [
                agent.choose_best_path(
                    self.workspace,
                    price_weight=self.price_weight,
                    use_pricing=use_pricing,
                    switch_threshold=self.switch_threshold if use_pricing else 0.0,
                )
                for agent in self.agents
            ]
            demand = self._compute_demand(selections)
            overflow = self.workspace.xp.maximum(demand - self.workspace.capacity, 0.0)
            demand_np = self.workspace.to_numpy(demand)
            capacity_np = self.workspace.to_numpy(self.workspace.capacity)
            conflict_degree = float(np.sum(self.workspace.to_numpy(overflow)))
            utilization_ratio = float(np.count_nonzero(demand_np) / demand_np.size)

            conflict_idxs = np.argwhere(demand_np > capacity_np)
            conflict_cells: List[Tuple[SpaceTimeSlot, int, float, float]] = []
            for idx in conflict_idxs:
                cell = (int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3]))
                conflict_cells.append(
                    (
                        cell,
                        int(demand_np[cell]),
                        float(capacity_np[cell]),
                        float(self.workspace.to_numpy(self.workspace.prices[cell])),
                    )
                )
            conflict_cells.sort(key=lambda c: (c[1] - c[2], c[3]), reverse=True)

            total_utility = float(sum(sel.path.target_utility for sel in selections))
            total_path_cost = float(sum(sel.path.path_cost for sel in selections))
            total_penalty = float(sum(sel.price_penalty for sel in selections))
            total_score = float(sum(sel.score for sel in selections))

            price_change = self.workspace.zeros_like_prices(dtype=float)
            if use_pricing:
                price_change = self._update_price(demand)
            else:
                self.workspace.prices.fill(0.0)

            top_change_cells = []
            if use_pricing:
                price_change_np = self.workspace.to_numpy(price_change)
                flat = price_change_np.ravel()
                k = min(8, flat.size)
                idxs = np.argpartition(np.abs(flat), -k)[-k:]
                idxs = sorted(idxs, key=lambda i: abs(flat[i]), reverse=True)
                for i in idxs:
                    value = float(flat[i])
                    if abs(value) < 1e-9:
                        continue
                    cell_idx = np.unravel_index(i, price_change_np.shape)
                    top_change_cells.append(
                        ((int(cell_idx[0]), int(cell_idx[1]), int(cell_idx[2]), int(cell_idx[3])), value)
                    )

            record = IterationRecord(
                iteration=it,
                selections=selections,
                demand=demand_np,
                conflict_cells=conflict_cells,
                total_utility=total_utility,
                total_path_cost=total_path_cost,
                total_price_penalty=total_penalty,
                total_score=total_score,
                conflict_degree=conflict_degree,
                utilization_ratio=utilization_ratio,
                top_price_cells=self.workspace.top_priced_slots(k=8),
                top_price_changes=top_change_cells,
            )
            records.append(record)
            price_snapshots.append(self.workspace.to_numpy(self.workspace.prices))

            choices = {sel.robot_id: sel.path.path_id for sel in selections}
            if choices == prev_choice_ids:
                stable_count += 1
            else:
                stable_count = 0
            prev_choice_ids = choices

            if it + 1 >= self.min_iters and stable_count >= self.stable_rounds:
                break

        return CoordinatorResult(records=records, price_snapshots=price_snapshots)
