from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import (
    CandidatePath,
    CoordinatorResult,
    GridWorkspace,
    IterationRecord,
    PricingCoordinator,
    RobotAgent,
    TargetPoint,
)


@dataclass
class ExperimentConfig:
    seed: int = 7
    num_robots: int = 4
    num_targets: int = 6
    grid_size: int = 10
    robot_base_radius: float = 0.66
    robot_home_radius: float = 0.20
    max_iters: int = 12
    min_pricing_iters: int = 6
    alpha: float = 1.0
    decay: float = 0.12
    price_weight: float = 1.6
    max_price: float = 25.0
    switch_threshold: float = 1.5
    output_dir: str = "outputs"
    time_horizon: int = 24
    time_bin_size: float = 0.8
    nominal_speed: float = 0.32
    array_backend: str = "numpy"


@dataclass
class ExperimentOutcome:
    workspace_pricing: GridWorkspace
    workspace_baseline: GridWorkspace
    targets: List[TargetPoint]
    agents_pricing: List[RobotAgent]
    baseline_record: IterationRecord
    pricing_result: CoordinatorResult
    selected_paths_pricing: Dict[int, CandidatePath]
    robot_bases: List[np.ndarray]

class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _create_workspace(self) -> GridWorkspace:
        grid_size = max(6, int(self.config.grid_size))
        bottleneck_width = max(2, int(round(grid_size * 0.4)))
        z0 = max(1, int(round(grid_size * 0.3)))
        z1 = min(grid_size, max(z0 + 1, int(round(grid_size * 0.7))))
        ws = GridWorkspace(
            dims=(grid_size, grid_size, grid_size),
            world_min=np.array([-0.72, -0.72, 0.38], dtype=float),
            world_max=np.array([0.72, 0.72, 1.28], dtype=float),
            default_capacity=2,
            time_horizon=self.config.time_horizon,
            time_bin_size=self.config.time_bin_size,
            array_backend=self.config.array_backend,
        )
        ws.configure_center_bottleneck(width_xy=bottleneck_width, capacity_value=1, z_slice=(z0, z1))
        return ws

    def _robot_layout(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        base_radius = float(np.clip(self.config.robot_base_radius, 0.52, 0.90))
        ee_radius = float(np.clip(self.config.robot_home_radius, 0.14, base_radius - 0.18))
        robot_bases = [
            np.array([0.0, base_radius, 0.0], dtype=float),
            np.array([base_radius, 0.0, 0.0], dtype=float),
            np.array([0.0, -base_radius, 0.0], dtype=float),
            np.array([-base_radius, 0.0, 0.0], dtype=float),
        ]
        start_ee = [
            np.array([0.0, ee_radius, 0.84], dtype=float),
            np.array([ee_radius, 0.0, 0.84], dtype=float),
            np.array([0.0, -ee_radius, 0.84], dtype=float),
            np.array([-ee_radius, 0.0, 0.84], dtype=float),
        ]
        return robot_bases[: self.config.num_robots], start_ee[: self.config.num_robots]

    def _robot_sector_frames(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        _, start_ee = self._robot_layout()
        frames: List[Tuple[np.ndarray, np.ndarray]] = []
        for start in start_ee[: self.config.num_robots]:
            outward = np.asarray(start[:2], dtype=float)
            outward /= max(np.linalg.norm(outward), 1e-6)
            tangent = np.array([-outward[1], outward[0]], dtype=float)
            frames.append((outward, tangent))
        return frames

    def _generate_targets(self, workspace: GridWorkspace) -> List[TargetPoint]:
        n_targets = int(np.clip(self.config.num_targets, 4, 8))
        targets: List[TargetPoint] = []
        frames = self._robot_sector_frames()
        target_specs: List[Tuple[np.ndarray, Tuple[float, float], Tuple[float, float], float]] = [
            (np.array([0.10, 0.02], dtype=float), (58.0, 62.0), (0.74, 0.86), 0.03),
            (np.array([-0.02, 0.12], dtype=float), (49.0, 54.0), (0.74, 0.96), 0.035),
        ]

        flank_offsets = [0.18, -0.18, 0.18, -0.18]
        for idx, (outward, tangent) in enumerate(frames[:4]):
            anchor_xy = 0.48 * outward + flank_offsets[idx] * tangent
            target_specs.append((anchor_xy, (43.0, 49.0), (0.72, 1.08), 0.05))

        extra_offsets = [-0.20, 0.20]
        extra_frames = [frames[0], frames[1]]
        for offset, (outward, tangent) in zip(extra_offsets, extra_frames):
            anchor_xy = 0.56 * outward + offset * tangent
            target_specs.append((anchor_xy, (38.0, 45.0), (0.70, 1.04), 0.05))

        for t_id in range(n_targets):
            anchor_xy, utility_range, z_range, jitter_scale = target_specs[t_id]
            jitter = self.rng.uniform(-jitter_scale, jitter_scale, size=2)
            xy = anchor_xy + jitter
            utility = float(self.rng.uniform(*utility_range))
            z = float(self.rng.uniform(*z_range))
            pos = np.array([xy[0], xy[1], z], dtype=float)
            pos = np.minimum(np.maximum(pos, workspace.world_min + 1e-3), workspace.world_max - 1e-3)
            targets.append(TargetPoint(target_id=t_id, position=pos, utility=utility))
        return targets

    def _polyline_cost(self, waypoints: Sequence[np.ndarray], style: str) -> float:
        length = 0.0
        for i in range(len(waypoints) - 1):
            length += float(np.linalg.norm(waypoints[i + 1] - waypoints[i]))

        style_factor = {
            "direct": 1.00,
            "via_center": 1.04,
            "up_then_translate": 0.87,
            "edge_detour": 0.80,
        }[style]
        turn_penalty = 0.10 * max(0, len(waypoints) - 2)
        return length * 14.0 * style_factor + turn_penalty

    def _path_duration(self, waypoints: Sequence[np.ndarray]) -> float:
        length = 0.0
        for i in range(len(waypoints) - 1):
            length += float(np.linalg.norm(waypoints[i + 1] - waypoints[i]))
        return float(length / max(self.config.nominal_speed, 1e-6))

    def _resource_cells(self, cells: Sequence[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        # The market prices shared transit resources, not the final target occupancy itself.
        # Excluding the terminal cell lets alternative routes to the same goal compete on the
        # corridor they consume instead of collapsing onto a single priced goal cell.
        if len(cells) <= 1:
            return list(cells)
        return list(cells[:-1])

    def _resource_slots(
        self,
        cells: Sequence[Tuple[int, int, int]],
        slots: Sequence[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        if not slots:
            return []
        if not cells:
            return list(slots)
        terminal = cells[-1]
        return [slot for slot in slots if slot[:3] != terminal]

    def _candidate_waypoints(self, start: np.ndarray, target: np.ndarray) -> Dict[str, List[np.ndarray]]:
        top_z = 1.16
        center_hub = np.array([0.0, 0.0, max(top_z - 0.07, start[2], target[2])], dtype=float)

        start_xy = start[:2]
        start_norm = np.linalg.norm(start_xy)
        start_dir = start_xy / start_norm if start_norm > 1e-6 else np.array([1.0, 0.0])
        target_xy = target[:2]
        target_norm = np.linalg.norm(target_xy)
        target_dir = target_xy / target_norm if target_norm > 1e-6 else start_dir

        ring_radius = float(np.clip(max(start_norm, target_norm) + 0.12, 0.42, 0.60))
        start_ring = np.array([ring_radius * start_dir[0], ring_radius * start_dir[1], top_z - 0.02], dtype=float)
        target_ring = np.array([ring_radius * target_dir[0], ring_radius * target_dir[1], top_z - 0.02], dtype=float)
        tangent = np.array([-start_dir[1], start_dir[0]], dtype=float)
        side_ring = np.array([ring_radius * tangent[0], ring_radius * tangent[1], top_z - 0.02], dtype=float)

        return {
            "direct": [start, target],
            "via_center": [start, center_hub, target],
            "up_then_translate": [
                start,
                np.array([start[0], start[1], top_z], dtype=float),
                np.array([target[0], target[1], top_z], dtype=float),
                target,
            ],
            "edge_detour": [
                start,
                np.array([start[0], start[1], top_z - 0.02], dtype=float),
                start_ring,
                side_ring,
                target_ring,
                np.array([target[0], target[1], top_z - 0.02], dtype=float),
                target,
            ],
        }

    def _build_agents(self, workspace: GridWorkspace, targets: Sequence[TargetPoint]) -> List[RobotAgent]:
        robot_bases, start_ee_positions = self._robot_layout()
        return self.build_agents_for_starts(workspace, targets, start_ee_positions, robot_bases=robot_bases)

    def build_agents_for_starts(
        self,
        workspace: GridWorkspace,
        targets: Sequence[TargetPoint],
        start_ee_positions: Sequence[np.ndarray],
        robot_bases: Optional[Sequence[np.ndarray]] = None,
        robot_ids: Optional[Sequence[int]] = None,
    ) -> List[RobotAgent]:
        if robot_bases is None:
            robot_bases = self._robot_layout()[0]
        agents: List[RobotAgent] = []

        num_agents = min(self.config.num_robots, len(start_ee_positions), len(robot_bases))
        if robot_ids is None:
            robot_ids = list(range(num_agents))
        else:
            num_agents = min(num_agents, len(robot_ids))
        for robot_id in range(num_agents):
            agent = RobotAgent(
                robot_id=int(robot_ids[robot_id]),
                base_position=robot_bases[robot_id],
                start_ee_position=start_ee_positions[robot_id],
            )

            for target in targets:
                by_style = self._candidate_waypoints(start_ee_positions[robot_id], target.position)
                for style, waypoints in by_style.items():
                    path_wps = waypoints
                    cells = workspace.path_to_cells(path_wps)
                    slots = workspace.path_to_spacetime_slots(path_wps, nominal_speed=self.config.nominal_speed)
                    path_cost = self._polyline_cost(path_wps, style)
                    path = CandidatePath(
                        path_id=f"r{robot_id}_t{target.target_id}_{style}",
                        robot_id=robot_id,
                        target_id=target.target_id,
                        style=style,
                        waypoints=[np.array(w, dtype=float) for w in path_wps],
                        cells=cells,
                        resource_cells=self._resource_cells(cells),
                        resource_slots=self._resource_slots(cells, slots),
                        path_cost=path_cost,
                        target_utility=float(target.utility),
                        nominal_duration=self._path_duration(path_wps),
                    )
                    agent.candidate_paths.append(path)

            agents.append(agent)

        return agents

    def plan_online_assignments(
        self,
        workspace: GridWorkspace,
        targets: Sequence[TargetPoint],
        start_ee_positions: Sequence[np.ndarray],
        robot_bases: Optional[Sequence[np.ndarray]] = None,
        robot_ids: Optional[Sequence[int]] = None,
    ) -> Dict[int, CandidatePath]:
        agents = self.build_agents_for_starts(
            workspace,
            targets,
            start_ee_positions,
            robot_bases=robot_bases,
            robot_ids=robot_ids,
        )

        scored_candidates: List[Tuple[float, float, int, CandidatePath]] = []
        for agent in agents:
            best_by_target: Dict[int, CandidatePath] = {}
            best_score_by_target: Dict[int, float] = {}
            for path in agent.candidate_paths:
                score = path.score(workspace, price_weight=self.config.price_weight, use_pricing=True)
                if path.target_id not in best_by_target:
                    best_by_target[path.target_id] = path
                    best_score_by_target[path.target_id] = score
                    continue
                prev = best_by_target[path.target_id]
                prev_score = best_score_by_target[path.target_id]
                if score > prev_score + 1e-9 or (
                    abs(score - prev_score) <= 1e-9 and path.path_cost < prev.path_cost
                ):
                    best_by_target[path.target_id] = path
                    best_score_by_target[path.target_id] = score

            for target_id, path in best_by_target.items():
                scored_candidates.append(
                    (
                        best_score_by_target[target_id],
                        -path.path_cost,
                        agent.robot_id,
                        path,
                    )
                )

        scored_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

        assigned_paths: Dict[int, CandidatePath] = {}
        assigned_targets = set()
        for score, _, robot_id, path in scored_candidates:
            if score <= 0.0:
                continue
            if robot_id in assigned_paths or path.target_id in assigned_targets:
                continue
            assigned_paths[robot_id] = path
            assigned_targets.add(path.target_id)

        demand = workspace.zeros_like_prices(dtype=float)
        for path in assigned_paths.values():
            for slot in path.resource_slots:
                demand[slot] += 1.0

        delta = demand - workspace.capacity
        updated = workspace.prices * (1.0 - self.config.decay) + self.config.alpha * delta
        workspace.prices = workspace.xp.clip(updated, 0.0, self.config.max_price)
        return assigned_paths

    def _format_slot(self, slot: Tuple[int, ...], workspace: GridWorkspace) -> str:
        if len(slot) == 4:
            t0, t1 = workspace.slot_time_range((slot[0], slot[1], slot[2], slot[3]))
            return f"({slot[0]}, {slot[1]}, {slot[2]})@[{t0:.1f},{t1:.1f})"
        return str(slot)

    def _print_targets(self, targets: Sequence[TargetPoint]) -> None:
        print("\n=== Targets ===")
        for t in targets:
            x, y, z = t.position
            print(f"T{t.target_id}: pos=({x:+.3f}, {y:+.3f}, {z:+.3f}), utility={t.utility:.2f}")

    def _print_iteration(
        self,
        record: IterationRecord,
        workspace: GridWorkspace,
        title: str,
        max_cells: int = 5,
    ) -> None:
        print(f"\n[{title}] Iter {record.iteration}")
        for sel in sorted(record.selections, key=lambda s: s.robot_id):
            p = sel.path
            print(
                f"  R{sel.robot_id}: target=T{p.target_id}, path={p.style:<17} "
                f"score={sel.score:7.2f}  (u={p.target_utility:6.2f}, c={p.path_cost:6.2f}, "
                f"dur={p.nominal_duration:5.2f}s, price_pen={sel.price_penalty:6.2f})"
            )

        if record.conflict_cells:
            top_conflict = record.conflict_cells[:max_cells]
            print("  Conflict cells (top):", ", ".join(
                f"{self._format_slot(c, workspace)} demand={d}/cap={cap:.0f}@price={price:.2f}"
                for c, d, cap, price in top_conflict
            ))
        else:
            print("  Conflict cells (top): none")

        if record.top_price_changes:
            print(
                "  Price delta (top):",
                ", ".join(
                    f"{self._format_slot(cell, workspace)}:{delta:+.2f}"
                    for cell, delta in record.top_price_changes[:max_cells]
                ),
            )

        print(
            "  Totals: "
            f"utility={record.total_utility:.2f}, path_cost={record.total_path_cost:.2f}, "
            f"net_value={record.total_utility - record.total_path_cost:.2f}, "
            f"price_penalty={record.total_price_penalty:.2f}, score={record.total_score:.2f}, "
            f"conflict_degree={record.conflict_degree:.2f}, utilization={record.utilization_ratio:.3f}"
        )

    def _bottleneck_usage(self, record: IterationRecord, workspace: GridWorkspace) -> float:
        mask = workspace.bottleneck_mask
        if not np.any(mask):
            return 0.0
        demand = np.asarray(record.demand[mask, :], dtype=float)
        cap = workspace.to_numpy(workspace.capacity)[mask, :]
        return float(np.sum(np.minimum(demand, cap)) / np.sum(cap))

    def _print_comparison(
        self,
        baseline: IterationRecord,
        pricing: IterationRecord,
        ws_base: GridWorkspace,
        ws_price: GridWorkspace,
    ) -> None:
        b_usage = self._bottleneck_usage(baseline, ws_base)
        p_usage = self._bottleneck_usage(pricing, ws_price)

        print("\n=== Baseline vs Pricing ===")
        print(f"{'Metric':<24} {'NoPricing':>12} {'Pricing':>12}")
        print(f"{'Conflict degree':<24} {baseline.conflict_degree:12.2f} {pricing.conflict_degree:12.2f}")
        print(f"{'Total utility':<24} {baseline.total_utility:12.2f} {pricing.total_utility:12.2f}")
        print(f"{'Total path cost':<24} {baseline.total_path_cost:12.2f} {pricing.total_path_cost:12.2f}")
        print(
            f"{'Utility - path cost':<24} "
            f"{baseline.total_utility - baseline.total_path_cost:12.2f} "
            f"{pricing.total_utility - pricing.total_path_cost:12.2f}"
        )
        print(f"{'Total score':<24} {baseline.total_score:12.2f} {pricing.total_score:12.2f}")
        print(f"{'Global utilization':<24} {baseline.utilization_ratio:12.3f} {pricing.utilization_ratio:12.3f}")
        print(f"{'Bottleneck usage':<24} {b_usage:12.3f} {p_usage:12.3f}")

    def _plot_price_history(
        self,
        pricing_result: CoordinatorResult,
        workspace: GridWorkspace,
        out_path: Path,
        top_k: int = 8,
    ) -> None:
        snapshots = np.stack(pricing_result.price_snapshots, axis=0)
        max_map = np.max(snapshots, axis=0)
        flat = max_map.ravel()
        k = min(top_k, flat.size)
        idxs = np.argpartition(flat, -k)[-k:]
        idxs = sorted(idxs, key=lambda i: flat[i], reverse=True)

        x = np.arange(snapshots.shape[0])
        fig, ax = plt.subplots(figsize=(9, 5))
        for idx in idxs:
            cell = np.unravel_index(idx, max_map.shape)
            if max_map[cell] <= 1e-8:
                continue
            series = snapshots[:, cell[0], cell[1], cell[2], cell[3]]
            t0, t1 = workspace.slot_time_range((int(cell[0]), int(cell[1]), int(cell[2]), int(cell[3])))
            label = f"cell({int(cell[0])},{int(cell[1])},{int(cell[2])})@[{t0:.1f},{t1:.1f})"
            ax.plot(x, series, linewidth=2.0, label=label)

        ax.set_title("Dynamic Space-Time Slot Prices Across Iterations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Price")
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)

    def _plot_3d_paths(
        self,
        agents: Sequence[RobotAgent],
        targets: Sequence[TargetPoint],
        baseline_record: IterationRecord,
        pricing_record: IterationRecord,
        workspace: GridWorkspace,
        out_path: Path,
    ) -> None:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for agent in agents:
            for path in agent.candidate_paths:
                pts = np.array(path.waypoints)
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="0.75", alpha=0.18, linewidth=0.8)

        bl_map = {sel.robot_id: sel.path for sel in baseline_record.selections}
        pr_map = {sel.robot_id: sel.path for sel in pricing_record.selections}

        baseline_label_used = False
        pricing_label_used = False
        for rid in sorted(pr_map.keys()):
            if rid in bl_map:
                pts = np.array(bl_map[rid].waypoints)
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    color="#b22222",
                    linestyle="--",
                    linewidth=2.0,
                    alpha=0.9,
                    label="Baseline selected" if not baseline_label_used else None,
                )
                baseline_label_used = True

            pts = np.array(pr_map[rid].waypoints)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color="#1f77b4",
                linewidth=2.8,
                alpha=0.95,
                label="Pricing selected" if not pricing_label_used else None,
            )
            pricing_label_used = True

        t_xyz = np.array([t.position for t in targets])
        t_u = np.array([t.utility for t in targets])
        scatter = ax.scatter(
            t_xyz[:, 0],
            t_xyz[:, 1],
            t_xyz[:, 2],
            c=t_u,
            s=70,
            cmap="viridis",
            edgecolors="black",
            label="Targets",
        )

        top_price_cells = workspace.top_priced_cells(k=20)
        if top_price_cells:
            centers = np.array([workspace.grid_to_world(c) for c, _ in top_price_cells])
            values = np.array([v for _, v in top_price_cells])
            ax.scatter(
                centers[:, 0],
                centers[:, 1],
                centers[:, 2],
                c="#ff7f0e",
                s=40 + 80 * (values / max(1e-6, values.max())),
                marker="s",
                alpha=0.6,
                label="High-price cells",
            )

        for t in targets:
            ax.text(t.position[0], t.position[1], t.position[2] + 0.02, f"T{t.target_id}", fontsize=8)

        ax.set_xlim(workspace.world_min[0], workspace.world_max[0])
        ax.set_ylim(workspace.world_min[1], workspace.world_max[1])
        ax.set_zlim(workspace.world_min[2], workspace.world_max[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Candidate Paths, Selected Paths, and High-price Cells")
        ax.legend(loc="upper left")
        fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.08, label="Target Utility")
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)

    def run(self, verbose: bool = True) -> ExperimentOutcome:
        ws_baseline = self._create_workspace()
        ws_pricing = ws_baseline.clone()
        ws_baseline.reset_prices()
        ws_pricing.reset_prices()

        targets = self._generate_targets(ws_baseline)
        agents_baseline = self._build_agents(ws_baseline, targets)
        agents_pricing = self._build_agents(ws_pricing, targets)
        robot_bases, _ = self._robot_layout()

        if verbose:
            print(
                f"\n=== Workspace ===\n"
                f"grid={ws_baseline.dims[0]}x{ws_baseline.dims[1]}x{ws_baseline.dims[2]}, "
                f"time_horizon={ws_baseline.time_horizon}, time_bin={ws_baseline.time_bin_size:.2f}s, "
                f"backend={self.config.array_backend}"
            )
            self._print_targets(targets)

        baseline = PricingCoordinator(
            workspace=ws_baseline,
            agents=agents_baseline,
            alpha=self.config.alpha,
            decay=self.config.decay,
            max_price=self.config.max_price,
            price_weight=self.config.price_weight,
            stable_rounds=1,
            min_iters=1,
        ).run(max_iters=1, use_pricing=False)

        if verbose:
            self._print_iteration(baseline.records[-1], ws_baseline, title="NoPricing")

        pricing = PricingCoordinator(
            workspace=ws_pricing,
            agents=agents_pricing,
            alpha=self.config.alpha,
            decay=self.config.decay,
            max_price=self.config.max_price,
            price_weight=self.config.price_weight,
            stable_rounds=2,
            min_iters=self.config.min_pricing_iters,
            switch_threshold=self.config.switch_threshold,
        ).run(max_iters=self.config.max_iters, use_pricing=True)

        if verbose:
            for rec in pricing.records:
                self._print_iteration(rec, ws_pricing, title="Pricing")
            self._print_comparison(
                baseline.records[-1],
                pricing.records[-1],
                ws_baseline,
                ws_pricing,
            )

        price_curve_path = self.output_dir / "price_history.png"
        paths_3d_path = self.output_dir / "paths_3d.png"
        self._plot_price_history(pricing, ws_pricing, price_curve_path)
        self._plot_3d_paths(
            agents_pricing,
            targets,
            baseline.records[-1],
            pricing.records[-1],
            ws_pricing,
            paths_3d_path,
        )

        if verbose:
            print("\n=== Saved Figures ===")
            print(f"- {price_curve_path}")
            print(f"- {paths_3d_path}")

        selected_paths = {sel.robot_id: sel.path for sel in pricing.records[-1].selections}

        return ExperimentOutcome(
            workspace_pricing=ws_pricing,
            workspace_baseline=ws_baseline,
            targets=targets,
            agents_pricing=agents_pricing,
            baseline_record=baseline.records[-1],
            pricing_result=pricing,
            selected_paths_pricing=selected_paths,
            robot_bases=robot_bases,
        )
