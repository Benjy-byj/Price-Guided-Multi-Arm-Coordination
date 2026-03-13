from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import CandidatePath, GridWorkspace, TargetPoint
from .experiment import ExperimentRunner
from .pybullet_exec import PyBulletExecutor


@dataclass
class ActiveStreamTarget:
    target: TargetPoint
    body_id: int
    text_id: int
    spawn_time: float
    deadline_duration: Optional[float]


@dataclass
class StreamResult:
    generated: int
    touched: int
    soft_touched: int
    remaining: int
    simulated_seconds: float
    cleared: bool = False
    clear_time: Optional[float] = None


@dataclass
class RobotAssignmentState:
    target_id: Optional[int] = None
    assigned_time: float = 0.0
    last_progress_time: float = 0.0
    best_distance: float = float("inf")
    cooldown_until: Dict[int, float] = field(default_factory=dict)


class OnlineTargetStreamDemo:
    def __init__(
        self,
        runner: ExperimentRunner,
        executor: PyBulletExecutor,
        workspace: GridWorkspace,
        seed: int,
        robot_bases: Sequence[np.ndarray],
        home_positions: Sequence[np.ndarray],
        replan_interval: float = 1.0,
        touch_threshold: float = 0.16,
        waypoint_tolerance: float = 0.025,
        min_deadline: float = 8.0,
        max_deadline: float = 12.0,
        urgency_bonus: float = 12.0,
        overtime_bonus_rate: float = 3.0,
        overtime_saturation: float = 12.0,
        max_ik_error: float = 0.18,
        assignment_touch_slack: float = 0.01,
        progress_epsilon: float = 0.015,
        stall_timeout: float = 4.0,
        reassign_cooldown: float = 6.0,
        max_assignment_age: float = 20.0,
        soft_touch_margin: float = 0.02,
        strict_touch: bool = False,
    ) -> None:
        self.runner = runner
        self.executor = executor
        self.workspace = workspace
        self.robot_bases = [np.asarray(b, dtype=float) for b in robot_bases]
        self.home_positions = [np.asarray(p, dtype=float) for p in home_positions]
        self.replan_interval = replan_interval
        self.touch_threshold = touch_threshold
        self.waypoint_tolerance = waypoint_tolerance
        self.min_deadline = min_deadline
        self.max_deadline = max_deadline
        self.urgency_bonus = urgency_bonus
        self.overtime_bonus_rate = overtime_bonus_rate
        self.overtime_saturation = overtime_saturation
        self.max_ik_error = max_ik_error
        self.assignment_touch_slack = assignment_touch_slack
        self.progress_epsilon = progress_epsilon
        self.stall_timeout = stall_timeout
        self.reassign_cooldown = reassign_cooldown
        self.max_assignment_age = max_assignment_age
        self.soft_touch_margin = soft_touch_margin
        self.strict_touch = strict_touch
        self.rng = np.random.default_rng(seed)
        self._next_target_id = 0
        self._stream_center = 0.5 * (self.workspace.world_min + self.workspace.world_max)
        self._stream_center[2] = 0.88

    def _robot_frame(self, robot_index: int) -> Tuple[np.ndarray, np.ndarray]:
        outward = np.asarray(self.home_positions[robot_index][:2], dtype=float)
        outward /= max(np.linalg.norm(outward), 1e-6)
        tangent = np.array([-outward[1], outward[0]], dtype=float)
        return outward, tangent

    def _assignment_reach_limit(self) -> float:
        if self.strict_touch:
            return min(self.max_ik_error, self.touch_threshold)
        return min(self.max_ik_error, self.touch_threshold + self.assignment_touch_slack)

    def _reachable_by_any_robot(self, position: np.ndarray) -> bool:
        effective_limit = self._assignment_reach_limit()
        robot_order = sorted(
            range(len(self.home_positions)),
            key=lambda robot_id: float(
                np.linalg.norm(np.asarray(self.home_positions[robot_id], dtype=float) - np.asarray(position, dtype=float))
            ),
        )
        for robot_id in robot_order:
            if self.executor.estimate_ik_error(robot_id, np.asarray(position, dtype=float)) <= effective_limit:
                return True
        return False

    def _sample_target(
        self,
        robot_index_hint: Optional[int] = None,
        mode_hint: Optional[str] = None,
    ) -> TargetPoint:
        margin = np.array([0.05, 0.05, 0.05], dtype=float)
        low = self.workspace.world_min + margin
        high = self.workspace.world_max - margin

        for _ in range(80):
            if robot_index_hint is None:
                robot_index = int(self.rng.integers(0, len(self.home_positions)))
            else:
                robot_index = int(robot_index_hint % len(self.home_positions))
            outward, tangent = self._robot_frame(robot_index)
            if mode_hint is None:
                mode = str(self.rng.choice(["inward", "flank", "outer"], p=[0.28, 0.32, 0.40]))
            else:
                mode = str(mode_hint)

            if mode == "inward":
                radial = self.rng.uniform(0.08, 0.24)
                lateral = self.rng.normal(0.0, 0.08)
                xy = radial * (-outward) + lateral * tangent
                z = self.rng.uniform(0.74, 1.04)
                utility = float(self.rng.uniform(48.0, 60.0))
            elif mode == "flank":
                radial = self.rng.uniform(0.34, 0.56)
                lateral = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.16, 0.30)
                xy = radial * outward + lateral * tangent
                z = self.rng.uniform(0.72, 1.10)
                utility = float(self.rng.uniform(44.0, 58.0))
            else:
                radial = self.rng.uniform(0.56, 0.69)
                lateral = self.rng.choice([-1.0, 1.0]) * self.rng.uniform(0.00, 0.16)
                xy = radial * outward + lateral * tangent
                z = self.rng.uniform(0.70, 1.12)
                utility = float(self.rng.uniform(42.0, 54.0))

            jitter = self.rng.normal(loc=0.0, scale=np.array([0.03, 0.03, 0.05], dtype=float))
            pos = np.clip(np.array([xy[0], xy[1], z], dtype=float) + jitter, low, high)
            if not self._reachable_by_any_robot(pos):
                continue
            target = TargetPoint(
                target_id=self._next_target_id,
                position=np.asarray(pos, dtype=float),
                utility=utility,
            )
            self._next_target_id += 1
            return target

        raise RuntimeError("Failed to sample a target reachable by any robot after 80 attempts.")

    def _remaining_time(self, active: ActiveStreamTarget, sim_time: float) -> float:
        if active.deadline_duration is None:
            return float("inf")
        return active.deadline_duration - max(0.0, sim_time - active.spawn_time)

    def _effective_utility(self, active: ActiveStreamTarget, sim_time: float) -> float:
        if active.deadline_duration is None:
            return float(active.target.utility)
        remaining = self._remaining_time(active, sim_time)
        elapsed = max(0.0, sim_time - active.spawn_time)
        progress = min(1.0, elapsed / max(active.deadline_duration, 1e-6))
        overdue = max(0.0, -remaining)
        # Under overload, unbounded linear lateness rewards make ancient tasks dominate forever.
        # A saturating overtime term still increases value after timeout, but prevents the queue
        # from collapsing into repeated retries on the oldest far-away targets.
        saturated_overdue = self.overtime_saturation * np.tanh(overdue / max(self.overtime_saturation, 1e-6))
        return float(
            active.target.utility
            + self.urgency_bonus * progress
            + self.overtime_bonus_rate * saturated_overdue
        )

    def _label_for_target(self, active: ActiveStreamTarget, sim_time: float) -> str:
        if active.deadline_duration is None:
            return f"u={active.target.utility:04.1f}"
        remaining = self._remaining_time(active, sim_time)
        effective_utility = self._effective_utility(active, sim_time)
        if remaining >= 0.0:
            return f"{remaining:04.1f}s | u={effective_utility:04.1f}"
        return f"late {abs(remaining):04.1f}s | u={effective_utility:04.1f}"

    def _refresh_labels(self, active_targets: Sequence[ActiveStreamTarget], sim_time: float) -> None:
        for active in active_targets:
            if active.deadline_duration is None and active.text_id < 0:
                continue
            active.text_id = self.executor.update_debug_text(
                active.target.position,
                active.text_id,
                text=self._label_for_target(active, sim_time),
                text_offset=np.array([0.0, 0.0, 0.03], dtype=float),
                text_color=[0.85, 0.1, 0.1],
                text_size=1.2,
            )

    def _cooldown_active(self, state: RobotAssignmentState, target_id: int, sim_time: float) -> bool:
        until = state.cooldown_until.get(target_id)
        return until is not None and sim_time < until

    def _release_assignment(
        self,
        robot_id: int,
        states: Dict[int, RobotAssignmentState],
        current_waypoints: Dict[int, List[np.ndarray]],
        sim_time: float,
        keep_cooldown: bool = True,
    ) -> Optional[int]:
        state = states[robot_id]
        target_id = state.target_id
        if target_id is None:
            return None
        if keep_cooldown:
            state.cooldown_until[target_id] = sim_time + self.reassign_cooldown
        state.target_id = None
        state.assigned_time = 0.0
        state.last_progress_time = 0.0
        state.best_distance = float("inf")
        current_waypoints[robot_id] = [self.home_positions[robot_id].copy()]
        return target_id

    def _path_is_reachable(self, robot_id: int, path: CandidatePath) -> bool:
        # A coarse IK gate is enough here: it filters out targets that look good in the
        # abstract market layer but cannot be reached robustly by the PyBullet robot.
        final_error = self.executor.estimate_ik_error(robot_id, path.waypoints[-1])
        effective_limit = self._assignment_reach_limit()
        return final_error <= effective_limit

    def _plan_reachable_assignments(
        self,
        active_targets: Sequence[ActiveStreamTarget],
        ee_positions: Sequence[np.ndarray],
        free_robot_ids: Sequence[int],
        states: Dict[int, RobotAssignmentState],
        sim_time: float,
    ) -> Dict[int, CandidatePath]:
        busy_target_ids = {state.target_id for state in states.values() if state.target_id is not None}
        available_targets = [
            TargetPoint(
                target_id=active.target.target_id,
                position=active.target.position.copy(),
                utility=self._effective_utility(active, sim_time),
            )
            for active in active_targets
            if active.target.target_id not in busy_target_ids
        ]
        if not free_robot_ids or not available_targets:
            self.workspace.prices = self.workspace.xp.clip(
                self.workspace.prices * (1.0 - self.runner.config.decay),
                0.0,
                self.runner.config.max_price,
            )
            return {}

        agents = self.runner.build_agents_for_starts(
            self.workspace,
            available_targets,
            [ee_positions[robot_id] for robot_id in free_robot_ids],
            robot_bases=[self.robot_bases[robot_id] for robot_id in free_robot_ids],
            robot_ids=free_robot_ids,
        )

        scored_candidates: List[Tuple[float, float, int, CandidatePath]] = []
        for agent in agents:
            state = states[agent.robot_id]
            best_by_target: Dict[int, CandidatePath] = {}
            best_score_by_target: Dict[int, float] = {}
            for path in agent.candidate_paths:
                if self._cooldown_active(state, path.target_id, sim_time):
                    continue
                if not self._path_is_reachable(agent.robot_id, path):
                    continue
                score = path.score(self.workspace, price_weight=self.runner.config.price_weight, use_pricing=True)
                prev = best_by_target.get(path.target_id)
                prev_score = best_score_by_target.get(path.target_id)
                if prev is None or score > float(prev_score) + 1e-9 or (
                    abs(score - float(prev_score)) <= 1e-9 and path.path_cost < prev.path_cost
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

        demand = self.workspace.zeros_like_prices(dtype=float)
        for path in assigned_paths.values():
            for slot in path.resource_slots:
                demand[slot] += 1.0
        delta = demand - self.workspace.capacity
        updated = self.workspace.prices * (1.0 - self.runner.config.decay) + self.runner.config.alpha * delta
        self.workspace.prices = self.workspace.xp.clip(updated, 0.0, self.runner.config.max_price)
        return assigned_paths

    def _release_stalled_assignments(
        self,
        active_targets: Sequence[ActiveStreamTarget],
        ee_positions: Sequence[np.ndarray],
        states: Dict[int, RobotAssignmentState],
        current_waypoints: Dict[int, List[np.ndarray]],
        sim_time: float,
        verbose: bool,
    ) -> Tuple[bool, List[int]]:
        active_by_id = {active.target.target_id: active for active in active_targets}
        released_any = False
        soft_touched_ids: List[int] = []
        for robot_id, state in states.items():
            target_id = state.target_id
            if target_id is None:
                continue
            active = active_by_id.get(target_id)
            if active is None:
                self._release_assignment(robot_id, states, current_waypoints, sim_time, keep_cooldown=False)
                released_any = True
                continue

            distance = float(np.linalg.norm(np.asarray(ee_positions[robot_id], dtype=float) - active.target.position))
            if distance + self.progress_epsilon < state.best_distance:
                state.best_distance = distance
                state.last_progress_time = sim_time
                continue

            assigned_age = sim_time - state.assigned_time
            stalled_for = sim_time - state.last_progress_time
            if assigned_age >= self.max_assignment_age or stalled_for >= self.stall_timeout:
                if (not self.strict_touch) and distance <= self.touch_threshold + self.soft_touch_margin:
                    self._release_assignment(robot_id, states, current_waypoints, sim_time, keep_cooldown=False)
                    soft_touched_ids.append(target_id)
                    released_any = True
                    if verbose:
                        print(
                            f"[Stream] soft-complete R{robot_id} on T{target_id}: "
                            f"distance={distance:.3f}, assigned={assigned_age:.1f}s, stalled={stalled_for:.1f}s"
                        )
                    continue
                self._release_assignment(robot_id, states, current_waypoints, sim_time, keep_cooldown=True)
                released_any = True
                if verbose:
                    print(
                        f"[Stream] release R{robot_id} from T{target_id}: "
                        f"distance={distance:.3f}, assigned={assigned_age:.1f}s, stalled={stalled_for:.1f}s"
                    )
        return released_any, soft_touched_ids

    def _spawn_stream_target(self, sim_time: float) -> ActiveStreamTarget:
        target = self._sample_target()
        deadline_duration = float(self.rng.uniform(self.min_deadline, self.max_deadline))
        body_id, text_id = self.executor.create_sphere_marker(
            target.position,
            radius=0.012,
            rgba=[1.0, 0.1, 0.1, 0.95],
            text=f"{deadline_duration:04.1f}s | u={target.utility:04.1f}",
            text_offset=np.array([0.0, 0.0, 0.03], dtype=float),
            text_color=[0.85, 0.1, 0.1],
            text_size=1.2,
        )
        return ActiveStreamTarget(
            target=target,
            body_id=body_id,
            text_id=text_id,
            spawn_time=sim_time,
            deadline_duration=deadline_duration,
        )

    def _spawn_batch_target(
        self,
        sim_time: float,
        robot_index_hint: Optional[int] = None,
        mode_hint: Optional[str] = None,
    ) -> ActiveStreamTarget:
        target = self._sample_target(robot_index_hint=robot_index_hint, mode_hint=mode_hint)
        body_id, text_id = self.executor.create_sphere_marker(
            target.position,
            radius=0.012,
            rgba=[1.0, 0.1, 0.1, 0.95],
            text=None,
        )
        return ActiveStreamTarget(
            target=target,
            body_id=body_id,
            text_id=text_id,
            spawn_time=sim_time,
            deadline_duration=None,
        )

    def _collect_touched(
        self,
        active_targets: List[ActiveStreamTarget],
        ee_positions: Sequence[np.ndarray],
    ) -> List[int]:
        touched_ids: List[int] = []
        survivors: List[ActiveStreamTarget] = []
        for active in active_targets:
            touched = any(
                np.linalg.norm(np.asarray(pos, dtype=float) - active.target.position) <= self.touch_threshold
                for pos in ee_positions
            )
            if touched:
                self.executor.remove_marker(active.body_id, active.text_id)
                touched_ids.append(active.target.target_id)
            else:
                survivors.append(active)
        active_targets[:] = survivors
        return touched_ids

    def _run_loop(
        self,
        active_targets: List[ActiveStreamTarget],
        generated: int,
        duration: float,
        verbose: bool,
        status_prefix: str,
        spawn_interval: Optional[float] = None,
        total_points: Optional[int] = None,
        stop_when_cleared: bool = False,
    ) -> StreamResult:
        next_spawn_time = 0.0
        sim_time = 0.0
        last_replan_time = -1e9
        last_status_second = -1

        current_waypoints: Dict[int, List[np.ndarray]] = {
            robot_id: [self.home_positions[robot_id].copy()]
            for robot_id in range(len(self.home_positions))
        }
        robot_states: Dict[int, RobotAssignmentState] = {
            robot_id: RobotAssignmentState() for robot_id in range(len(self.home_positions))
        }

        touched = 0
        soft_touched = 0
        needs_replan = True

        while sim_time < duration:
            while (
                spawn_interval is not None
                and total_points is not None
                and generated < total_points
                and sim_time + 1e-9 >= next_spawn_time
            ):
                active_targets.append(self._spawn_stream_target(sim_time))
                generated += 1
                next_spawn_time += spawn_interval
                needs_replan = True

            ee_positions = [self.executor.get_ee_position(robot_id) for robot_id in range(len(self.home_positions))]
            touched_now = self._collect_touched(active_targets, ee_positions)
            if touched_now:
                touched += len(touched_now)
                touched_set = set(touched_now)
                for robot_id, state in robot_states.items():
                    if state.target_id in touched_set:
                        self._release_assignment(
                            robot_id,
                            robot_states,
                            current_waypoints,
                            sim_time,
                            keep_cooldown=False,
                        )
                needs_replan = True
            released_any, soft_touched_now = self._release_stalled_assignments(
                active_targets,
                ee_positions,
                robot_states,
                current_waypoints,
                sim_time,
                verbose=verbose,
            )
            if soft_touched_now:
                soft_touched += len(soft_touched_now)
                soft_touched_set = set(soft_touched_now)
                survivors: List[ActiveStreamTarget] = []
                for active in active_targets:
                    if active.target.target_id in soft_touched_set:
                        self.executor.remove_marker(active.body_id, active.text_id)
                    else:
                        survivors.append(active)
                active_targets[:] = survivors
            if released_any:
                needs_replan = True

            if needs_replan or sim_time - last_replan_time >= self.replan_interval:
                free_robot_ids = [robot_id for robot_id, state in robot_states.items() if state.target_id is None]
                selected_paths = self._plan_reachable_assignments(
                    active_targets,
                    ee_positions,
                    free_robot_ids,
                    robot_states,
                    sim_time,
                )
                active_by_id = {active.target.target_id: active for active in active_targets}

                for robot_id in free_robot_ids:
                    if robot_id not in selected_paths:
                        current_waypoints[robot_id] = [self.home_positions[robot_id].copy()]
                        robot_states[robot_id].target_id = None
                        continue
                    # Skip the first point because it is the current EE state used during replanning.
                    current_waypoints[robot_id] = [
                        np.asarray(wp, dtype=float).copy() for wp in selected_paths[robot_id].waypoints[1:]
                    ]
                    if not current_waypoints[robot_id]:
                        current_waypoints[robot_id] = [self.home_positions[robot_id].copy()]
                    target_id = selected_paths[robot_id].target_id
                    robot_states[robot_id].target_id = target_id
                    robot_states[robot_id].assigned_time = sim_time
                    robot_states[robot_id].last_progress_time = sim_time
                    target_pos = active_by_id[target_id].target.position
                    robot_states[robot_id].best_distance = float(
                        np.linalg.norm(np.asarray(ee_positions[robot_id], dtype=float) - target_pos)
                    )

                last_replan_time = sim_time
                needs_replan = False
                if status_prefix == "Stream":
                    self._refresh_labels(active_targets, sim_time)

            for robot_id in range(len(self.home_positions)):
                ee_pos = ee_positions[robot_id]
                while current_waypoints[robot_id]:
                    wp = current_waypoints[robot_id][0]
                    if np.linalg.norm(ee_pos - wp) > self.waypoint_tolerance:
                        break
                    current_waypoints[robot_id].pop(0)
                if current_waypoints[robot_id]:
                    target_wp = current_waypoints[robot_id][0]
                elif robot_states[robot_id].target_id is None:
                    target_wp = self.home_positions[robot_id]
                else:
                    target_wp = ee_pos
                self.executor.set_ee_target(robot_id, target_wp)

            self.executor.step_simulation(1)
            sim_time += self.executor.time_step

            if stop_when_cleared and generated >= len(active_targets) + touched + soft_touched and not active_targets:
                return StreamResult(
                    generated=generated,
                    touched=touched,
                    soft_touched=soft_touched,
                    remaining=0,
                    simulated_seconds=sim_time,
                    cleared=True,
                    clear_time=sim_time,
                )

            current_second = int(sim_time)
            if verbose and current_second != last_status_second and current_second <= int(duration):
                last_status_second = current_second
                if status_prefix == "Stream":
                    self._refresh_labels(active_targets, sim_time)
                    overdue_count = sum(1 for active in active_targets if self._remaining_time(active, sim_time) < 0.0)
                    max_utility = 0.0
                    if active_targets:
                        max_utility = max(self._effective_utility(active, sim_time) for active in active_targets)
                    print(
                        f"[Stream {current_second:03d}s] generated={generated:3d}, "
                        f"touched={touched:3d}, remaining={len(active_targets):3d}, "
                        f"overdue={overdue_count:3d}, max_u={max_utility:05.1f}"
                    )
                else:
                    print(
                        f"[Batch  {current_second:03d}s] total={generated:3d}, "
                        f"touched={touched:3d}, soft={soft_touched:3d}, remaining={len(active_targets):3d}"
                    )

        return StreamResult(
            generated=generated,
            touched=touched,
            soft_touched=soft_touched,
            remaining=len(active_targets),
            simulated_seconds=sim_time,
            cleared=len(active_targets) == 0,
            clear_time=sim_time if len(active_targets) == 0 else None,
        )

    def run(
        self,
        total_points: int = 100,
        duration: float = 100.0,
        verbose: bool = True,
    ) -> StreamResult:
        spawn_interval = duration / max(1, total_points)
        active_targets: List[ActiveStreamTarget] = []

        if verbose:
            print("\n=== Streaming Targets ===")
            print(
                f"Generating {total_points} random targets over {duration:.1f}s; "
                "active untapped points stay red, touched points disappear."
            )

        return self._run_loop(
            active_targets=active_targets,
            generated=0,
            duration=duration,
            verbose=verbose,
            status_prefix="Stream",
            spawn_interval=spawn_interval,
            total_points=total_points,
            stop_when_cleared=False,
        )

    def run_batch(
        self,
        total_points: int = 40,
        max_duration: float = 120.0,
        verbose: bool = True,
    ) -> StreamResult:
        mode_cycle = ["inward", "flank", "outer", "outer"]
        active_targets = [
            self._spawn_batch_target(
                0.0,
                robot_index_hint=i,
                mode_hint=mode_cycle[i % len(mode_cycle)],
            )
            for i in range(total_points)
        ]

        if verbose:
            print("\n=== Batch Targets ===")
            print(
                f"Spawned {total_points} random targets at t=0.0s; "
                "measure simulated time until all are touched."
            )

        return self._run_loop(
            active_targets=active_targets,
            generated=total_points,
            duration=max_duration,
            verbose=verbose,
            status_prefix="Batch",
            spawn_interval=None,
            total_points=total_points,
            stop_when_cleared=True,
        )
