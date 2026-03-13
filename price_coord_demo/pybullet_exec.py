from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import CandidatePath, GridWorkspace, RobotAgent, TargetPoint


@dataclass
class PyBulletExecutor:
    gui: bool = True
    time_step: float = 1.0 / 240.0
    ee_link_index: int = 6
    joint_max_velocity: float = 0.9
    position_gain: float = 0.25
    real_time_factor: float = 4.0

    _pb: Optional[object] = field(default=None, init=False, repr=False)
    _client_id: Optional[int] = field(default=None, init=False, repr=False)
    _robot_ids: List[int] = field(default_factory=list, init=False, repr=False)
    _shadow_robot_ids: List[int] = field(default_factory=list, init=False, repr=False)
    _joint_indices: Dict[int, List[int]] = field(default_factory=dict, init=False, repr=False)
    _shadow_joint_indices: Dict[int, List[int]] = field(default_factory=dict, init=False, repr=False)
    _debug_item_ids: List[int] = field(default_factory=list, init=False, repr=False)
    _debug_text_ids: List[int] = field(default_factory=list, init=False, repr=False)

    def _ensure_pybullet(self) -> Tuple[object, object]:
        try:
            import pybullet as p
            import pybullet_data
        except ImportError as exc:
            raise RuntimeError(
                "PyBullet is not installed. Install with: pip install pybullet"
            ) from exc
        return p, pybullet_data

    def connect(self) -> None:
        p, pybullet_data = self._ensure_pybullet()
        self._pb = p
        mode = p.GUI if self.gui else p.DIRECT
        self._client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

    def _sleep_if_needed(self) -> None:
        if not self.gui or self.real_time_factor <= 0.0:
            return
        time.sleep(self.time_step / self.real_time_factor)

    def disconnect(self) -> None:
        if self._pb is not None and self._client_id is not None:
            self._pb.disconnect(self._client_id)
        self._pb = None
        self._client_id = None

    def _hide_robot(self, robot_id: int) -> None:
        assert self._pb is not None
        p = self._pb
        num_joints = p.getNumJoints(robot_id)
        for link_index in range(-1, num_joints):
            try:
                p.changeVisualShape(robot_id, link_index, rgbaColor=[0.0, 0.0, 0.0, 0.0])
            except Exception:
                pass
            try:
                p.setCollisionFilterGroupMask(robot_id, link_index, 0, 0)
            except Exception:
                pass

    def setup_scene(self, robot_bases: Sequence[np.ndarray]) -> None:
        if self._pb is None:
            self.connect()
        assert self._pb is not None
        p = self._pb

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

        p.loadURDF("plane.urdf")
        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.72,
                cameraYaw=42.0,
                cameraPitch=-34.0,
                cameraTargetPosition=[0.0, 0.0, 0.78],
            )

        self._robot_ids.clear()
        self._shadow_robot_ids.clear()
        self._joint_indices.clear()
        self._shadow_joint_indices.clear()

        for i, base in enumerate(robot_bases):
            base = np.asarray(base, dtype=float)
            yaw = float(np.arctan2(-base[1], -base[0]))
            orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
            rid = p.loadURDF("kuka_iiwa/model.urdf", base.tolist(), orn, useFixedBase=True)
            self._robot_ids.append(rid)

            sid = p.loadURDF("kuka_iiwa/model.urdf", base.tolist(), orn, useFixedBase=True)
            self._hide_robot(sid)
            self._shadow_robot_ids.append(sid)

            joints = []
            shadow_joints = []
            n_joints = p.getNumJoints(rid)
            for j in range(n_joints):
                info = p.getJointInfo(rid, j)
                if info[2] == p.JOINT_REVOLUTE:
                    joints.append(j)
                    shadow_joints.append(j)
                    p.resetJointState(rid, j, targetValue=0.0)
                    p.resetJointState(sid, j, targetValue=0.0)
            self._joint_indices[i] = joints
            self._shadow_joint_indices[i] = shadow_joints

        for _ in range(80):
            p.stepSimulation()
            self._sleep_if_needed()

    def _add_debug_item(self, uid: int) -> None:
        if uid >= 0:
            self._debug_item_ids.append(uid)

    def clear_debug(self) -> None:
        if self._pb is None:
            return
        p = self._pb
        for uid in self._debug_item_ids:
            try:
                p.removeBody(uid)
            except Exception:
                pass
        self._debug_item_ids.clear()
        for uid in self._debug_text_ids:
            try:
                p.removeUserDebugItem(uid)
            except Exception:
                pass
        self._debug_text_ids.clear()
        p.removeAllUserDebugItems()

    def create_sphere_marker(
        self,
        position: np.ndarray,
        radius: float = 0.015,
        rgba: Optional[Sequence[float]] = None,
        text: Optional[str] = None,
        text_offset: Optional[np.ndarray] = None,
        text_color: Optional[Sequence[float]] = None,
        text_size: float = 1.2,
    ) -> Tuple[int, int]:
        if self._pb is None:
            return -1, -1
        p = self._pb
        color = list(rgba) if rgba is not None else [1.0, 0.0, 0.0, 0.9]
        sphere = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        body = p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=sphere, basePosition=np.asarray(position).tolist())
        self._add_debug_item(body)

        text_id = -1
        if text:
            offset = np.asarray(text_offset if text_offset is not None else np.array([0.0, 0.0, 0.03]), dtype=float)
            color_rgb = list(text_color) if text_color is not None else [0.1, 0.1, 0.1]
            text_id = p.addUserDebugText(
                text,
                (np.asarray(position, dtype=float) + offset).tolist(),
                textColorRGB=color_rgb,
                textSize=text_size,
            )
            self._debug_text_ids.append(text_id)
        return body, text_id

    def add_debug_text(
        self,
        position: np.ndarray,
        text: str,
        text_offset: Optional[np.ndarray] = None,
        text_color: Optional[Sequence[float]] = None,
        text_size: float = 1.2,
    ) -> int:
        if self._pb is None:
            return -1
        p = self._pb
        offset = np.asarray(text_offset if text_offset is not None else np.array([0.0, 0.0, 0.03]), dtype=float)
        color_rgb = list(text_color) if text_color is not None else [0.1, 0.1, 0.1]
        text_id = p.addUserDebugText(
            text,
            (np.asarray(position, dtype=float) + offset).tolist(),
            textColorRGB=color_rgb,
            textSize=text_size,
        )
        self._debug_text_ids.append(text_id)
        return text_id

    def update_debug_text(
        self,
        position: np.ndarray,
        text_id: int,
        text: str,
        text_offset: Optional[np.ndarray] = None,
        text_color: Optional[Sequence[float]] = None,
        text_size: float = 1.2,
    ) -> int:
        if self._pb is None:
            return -1
        if text_id >= 0:
            try:
                self._pb.removeUserDebugItem(text_id)
            except Exception:
                pass
        return self.add_debug_text(
            position,
            text=text,
            text_offset=text_offset,
            text_color=text_color,
            text_size=text_size,
        )

    def remove_marker(self, body_id: int, text_id: int = -1) -> None:
        if self._pb is None:
            return
        p = self._pb
        if body_id >= 0:
            try:
                p.removeBody(body_id)
            except Exception:
                pass
        if text_id >= 0:
            try:
                p.removeUserDebugItem(text_id)
            except Exception:
                pass

    def add_target_markers(self, targets: Sequence[TargetPoint]) -> None:
        if self._pb is None:
            return
        utilities = np.array([t.utility for t in targets], dtype=float)
        umin, umax = float(np.min(utilities)), float(np.max(utilities))
        span = max(1e-6, umax - umin)

        for t in targets:
            ratio = float((t.utility - umin) / span)
            color = [ratio, 0.2, 1.0 - ratio, 0.9]
            self.create_sphere_marker(
                t.position,
                radius=0.015,
                rgba=color,
                text=f"T{t.target_id}|u={t.utility:.1f}",
                text_offset=np.array([0.0, 0.0, 0.03]),
                text_color=[0.1, 0.1, 0.1],
                text_size=1.2,
            )

    def draw_paths(
        self,
        agents: Sequence[RobotAgent],
        selected_paths: Optional[Dict[int, CandidatePath]] = None,
    ) -> None:
        if self._pb is None:
            return
        p = self._pb

        robot_palette = [
            [0.8, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.8],
            [0.7, 0.4, 0.1],
        ]

        for agent in agents:
            base_color = robot_palette[agent.robot_id % len(robot_palette)]
            for path in agent.candidate_paths:
                for i in range(len(path.waypoints) - 1):
                    p.addUserDebugLine(
                        path.waypoints[i].tolist(),
                        path.waypoints[i + 1].tolist(),
                        lineColorRGB=[0.7, 0.7, 0.7],
                        lineWidth=1.0,
                        lifeTime=0,
                    )

            if selected_paths and agent.robot_id in selected_paths:
                spath = selected_paths[agent.robot_id]
                for i in range(len(spath.waypoints) - 1):
                    p.addUserDebugLine(
                        spath.waypoints[i].tolist(),
                        spath.waypoints[i + 1].tolist(),
                        lineColorRGB=base_color,
                        lineWidth=4.0,
                        lifeTime=0,
                    )

    def draw_high_price_cells(
        self,
        workspace: GridWorkspace,
        threshold: float = 2.0,
        max_cells: int = 30,
    ) -> None:
        if self._pb is None:
            return
        p = self._pb
        top = workspace.top_priced_cells(k=max_cells)
        half = (workspace.cell_size * 0.45).tolist()

        for cell, price in top:
            if price < threshold:
                continue
            center = workspace.grid_to_world(cell)
            alpha = float(min(0.9, 0.2 + price / max(1.0, workspace.max_price_value())))
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half,
                rgbaColor=[1.0, 0.5, 0.0, alpha],
            )
            body = p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=vis, basePosition=center.tolist())
            self._add_debug_item(body)

    def _move_ee_to(self, robot_index: int, target_pos: np.ndarray, steps: int = 90) -> None:
        for _ in range(steps):
            self.set_ee_target(robot_index, target_pos)
            self.step_simulation(1)

    def get_ee_position(self, robot_index: int) -> np.ndarray:
        assert self._pb is not None
        state = self._pb.getLinkState(self._robot_ids[robot_index], self.ee_link_index)
        return np.asarray(state[4], dtype=float)

    def set_ee_target(self, robot_index: int, target_pos: np.ndarray) -> None:
        assert self._pb is not None
        p = self._pb
        rid = self._robot_ids[robot_index]
        joints = self._joint_indices[robot_index]
        ik = p.calculateInverseKinematics(
            rid,
            self.ee_link_index,
            np.asarray(target_pos, dtype=float).tolist(),
            maxNumIterations=80,
            residualThreshold=1e-4,
        )
        max_delta = self.joint_max_velocity * self.time_step
        for j_i, joint in enumerate(joints):
            current_q = float(p.getJointState(rid, joint)[0])
            target_q = float(ik[j_i])
            command_q = current_q + float(np.clip(target_q - current_q, -max_delta, max_delta))
            p.setJointMotorControl2(
                rid,
                joint,
                p.POSITION_CONTROL,
                targetPosition=command_q,
                force=280,
                maxVelocity=self.joint_max_velocity,
                positionGain=self.position_gain,
            )

    def estimate_ik_error(self, robot_index: int, target_pos: np.ndarray) -> float:
        assert self._pb is not None
        p = self._pb
        rid = self._shadow_robot_ids[robot_index]
        joints = self._shadow_joint_indices[robot_index]
        target = np.asarray(target_pos, dtype=float)

        saved_positions = [p.getJointState(rid, joint)[0] for joint in joints]
        ik = p.calculateInverseKinematics(
            rid,
            self.ee_link_index,
            target.tolist(),
            maxNumIterations=80,
            residualThreshold=1e-4,
        )
        try:
            for j_i, joint in enumerate(joints):
                p.resetJointState(rid, joint, targetValue=float(ik[j_i]))
            achieved = np.asarray(p.getLinkState(rid, self.ee_link_index)[4], dtype=float)
        finally:
            for joint, q in zip(joints, saved_positions):
                p.resetJointState(rid, joint, targetValue=float(q))
        return float(np.linalg.norm(achieved - target))

    def step_simulation(self, steps: int = 1) -> None:
        if self._pb is None:
            return
        for _ in range(steps):
            self._pb.stepSimulation()
            self._sleep_if_needed()

    def execute_paths(self, selected_paths: Dict[int, CandidatePath], waypoint_steps: int = 90) -> None:
        if self._pb is None:
            return
        for robot_index, path in selected_paths.items():
            for wp in path.waypoints:
                self._move_ee_to(robot_index, wp, steps=waypoint_steps)

        if self._pb is not None:
            self.step_simulation(steps=240)
