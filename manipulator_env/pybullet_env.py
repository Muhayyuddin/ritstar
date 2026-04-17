"""
pybullet_env.py — PyBullet simulation of UR10e + Robotiq 85 gripper.

Provides:
  - UR10eRobotiqEnv: loads the URDF, adds obstacles, computes FK/Jacobians,
    collision checking, and visualization of planned paths.
"""

from __future__ import annotations

import os
import time
import numpy as np
import pybullet as p
import pybullet_data
from typing import List, Tuple, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_URDF = os.path.join(_HERE, "models", "ur10e_robotiq85.urdf")


class UR10eRobotiqEnv:
    """PyBullet environment for UR10e + Robotiq 85.

    The 6 UR10e revolute joints are the planning DOFs.
    The gripper is kept at a fixed opening.

    Parameters
    ----------
    gui : bool
        If True, use the PyBullet GUI; otherwise DIRECT (headless).
    obstacles : list of dict
        Each dict describes an obstacle:
          {"type": "box",    "pos": [x,y,z], "half_extents": [hx,hy,hz]}
          {"type": "sphere", "pos": [x,y,z], "radius": r}
          {"type": "cylinder", "pos": [x,y,z], "radius": r, "height": h}
    """

    # UR10e revolute joint indices in the URDF (0-indexed among *movable* joints)
    UR_JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    NUM_DOF = 6

    # Joint limits (from UR10e spec)
    JOINT_LIMITS_LOWER = np.array([-2 * np.pi] * 3 + [-2 * np.pi] * 3)
    JOINT_LIMITS_UPPER = np.array([2 * np.pi] * 3 + [2 * np.pi] * 3)

    def __init__(self, gui: bool = True,
                 obstacles: Optional[List[dict]] = None):
        mode = p.GUI if gui else p.DIRECT
        self.physics_client = p.connect(mode)
        cid = self.physics_client
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)

        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf",
                                   physicsClientId=cid)

        # Load UR10e + Robotiq
        self.robot_id = p.loadURDF(
            _URDF,
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=cid,
        )

        # Build joint-name → index map
        self._joint_name_to_idx = {}
        self._all_joint_indices = []
        n_joints = p.getNumJoints(self.robot_id, physicsClientId=cid)
        for i in range(n_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=cid)
            name = info[1].decode("utf-8")
            joint_type = info[2]
            self._joint_name_to_idx[name] = i
            if joint_type != p.JOINT_FIXED:
                self._all_joint_indices.append(i)

        # UR10e arm joint indices
        self.arm_joint_indices = [
            self._joint_name_to_idx[n] for n in self.UR_JOINT_NAMES
        ]

        # End-effector link index (tool0)
        self.ee_link_idx = self._joint_name_to_idx.get(
            "robotiq_85_base_joint",
            self._joint_name_to_idx.get("flange-tool0", -1),
        )
        # Use the link attached by robotiq_85_base_joint (= child link index)
        # In PyBullet, joint index == child link index
        self.ee_link_idx = self._joint_name_to_idx["robotiq_85_base_joint"]

        # Store obstacle body IDs
        self.obstacle_ids: List[int] = []
        if obstacles:
            for obs in obstacles:
                self._add_obstacle(obs)

        # Disable collisions between gripper mimic links (they overlap)
        self._disable_gripper_self_collision()

        # Step once to settle
        p.stepSimulation(physicsClientId=self.physics_client)

    # ─── Obstacle creation ────────────────────────────────────────

    def _add_obstacle(self, obs: dict) -> int:
        """Add an obstacle to the scene and return its body ID."""
        pos = obs["pos"]
        rgba = obs.get("color", [0.8, 0.2, 0.2, 0.8])
        cid = self.physics_client

        if obs["type"] == "box":
            he = obs["half_extents"]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he,
                                         physicsClientId=cid)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he,
                                      rgbaColor=rgba, physicsClientId=cid)
        elif obs["type"] == "sphere":
            r = obs["radius"]
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=r,
                                         physicsClientId=cid)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=r,
                                      rgbaColor=rgba, physicsClientId=cid)
        elif obs["type"] == "cylinder":
            r = obs["radius"]
            h = obs["height"]
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h,
                                         physicsClientId=cid)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h,
                                      rgbaColor=rgba, physicsClientId=cid)
        else:
            raise ValueError(f"Unknown obstacle type: {obs['type']}")

        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            physicsClientId=cid,
        )
        self.obstacle_ids.append(body_id)
        return body_id

    def _disable_gripper_self_collision(self):
        """Disable self-collision between overlapping gripper links and
        adjacent links connected by fixed joints."""
        gripper_names = [
            n for n in self._joint_name_to_idx
            if "robotiq" in n or "finger" in n
        ]
        gripper_indices = [self._joint_name_to_idx[n] for n in gripper_names]
        cid = self.physics_client
        for i in range(len(gripper_indices)):
            for j in range(i + 1, len(gripper_indices)):
                p.setCollisionFilterPair(
                    self.robot_id, self.robot_id,
                    gripper_indices[i], gripper_indices[j], 0,
                    physicsClientId=cid,
                )
        # Disable collision between wrist_3/flange/tool0 and gripper base
        # (they are connected via fixed joints and overlap)
        ee_chain = []
        for name in ["wrist_3_joint", "wrist_3-flange", "flange-tool0",
                      "robotiq_85_base_joint"]:
            if name in self._joint_name_to_idx:
                ee_chain.append(self._joint_name_to_idx[name])
        for i in range(len(ee_chain)):
            for j in range(i + 1, len(ee_chain)):
                p.setCollisionFilterPair(
                    self.robot_id, self.robot_id,
                    ee_chain[i], ee_chain[j], 0,
                    physicsClientId=cid,
                )
            # Also disable with all gripper links
            for gi in gripper_indices:
                p.setCollisionFilterPair(
                    self.robot_id, self.robot_id,
                    ee_chain[i], gi, 0,
                    physicsClientId=cid,
                )

    # ─── Rendering control ────────────────────────────────────────

    def disable_rendering(self):
        """Disable GUI rendering (useful during planning to avoid visual jitter)."""
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0,
                                   physicsClientId=self.physics_client)

    def enable_rendering(self):
        """Re-enable GUI rendering."""
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1,
                                   physicsClientId=self.physics_client)

    # ─── Robot state ──────────────────────────────────────────────

    def set_joint_positions(self, q: np.ndarray):
        """Instantly set the 6 UR10e joint positions (no simulation step)."""
        cid = self.physics_client
        for idx, val in zip(self.arm_joint_indices, q):
            p.resetJointState(self.robot_id, idx, val, physicsClientId=cid)

    def get_joint_positions(self) -> np.ndarray:
        """Return current 6-DOF joint positions."""
        states = p.getJointStates(self.robot_id, self.arm_joint_indices,
                                  physicsClientId=self.physics_client)
        return np.array([s[0] for s in states])

    def get_ee_pose(self, q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position, quaternion) of the end-effector.

        If *q* is provided, sets joints first (instantaneous).
        """
        if q is not None:
            self.set_joint_positions(q)
        state = p.getLinkState(self.robot_id, self.ee_link_idx,
                               computeForwardKinematics=True,
                               physicsClientId=self.physics_client)
        pos = np.array(state[4])   # worldLinkFramePosition
        orn = np.array(state[5])   # worldLinkFrameOrientation
        return pos, orn

    # ─── Jacobian ─────────────────────────────────────────────────

    def compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute the 6×6 geometric Jacobian at configuration *q*.

        Returns (6, 6) array: [linear_vel (3×6); angular_vel (3×6)].
        """
        self.set_joint_positions(q)
        # PyBullet needs positions/velocities/accelerations for ALL movable joints
        n_movable = len(self._all_joint_indices)
        all_positions = [0.0] * n_movable
        # Fill in with current joint states
        cid = self.physics_client
        for i, jidx in enumerate(self._all_joint_indices):
            state = p.getJointState(self.robot_id, jidx,
                                    physicsClientId=cid)
            all_positions[i] = state[0]

        qd_zero = [0.0] * n_movable
        qdd_zero = [0.0] * n_movable
        jac_t, jac_r = p.calculateJacobian(
            self.robot_id,
            self.ee_link_idx,
            localPosition=[0, 0, 0],
            objPositions=all_positions,
            objVelocities=qd_zero,
            objAccelerations=qdd_zero,
            physicsClientId=cid,
        )
        J_lin = np.array(jac_t)  # (3, n_movable)
        J_rot = np.array(jac_r)  # (3, n_movable)

        # Map arm joint indices to columns in the movable-joint Jacobian
        arm_cols = [self._all_joint_indices.index(j) for j in self.arm_joint_indices]
        J_lin_arm = J_lin[:, arm_cols]
        J_rot_arm = J_rot[:, arm_cols]
        return np.vstack([J_lin_arm, J_rot_arm])  # (6, 6)

    # ─── Collision checking ───────────────────────────────────────

    def is_collision_free(self, q: np.ndarray) -> bool:
        """Return True if configuration *q* is collision-free.

        Checks:
          1. Joint limits
          2. Self-collision of the arm
          3. Collision with obstacles and ground plane
        """
        # Joint limits
        if np.any(q < self.JOINT_LIMITS_LOWER) or np.any(q > self.JOINT_LIMITS_UPPER):
            return False

        self.set_joint_positions(q)
        cid = self.physics_client
        p.performCollisionDetection(physicsClientId=cid)

        # Check collision with each obstacle
        for obs_id in self.obstacle_ids:
            contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=obs_id,
                                          physicsClientId=cid)
            if contacts:
                return False

        # Check collision with ground
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.plane_id,
                                      physicsClientId=cid)
        if contacts:
            # Ignore base-related links touching ground (robot is table-mounted)
            for c in contacts:
                link_idx = c[3]  # linkIndexA
                if link_idx > 1:  # skip base_link (-1), base_link_inertia (0), shoulder (1 is joint idx)
                    return False

        # Self-collision
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id,
                                      physicsClientId=cid)
        if contacts:
            return False

        return True

    def is_edge_collision_free(self, q1: np.ndarray, q2: np.ndarray,
                               n_checks: int = 10) -> bool:
        """Check if straight-line path between q1 and q2 is collision-free."""
        for i in range(n_checks + 1):
            t = i / n_checks
            q = q1 + t * (q2 - q1)
            if not self.is_collision_free(q):
                return False
        return True

    # ─── Visualization ────────────────────────────────────────────

    def visualize_path(self, path: List[np.ndarray],
                       delay: float = 0.03,
                       trail: bool = True):
        """Animate the robot along a joint-space path.

        Parameters
        ----------
        path : list of (6,) arrays
        delay : float — seconds between waypoints
        trail : bool — if True, draw EE trace in green
        """
        prev_pos = None
        for q in path:
            self.set_joint_positions(q)
            p.stepSimulation(physicsClientId=self.physics_client)
            if trail:
                pos, _ = self.get_ee_pose()
                if prev_pos is not None:
                    p.addUserDebugLine(
                        prev_pos.tolist(), pos.tolist(),
                        lineColorRGB=[0, 1, 0], lineWidth=3, lifeTime=0,
                        physicsClientId=self.physics_client,
                    )
                prev_pos = pos
            time.sleep(delay)

    def visualize_config(self, q: np.ndarray, color: List[float] = None):
        """Set the robot to configuration *q* and optionally highlight EE."""
        self.set_joint_positions(q)
        p.stepSimulation(physicsClientId=self.physics_client)
        if color:
            pos, _ = self.get_ee_pose()
            cid = self.physics_client
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.03,
                                      rgbaColor=color,
                                      physicsClientId=cid)
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                              basePosition=pos.tolist(),
                              physicsClientId=cid)

    # ─── Helpers ──────────────────────────────────────────────────

    def get_bounds(self) -> List[Tuple[float, float]]:
        """Return joint-space bounds as list of (lower, upper) tuples."""
        return list(zip(
            self.JOINT_LIMITS_LOWER.tolist(),
            self.JOINT_LIMITS_UPPER.tolist(),
        ))

    def disconnect(self):
        """Disconnect from the physics server."""
        p.disconnect(self.physics_client)

    def __del__(self):
        try:
            p.disconnect(self.physics_client)
        except Exception:
            pass
