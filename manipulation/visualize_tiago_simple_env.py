#!/usr/bin/env python3
"""
Visualize the env_tiago_14d_simple benchmark environment (with flanking
obstacles that were added to make the planning problem non-trivial).

Run:
    python visualize_tiago_simple_env.py
"""

import os
import sys
import time
import importlib.util
import numpy as np
import pybullet as p
import pybullet_data

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# ── Load tiago_pro.py constants / helpers ────────────────────────────
_spec = importlib.util.spec_from_file_location(
    'tiago_pro_demo', os.path.join(HERE, 'tiago_pro.py'))
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

# ── Connect GUI ───────────────────────────────────────────────────────
cid = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
p.setGravity(0, 0, -9.81, physicsClientId=cid)

p.resetDebugVisualizerCamera(
    cameraDistance=2.0,
    cameraYaw=120,
    cameraPitch=-25,
    cameraTargetPosition=[0.2, 0.0, 0.8],
    physicsClientId=cid)

# ── Load Tiago Pro ────────────────────────────────────────────────────
tiago_orn = p.getQuaternionFromEuler([0, 0, demo.TIAGO_YAW])
rid = p.loadURDF(demo.TIAGO_URDF,
                 basePosition=[demo.TIAGO_X, demo.TIAGO_Y, 0.0],
                 baseOrientation=tiago_orn,
                 useFixedBase=True,
                 physicsClientId=cid)

jmap, lmap = demo.build_joint_maps(cid, rid)
arm_left_idx  = [jmap[n] for n in demo.ARM_LEFT_JOINTS]
arm_right_idx = [jmap[n] for n in demo.ARM_RIGHT_JOINTS]
torso_idx     = jmap[demo.TORSO_JOINT]
ee_left_idx   = lmap[demo.EE_LEFT_LINK]
ee_right_idx  = lmap[demo.EE_RIGHT_LINK]

# Fix torso at midpoint — lock it with POSITION_CONTROL so gravity cannot move it
_ti = p.getJointInfo(rid, torso_idx, physicsClientId=cid)
torso_mid = 0.5 * (float(_ti[8]) + float(_ti[9]))
p.resetJointState(rid, torso_idx, torso_mid, physicsClientId=cid)
p.setJointMotorControl2(rid, torso_idx, p.POSITION_CONTROL,
                        targetPosition=torso_mid, force=10000,
                        physicsClientId=cid)

# Home pose
arm_home_right = demo.ARM_HOME * np.array([-1, 1, -1, 1, -1, 1, -1])
demo.set_joint_values(cid, rid, arm_left_idx,  demo.ARM_HOME)
demo.set_joint_values(cid, rid, arm_right_idx, arm_home_right)

# ── Build original scene (table + box) ───────────────────────────────
table_id, box_id = demo.build_scene(cid)

# ── Add flanking obstacles (same logic as env_tiago_14d_simple) ───────
_obs_h = demo.BOX_H * 1.4 * 0.75   # 25% shorter than original
_obs_w = demo.BOX_L / 2
_obs_l = demo.BOX_W
_obs_z = demo.TABLE_SURFACE_Z + _obs_h / 2
_obs_right_y = (demo.TABLE_CY + demo.TABLE_WID / 2 - _obs_w / 2)
_obs_left_y  = demo.BOX_Y - (_obs_right_y - demo.BOX_Y)

obs_left_id  = demo.add_box(cid,
                             pos=[demo.BOX_X, _obs_left_y,  _obs_z],
                             he=[_obs_l / 2, _obs_w / 2, _obs_h / 2],
                             color=[0.70, 0.25, 0.25, 1.0])
obs_right_id = demo.add_box(cid,
                             pos=[demo.BOX_X, _obs_right_y, _obs_z],
                             he=[_obs_l / 2, _obs_w / 2, _obs_h / 2],
                             color=[0.70, 0.25, 0.25, 1.0])

obstacles = [table_id, box_id, obs_left_id, obs_right_id]
print(f"Scene: table={table_id}  box={box_id}  "
      f"obs_left={obs_left_id}  obs_right={obs_right_id}")
print(f"Flanking obstacle positions:")
print(f"  left  y = {_obs_left_y:.3f}  (height = {_obs_h:.3f})")
print(f"  right y = {_obs_right_y:.3f}")

# ── Solve IK for goal pose ────────────────────────────────────────────
all_movable = [i for i in range(p.getNumJoints(rid, physicsClientId=cid))
               if p.getJointInfo(rid, i, physicsClientId=cid)[2] != p.JOINT_FIXED]

lpos, lorn, rpos, rorn = demo.compute_goal_targets()
q_left,  pos_err_l, orn_err_l = demo.solve_arm_ik(
    cid, rid, arm_left_idx, ee_left_idx, lpos, lorn,
    all_movable, demo.ARM_HOME, obstacle_ids=obstacles,
    label='LEFT', n_seeds=200)
demo.set_joint_values(cid, rid, arm_left_idx, q_left)

q_right, pos_err_r, orn_err_r = demo.solve_arm_ik(
    cid, rid, arm_right_idx, ee_right_idx, rpos, rorn,
    all_movable, arm_home_right, obstacle_ids=obstacles,
    label='RIGHT', n_seeds=200)
demo.set_joint_values(cid, rid, arm_right_idx, q_right)

print(f"\nGoal IK errors — LEFT  pos={float(np.atleast_1d(pos_err_l)[0]):.4f}m"
      f"  orn={float(np.atleast_1d(orn_err_l)[0]):.4f}")
print(f"               — RIGHT pos={float(np.atleast_1d(pos_err_r)[0]):.4f}m"
      f"  orn={float(np.atleast_1d(orn_err_r)[0]):.4f}")

# Draw goal EE positions as red dots
def _ee_pos(arm_idx, ee_idx):
    demo.set_joint_values(cid, rid, arm_idx, q_left if arm_idx is arm_left_idx else q_right)
    ls = p.getLinkState(rid, ee_idx, physicsClientId=cid)
    return list(ls[4])

ee_l_pos = p.getLinkState(rid, ee_left_idx,  physicsClientId=cid)[4]
ee_r_pos = p.getLinkState(rid, ee_right_idx, physicsClientId=cid)[4]
for pos in [ee_l_pos, ee_r_pos]:
    p.addUserDebugPoints([pos], [[1, 0, 0]], pointSize=10, physicsClientId=cid)

print("\nVisualization running — close the window or press Ctrl+C to exit.")
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=cid)

try:
    while True:
        p.stepSimulation(physicsClientId=cid)
        time.sleep(1 / 60)
except KeyboardInterrupt:
    pass

p.disconnect(cid)
