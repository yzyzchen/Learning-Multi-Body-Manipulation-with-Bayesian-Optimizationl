import gym
from gym import spaces
import numpy as np

import pybullet as p

from panda_pushing_env import PandaPushingEnv

def planar_pose_to_world_pose(planar_pose):
        theta = planar_pose[-1]
        plane_z = 0
        world_pos = np.array([planar_pose[0], planar_pose[1], plane_z])
        quat = np.array([0., 0., np.sin(theta * 0.5), np.cos(theta * 0.5)])
        world_pose = np.concatenate([world_pos, quat])
        return world_pose

if __name__ == "__main__":
    env = PandaPushingEnv(debug=True)  # 设定 debug=True 以可视化环境
    initial_state = env.reset()  # 重置环境并获取初始状态

    print("Initial State:", initial_state)

    for _ in range(10):
            action = env.action_space.sample()
            next_state, _, _, _ = env.step(action)
            
            state = next_state