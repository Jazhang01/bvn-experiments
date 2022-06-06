import os
from gym import utils
import numpy as np

import gym.envs.robotics
from goal_env.robotics import fetch_env


# Ensure we get the path separator correct on windows
MUJOCO_ROBOT_ROOT = os.path.dirname(os.path.abspath(gym.envs.robotics.__file__))
MODEL_XML_PATH = os.path.join(MUJOCO_ROBOT_ROOT, 'assets', 'fetch', 'slide.xml')


class FetchSlideEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', goal_sampling_type='default', far_goal_threshold=None):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.41, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1, target_range=0.3, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type,
            goal_sampling_type=goal_sampling_type, far_goal_threshold=far_goal_threshold)
        utils.EzPickle.__init__(self)
