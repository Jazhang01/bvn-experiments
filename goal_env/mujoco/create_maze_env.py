from .ant_maze_env import AntMazeEnv
from .point_maze_env import PointMazeEnv
from collections import OrderedDict
import gym
import numpy as np
from gym import Wrapper


class MazeWrapper(Wrapper):
    def __init__(self, env, maze_size_scaling, random_start, low, high):
        super(MazeWrapper, self).__init__(env)
        ob_space = env.observation_space
        self.maze_size_scaling = maze_size_scaling
        low = np.array(low, dtype=ob_space.dtype)
        high = np.array(high, dtype=ob_space.dtype)
        maze_low = np.array(np.array([-4, -4]) / 8 * maze_size_scaling, dtype=ob_space.dtype)
        maze_high = np.array(np.array([20, 20]) / 8 * maze_size_scaling, dtype=ob_space.dtype)
        self.maze_size_scaling = maze_size_scaling
        self.goal_space = gym.spaces.Box(low=low, high=high)
        self.maze_space = gym.spaces.Box(low=maze_low, high=maze_high)

        self.goal_dim = low.size
        self.distance_threshold = maze_size_scaling / 8.

        self.observation_space = gym.spaces.Dict(OrderedDict({
            'observation': ob_space,
            'desired_goal': self.goal_space,
            'achieved_goal': self.goal_space,
        }))
        self.goal = None
        self.random_start = random_start

    def seed(self, seed=None):
        # not good, because these can deviate.
        self.action_space.seed(seed)
        self.goal_space.seed(seed)
        self.maze_space.seed(seed)
        return self.env.seed(seed)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        out = {'observation': observation,
               'desired_goal': self.goal,
               'achieved_goal': observation[..., :self.goal_dim]}
        reward = -np.linalg.norm(observation[..., :self.goal_dim] - self.goal, axis=-1)
        info['is_success'] = (reward > -self.distance_threshold)
        reward = self.compute_reward(out['achieved_goal'], self.goal)
        return out, reward, done, info

    # this overwrites the native reset
    def reset(self):
        from termcolor import cprint
        observation = self.env.reset()
        self.goal = self.goal_space.sample()
        while self.env._is_in_collision(self.goal):
            self.goal = self.goal_space.sample()

        # # random start a position without collision
        # if self.random_start:
        #     xy = self.maze_space.sample()
        #     while self.env._is_in_collision(xy):
        #         xy = self.maze_space.sample()
        #     self.env.wrapped_env.set_xy(xy)
        #     observation = self.env._get_obs()

        out = {'observation': observation,
               'desired_goal': self.goal,
               'achieved_goal': observation[..., :self.goal_dim]}
        return out

    def compute_reward(self, achieved_goal, goal, info=None):
        dist = np.linalg.norm(achieved_goal - goal, axis=-1)
        return -(dist > self.distance_threshold).astype(np.float32)

    def render(self, *args, **kwargs):
        self.render_callback()
        return self.env.render(*args, **kwargs)

    def render_callback(self):
        if self.goal is None:
            return
        # the environment uses a wrapped_env for the mujoco model.
        # self.env.unwrapped.sim is not a standard api that the maze_env follows.
        mujoco_env = self.env.unwrapped.wrapped_env
        # Visualize target.
        sites_offset = (mujoco_env.sim.data.site_xpos - mujoco_env.sim.model.site_pos).copy()
        site_id = mujoco_env.sim.model.site_name2id('target0')
        goal = np.concatenate([self.goal, sites_offset[0][-1:]])
        # env.sim.model.site_pos[site_id] = goal - sites_offset[0]
        mujoco_env.sim.model.site_pos[site_id] = goal
        mujoco_env.sim.forward()


def create_maze_env(env_name=None, top_down_view=False, maze_size_scaling=8, random_start=True, goal_args=None):
    goal_args = [] if goal_args is None else goal_args
    n_bins = 0
    manual_collision = False
    if env_name.startswith('Ego'):
        n_bins = 8
        env_name = env_name[3:]
    if env_name.startswith('Ant'):
        manual_collision = True
        cls = AntMazeEnv
        env_name = env_name[3:]
        maze_size_scaling = maze_size_scaling
    elif env_name.startswith('Point'):
        cls = PointMazeEnv
        manual_collision = True
        env_name = env_name[5:]
        maze_size_scaling = maze_size_scaling
    else:
        assert False, 'unknown env %s' % env_name

    observe_blocks = False
    put_spin_near_agent = False
    if env_name == 'Maze':
        maze_id = 'Maze'
    elif env_name == 'Maze-train':
        maze_id = 'Maze-train'
    elif env_name == 'Push':
        maze_id = 'Push'
    elif env_name == 'Fall':
        maze_id = 'Fall'
    elif env_name == 'Block':
        maze_id = 'Block'
        put_spin_near_agent = True
        observe_blocks = True
    elif env_name == 'BlockMaze':
        maze_id = 'BlockMaze'
        put_spin_near_agent = True
        observe_blocks = True
    else:
        raise ValueError('Unknown maze environment %s' % env_name)

    gym_mujoco_kwargs = {
        'maze_id': maze_id,
        'n_bins': n_bins,
        'observe_blocks': observe_blocks,
        'put_spin_near_agent': put_spin_near_agent,
        'top_down_view': top_down_view,
        'manual_collision': manual_collision,
        'maze_size_scaling': maze_size_scaling,
    }
    gym_env = cls(**gym_mujoco_kwargs)
    goal_args = np.array(goal_args) / 8 * maze_size_scaling
    return MazeWrapper(gym_env, maze_size_scaling, random_start, *goal_args)
