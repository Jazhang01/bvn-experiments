from goal_env.robotics.fetch_env import FetchEnv
from goal_env.robotics.fetch.slide import FetchSlideEnv
from goal_env.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from goal_env.robotics.fetch.push import FetchPushEnv
from goal_env.robotics.fetch.reach import FetchReachEnv
from goal_env.robotics.fetch.push_low_friction import FetchPushLowFrictionEnv

from gym.envs.registration import registry, register, make, spec
import numpy as np

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    
    for mode in ['Near', 'Far', 'Left', 'Right', 'Train', 'Test']:

        # Keep this for backward compatability
        if mode == 'Train':
            goal_sampling_type = 'near'
        elif mode == 'Test':
            goal_sampling_type = 'far'
        else:
            goal_sampling_type = mode.lower()
        
        kwargs = {
            'reward_type': reward_type,
            'goal_sampling_type': goal_sampling_type,
        }

        # Fetch
        register(
            id='FetchSlide{}{}-v1'.format(mode, suffix),
            entry_point='goal_env.robotics:FetchSlideEnv',
            kwargs={**kwargs, 'far_goal_threshold': 0.2},
            max_episode_steps=50,
        )

        register(
            id='FetchPickAndPlace{}{}-v1'.format(mode, suffix),
            entry_point='goal_env.robotics:FetchPickAndPlaceEnv',
            kwargs={**kwargs, 'far_goal_threshold': 0.2},
            max_episode_steps=50,
        )

        register(
            id='FetchReach{}{}-v1'.format(mode, suffix),
            entry_point='goal_env.robotics:FetchReachEnv',
            kwargs={**kwargs, 'far_goal_threshold': 0.1},
            max_episode_steps=50,
        )

        register(
            id='FetchPush{}{}-v1'.format(mode, suffix),
            entry_point='goal_env.robotics:FetchPushEnv',
            kwargs={**kwargs, 'far_goal_threshold': 0.15},
            max_episode_steps=50,
        )

register(
    id='FetchPushLowFriction-v1',
    entry_point='goal_env.robotics:FetchPushLowFrictionEnv',    
    max_episode_steps=50,
)

register(
    id='FetchPushFixLeft-v1',
    entry_point='goal_env.robotics:FetchPushEnv',
    kwargs={'goal_sampling_type': 'fix', 'fix_goal': np.array([1.27191285, 0.87932346, 0.42469975])},
    max_episode_steps=50,
)

register(
    id='FetchPushFixRight-v1',
    entry_point='goal_env.robotics:FetchPushEnv',
    kwargs={'goal_sampling_type': 'fix', 'fix_goal': np.array([1.32169899, 0.60094283, 0.42469975])},
    max_episode_steps=50,
)
