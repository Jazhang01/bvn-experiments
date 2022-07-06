import roboverse
import gym

import random
from PIL import Image
import numpy as np
import os
from rl.replays.offline_demo import Replay

import rl

from rl import Args, LfGR
from rl.agent import PixelDDPGAgent
from rl.algo.core import Algo, OfflineAlgo, SACAlgo
from rl.algo.mega import OMEGA
from rl.learner import Learner, PixelLearner, TD3Learner, SACLearner
from rl.utils import vec_env
from rl.launcher import get_env_params, get_env_with_id

import torch

import wandb
from rl.utils.logger import WandBLogger

def load_demo(i=0):    
    import pickle
    demo_path = f'/media/4tb/jason/pybullet_data/data/sawyer_rig_affordances_demos_{i}.pkl'
    with open(demo_path, 'rb') as f:
        demo = pickle.load(f)
    return demo

def look_at_demo(i=0, traj=0, steps=[0, 20, 40, 60, 70]):
    demo = load_demo(i)
    demo_obs = demo[traj]['observations']

    imshape = (3, 48, 48)
    transpose = (2, 1, 0)

    for step in steps:
        demo_img = demo_obs[step]['image_observation'].reshape(imshape)
        demo_img = (np.transpose(demo_img, transpose) * 255).astype(np.uint8)
        demo_im = Image.fromarray(demo_img)

        save_folder = '/home/jason/pybullet/fiddling_imgs/'
        demo_img_name = f'demo_traj{traj}-step{step}.png'
        demo_im.save(os.path.join(save_folder, demo_img_name))
    
    return demo

def look_at_observation(obs, path='tmp.png'):
    obs = (np.transpose(obs, (2, 1, 0)) * 255).astype(np.uint8)
    im = Image.fromarray(obs)
    im.save(path)

def store_demo_in_replay(demo, replay):
    for demo_traj in demo:
        traj = dict(
            observations=[tran['image_observation'] for tran in demo_traj['observations']],
            desired_goals=[tran['desired_goal'] for tran in demo_traj['observations']],
            achieved_goals=[tran['achieved_goal'] for tran in demo_traj['observations']],
            actions=demo_traj['actions'],
            rewards=np.expand_dims(demo_traj['rewards'], axis=1)
        )
        replay.store(traj)

# def store_demo_in_replay(demo, replay, pixel=False, img_shape=None):
#     assert not (pixel and img_shape is None)
#     for demo_traj in demo:
#         traj_length = len(demo_traj['actions'])
#         for i in range(traj_length):
#             if pixel:
#                 ob = demo_traj['observations'][i]['image_observation'].reshape(img_shape)
#                 o2 = demo_traj['next_observations'][i]['image_observation'].reshape(img_shape)
#             else:
#                 ob = demo_traj['observations'][i]['observation']
#                 o2 = demo_traj['next_observations'][i]['observation']
#             ag = demo_traj['observations'][i]['achieved_goal']
#             bg = demo_traj['observations'][i]['desired_goal']
#             a = demo_traj['actions'][i]
#             r = demo_traj['rewards'][i]
            
#             replay.store(ob, ag, bg, a, r, o2)

def store_demos_in_replay(n, replay):
    import time
    start_time = time.time()
    for i in range(n):
        demo = load_demo(i)
        store_demo_in_replay(demo, replay)
    print(f'store time: {time.time() - start_time}')

def get_sawyer_vec_env(num_envs):
    ## For some reason, there is a bug that occurs that gets fixed when I run a reset on some random
    # sawyer rig affordances environment
    e = gym.make('SawyerRigAffordances-v6')
    e.reset()

    from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
    envs = []
    for _ in range(num_envs):
        state_env = gym.make('SawyerRigAffordances-v6')
        imsize = state_env.obs_img_dim

        renderer_kwargs=dict(
                create_image_format='HWC',
                output_image_format='CWH',
                width=imsize,
                height=imsize,
                flatten_image=False)

        renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
        env = InsertImageEnv(state_env, renderer=renderer)
        envs.append(lambda: env)
    
    return vec_env.CustomPixelVecEnv(envs)


# each demo contains 50 trajectories, each trajectory has horizon 75
# less_data: train replay: 0, 5, 10 and test replay: 15     ->  11250/3750 train/test transitions
# more_data: train replay: 0-3, 5-8 and test replay: 10-13  ->  30000/15000 train/test transitions
# many_data: train replay: 0-19 and test replay: 20-29      ->  75000/37500 train/test transitions
def launch():
    env_params, _ = get_env_params(Args.env_name)

    obs_shape = env_params['image_obs']
    goal_shape = env_params['goal']
    action_shape = env_params['action']
    horizon = 75
    
    replay = Replay(obs_shape, goal_shape, action_shape, Args.buffer_size, horizon, gamma=Args.gamma)
    test_replay = Replay(obs_shape, goal_shape, action_shape, Args.buffer_size, horizon, gamma=Args.gamma)
    for i in range(0, 20):
        di = load_demo(i)
        store_demo_in_replay(di, replay)
    for i in range(20, 30):
        di = load_demo(i)
        store_demo_in_replay(di, test_replay)

    if Args.env_name == 'SawyerRigAffordances-v6':
        env = get_sawyer_vec_env(num_envs=Args.n_workers)
    else:
        env = get_env_with_id(num_envs=Args.n_workers, env_id=Args.env_name)
    env.seed(Args.seed)

    if Args.test_env_name is None:
        test_env = None
    else:
        if Args.env_name == 'SawyerRigAffordances-v6':
            test_env = get_sawyer_vec_env(num_envs=Args.n_workers)
        else:
            test_env = get_env_with_id(num_envs=Args.n_workers, env_id=Args.env_name)
        test_env.seed(Args.seed + 100)

    assert Args.agent_type == 'ddpg'
    agent = PixelDDPGAgent(env_params, Args)
    learner = PixelLearner(agent, Args)
    algo = rl.algo.core.TempOfflineAlgoForPixel(env=env, test_env=test_env, env_params=env_params, args=Args, agent=agent, replay=replay,
                        test_replay=test_replay, learner=learner, reward_func=None)

    return algo

def fetch():
    Args.cuda = True
    Args.cuda_name = 'cuda'
    Args.record_video = True

    Args.env_name = 'SawyerRigAffordances-v6'
    Args.n_workers = 2

    Args.buffer_size = 100000
    Args.pixel_obs = True

    Args.n_epochs = 200
    Args.n_cycles = 10
    Args.batch_size = 1024
    Args.n_test_rollouts = 1

    Args.checkpoint_freq = 0  # error occurs when trying to do checkpoint (probably because I'm not using the fetch / jaynes thing)

    seeds = [40, 50, 60]
    methods = ['uvfa', 'sa', 'usfa', 'bvn']
    methods = ['sa']
    critic_types = ['td', 'sa_metric', 'usfa_metric', 'state_asym_metric']
    critic_types = ['sa_metric']

    for seed in seeds:
        for method, critic_type in zip(methods, critic_types):
            Args.critic_type = critic_type

            algo = launch()
            logger = WandBLogger(project_name='SawyerRigAffordances',
                                run_name=f'many_data-{seed}',
                                run_group=method,
                                mode='online')
            algo.fit_replay(logger, steps=5000)
            logger.finish()

fetch()

### Attempt to run fitted critic on environment
### doesn't seem to work; possible cause:
## goal environment is set to one in batch, 
## which may have different configuration than the one after the environment resets

# e = algo.env
# batch = algo.replay.sample(2)
# bg = batch['bg']
# o = e.reset()
# ob = o['image_observation']
# frames = []
# for timestep in range(100):
#     a = algo.best_random_action(e, ob, bg, n=500)
#     o, _, _, _ = e.step(a)
#     ob = o['image_observation']
#     frames.append(ob)

# frames = np.array(frames)
# frames = np.concatenate(frames.transpose([1, 0, 4, 3, 2]))
# video_path = 'video/vid.mp4'
# from ml_logger import logger

# logger.save_video(frames, video_path)

