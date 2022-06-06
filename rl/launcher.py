import random
from functools import partial

import gym
import numpy as np
import torch

from rl import Args, LfGR
from rl.agent import Agent, TD3Agent, SACAgent
from rl.algo.core import Algo, OfflineAlgo, SACAlgo
from rl.algo.mega import OMEGA
from rl.learner import Learner, TD3Learner, SACLearner
from rl.replays.her import Replay
from rl.utils import vec_env


def get_env_params(env_id):
    env = gym.make(env_id)
    # import ipdb; ipdb.set_trace()
    # if env_id == 'Point2D-Easy-UWall-Hard-Init-v2':
    #     env = TimeLimit(GymComputeRewardInterfaceWrapper(env), max_episode_steps=200)
    # elif not isinstance(env.env, TimeLimit):
    #     # HACK: we know multiworld env has no TimeLimit Wrapper
    #     env = GymComputeRewardInterfaceWrapper(env)        
    #     env = TimeLimit(env, max_episode_steps=50)
    #     print('No time limit specified. Use timelimt=50')

    env.seed(100)
    obs = env.reset()
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'action_space': env.action_space,
              'max_timesteps': env._max_episode_steps}
    reward_func = env.compute_reward
    del env  # to avoid memory leak
    return params, reward_func


def get_env_with_id(num_envs, env_id):
    vec_fn = vec_env.DummyVecEnv if Args.debug else vec_env.SubprocVecEnv
    env = vec_fn([lambda: gym.make(env_id) for _ in range(num_envs)])
    env.name = env_id
    return env


def get_env_with_fn(num_envs, env_fn, *args, **kwargs):
    vec_fn = vec_env.DummyVecEnv if Args.debug else vec_env.SubprocVecEnv
    return vec_fn([lambda: env_fn(*args, **kwargs) for _ in range(num_envs)])


def launch(deps=None):
    from ml_logger import logger, RUN

    RUN._update(deps)
    Args._update(deps)
    LfGR._update(deps)
    OMEGA._update(deps)

    if not RUN.resume:
        logger.remove("metrics.pkl", 'videos', 'models', 'outputs.log')

    logger.log_params(Args=vars(Args), LfGR=vars(LfGR), OMEGA=vars(OMEGA))
    logger.log_text("""
             keys:
             - run.status
             - Args.env_name
             - host.hostname
             charts:
             - yKeys: [test/success, train/success/mean]
               xKey: env_steps
               yDomain: [0, 1]
             - yKeys: [test/success, train/success/mean]
               xKey: epoch
               yDomain: [0, 1]
             - yKeys: [q_ag2/mean, q_future/mean, q_bg/mean]
               xKey: epoch
             - yKeys: [intersection_loss/mean, neighbor_loss/mean, tension_loss/mean]
               xKey: epoch
             - yKeys: [r/mean, r_ag_ag2/mean, r_future_ag/mean]
               xKey: epoch
             - type: video
               glob: "**/*_agent.mp4"
             - yKey: dt_epoch
               xKey: epoch
             """, ".charts.yml", dedent=True, overwrite=True)

    torch.set_num_threads(2)

    # rank = mpi_utils.get_rank()
    # seed = Args.seed + rank * Args.n_workers
    random.seed(Args.seed)
    np.random.seed(Args.seed)
    torch.manual_seed(Args.seed)
    if Args.cuda:
        torch.cuda.manual_seed(Args.seed)

    env_params, reward_func = get_env_params(Args.env_name)

    env = get_env_with_id(num_envs=Args.n_workers, env_id=Args.env_name)
    env.seed(Args.seed)    

    if Args.test_env_name is None:
        test_env = None
    else:
        test_env = get_env_with_id(num_envs=Args.n_workers, env_id=Args.test_env_name)
        test_env.seed(Args.seed + 100)

    agent = {'ddpg': Agent, 'td3': TD3Agent, 'sac': SACAgent}[Args.agent_type](env_params, Args)
    replay = Replay(env_params, Args, reward_func)
    learner = {'ddpg': Learner, 'td3': TD3Learner, 'sac': SACLearner}[Args.agent_type](agent, Args)

    if Args.train_type == 'online':
        if Args.agent_type == 'sac':
          algo = SACAlgo(env=env, test_env=test_env, env_params=env_params, args=Args, agent=agent, replay=replay,
                    learner=learner, reward_func=reward_func)
        else:
          algo = Algo(env=env, test_env=test_env, env_params=env_params, args=Args, agent=agent, replay=replay,
                    learner=learner, reward_func=reward_func)
    elif Args.train_type == 'offline':
        algo = OfflineAlgo(env=env, test_env=test_env, env_params=env_params, args=Args, agent=agent, replay=replay,
                           learner=learner, reward_func=reward_func)

    if LfGR.use_lfgr:
        from rl.replays.lfgr import GraphicalReplay
        algo.lfgr = GraphicalReplay(env_params['goal'], partial(env.compute_reward, info=None),
                                    plan_buffer_n=LfGR.plan_buffer_size, d_max=LfGR.d_max,
                                    obs_bucket_size=LfGR.obs_bucket_size)

    if False and Args.debug:
        checkpoint = "/geyang/LfGR/2020/12-09/003_maze/train/22.25.47/PointMaze-v1/0/models/*"
        epoch = 10
        algo.load_checkpoint(checkpoint)
        from tqdm import trange
        from rl.replays.lfgr import visualize_graph

        LfGR.store_interval = 1

        for i in trange(200):
            algo.collect_experience(epoch + i, train_agent=False)
            algo.logger.log_metrics_summary(key_values={"epoch": epoch + i})
            print(len(algo.lfgr.graph), algo.lfgr.k_steps)

        visualize_graph(algo.lfgr.graph, f"figures/{epoch}_graph.png")
    return algo
