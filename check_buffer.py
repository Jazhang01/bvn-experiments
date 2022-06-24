import numpy as np
import gym
from rl import Args, launcher
from rl.replays.her import Replay

Args.env_name = 'FetchPushLeft-v1'
Args.future_p = 0.8

env_params, reward_function = launcher.get_env_params(Args.env_name)
env = gym.make(Args.env_name)
env.seed(Args.seed)

fetch_env = env.unwrapped

her_replay = Replay(env_params, Args, reward_function)
her_replay.load('/home/jason/bvn/experiments/004_usfa/bvn-no-her-attempt/bvn/bvn/train/ddpg/FetchPushLeft-v1/100/replay/snapshot.pkl')

buffer = {}
for key in her_replay.buffers.keys():
    buffer[key] = her_replay.buffers[key][:her_replay.current_size]

#############

relabel_filter = env_params['is_train_goal']

#############

batch_size = 8

n_trajs = buffer['a'].shape[0]
horizon = buffer['a'].shape[1]
ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
t_samples = np.random.randint(0, horizon, size=batch_size)
batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}

future_offset = (np.random.uniform(size=batch_size) * (horizon - t_samples)).astype(int)
future_t = (t_samples + 1 + future_offset)

candidates1 = np.where(np.apply_along_axis(relabel_filter, 1, buffer['ag'][ep_idxes, future_t]))[0]
print(len(candidates1))
candidates2 = np.where(np.apply_along_axis(relabel_filter, 1, batch['ag']))[0]
print(len(candidates2))
candidates3 = np.where(np.apply_along_axis(relabel_filter, 1, batch['bg']))[0]
print(len(candidates3))
