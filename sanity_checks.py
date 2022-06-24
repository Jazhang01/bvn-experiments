import gym
from ml_logger import logger
from rl.agent import SACritic, Agent
from rl import Args, launcher
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

def set_args():
    Args.cuda = True
    Args.cuda_name = 'cuda:3'

    Args.record_video = False  # todo: when this is true, i get a "Offscreen framebuffer is not complete, error 0x8cdd"
    Args.clip_inputs = True
    Args.normalize_inputs = True
    Args.agent_type = 'ddpg'
    Args.critic_type = 'sa_metric'
    Args.critic_reduce_type = 'dot'  # this doesn't affect sa implementation
    Args.hid_size = 176
    Args.metric_embed_dim = 16
    Args.smooth_targ_policy = False

    Args.sa_predict_r = False

    Args.env_name = 'FetchPush-v1'
    Args.n_workers = 8
    Args.n_epochs = 250
    Args.seed = 102 # interesting seeds: 103

def write_env(env, path='tmp_img.png'):
    import cv2
    img = env.render(mode='rgb_array')
    cv2.imwrite(path, img)

def random_walk(env, steps=100):
    for _ in range(steps):
        a = env.action_space.sample()
        env.step(a)

def collect_random_obs_goal_action(env, num=1000):
    observations = [env.observation_space.sample() for _ in range(num)]
    actions = [env.action_space.sample() for _ in range(num)]

    return observations, actions

def collect_obs_goal_action_images_with_policy(env, agent, steps=200, sample_freq=10):
    observations = []
    actions = []
    imgs = []
    observation = env.reset()
    ob = observation['observation']
    bg = observation['desired_goal']
    for step_i in range(steps):
        a = agent.get_actions(ob, bg)

        if step_i % sample_freq == 0:
            observations.append(observation)
            actions.append(a)
            imgs.append(env.render(mode='rgb_array'))
        observation, _, _, info = env.step(a)
        ob = observation['observation']
        bg = observation['desired_goal']
    
    for i, img in enumerate(imgs):
        cv2.imwrite(f'sanity_check_imgs/tmp_img{i}.png', img)

    return observations, actions, imgs

def collect_obs_goal_action_images(env, steps=200, sample_freq=10, action_change_freq=100):
    observations = []
    actions = []
    imgs = []
    a = env.action_space.sample()
    for step_i in range(steps):
        o, _, _, _ = env.step(a)
        if step_i % action_change_freq == 0:
            a = env.action_space.sample()
        if step_i % sample_freq == 0:
            observations.append(o)
            actions.append(a)
            imgs.append(env.render(mode='rgb_array'))

    for i, img in enumerate(imgs):
        cv2.imwrite(f'sanity_check_imgs/tmp_img{i}.png', img)

    return observations, actions, imgs

def get_state_action_embeds(observations, actions, agent, enc):
    state_actions = []
    embeds = []
    for o, a in zip(observations, actions):
        obs, goal = o['observation'], o['desired_goal']
        obs_dim = obs.shape[-1]
        obs_goal = agent._process_inputs(*agent._preprocess_inputs(obs, goal))
        obs = obs_goal[:, :obs_dim]
        action = agent.to_tensor(a).unsqueeze(0)
        sa = torch.cat([obs, action], dim=-1)
        state_actions.append(sa)
        embed = enc(sa)
        embeds.append(embed)
    
    return state_actions, embeds

def get_goal_embeds(observations, agent, enc):
    goals, embeds = [], []
    for o in observations:
        obs, goal = o['observation'], o['desired_goal']
        obs_dim = obs.shape[-1]
        obs_goal = agent._process_inputs(*agent._preprocess_inputs(obs, goal))
        goal = obs_goal[:, obs_dim:]
        goals.append(goal)
        embed = enc(goal)
        embeds.append(embed)
    
    return goals, embeds


def check_nearest_neighbors(embeds):
    nearest_neighbor = {}
    for i, src in enumerate(embeds):
        nearest_index = None
        min_distance = None
        for j, vec in enumerate(embeds):
            dist = torch.linalg.norm(vec - src)
            if i != j and (min_distance is None or dist < min_distance):
                nearest_index = j
                min_distance = dist
        nearest_neighbor[i] = {
            'index': nearest_index,
            'dist': min_distance.item()
        }
    return nearest_neighbor

def pca(embeds, dim=2):
    data_matrix = torch.cat(embeds)
    u,s,v = torch.pca_lowrank(data_matrix)
    data_projected = torch.matmul(data_matrix, v[:, :dim])
    return data_projected.detach().cpu().numpy()

def tsne(embeds):
    from sklearn import manifold
    data_matrix = torch.cat(embeds)
    tsne = manifold.TSNE()
    data_projected = tsne.fit_transform(data_matrix.detach().cpu().numpy())
    return data_projected

def plot_pca_embeds(num, embeds, path='sanity_check_imgs/scatter.png', cmap='viridis'):
    # plt.clf()
    data_projected = pca(embeds, dim=2)
    colors = np.arange(1, num + 1) / (num + 1) * 100
    plt.scatter(data_projected[:num, 0], data_projected[:num, 1], c=colors, cmap=cmap)
    plt.savefig(path)

def plot_tsne_embeds(num, embeds, path='sanity_check_imgs/scatter.png', cmap='viridis'):
    data_projected = tsne(embeds)
    colors = np.arange(1, num + 1) / (num + 1) * 100
    plt.scatter(data_projected[:num, 0], data_projected[:num, 1], c=colors, cmap=cmap)
    plt.savefig(path)
    pass

#################
#################

set_args()

env_params, reward_function = launcher.get_env_params(Args.env_name)
env = gym.make(Args.env_name)
env.seed(Args.seed)

fetch_env = env.unwrapped

Args.hid_size = 176
Args.metric_embed_dim = 16

sa_agent = Agent(env_params, Args)
sa_critic = sa_agent.critic

run_name = 'bvn-saved1'
checkpoint = 'ep_0120'
logger.load_module(sa_agent.critic, f'experiments/003_sa/{run_name}/bvn/bvn/train/ddpg/{Args.env_name}/100/models/{checkpoint}/critic.pkl')
logger.load_module(sa_agent.actor, f'experiments/003_sa/{run_name}/bvn/bvn/train/ddpg/{Args.env_name}/100/models/{checkpoint}/actor.pkl')
logger.load_module(sa_agent.o_normalizer, f'experiments/003_sa/{run_name}/bvn/bvn/train/ddpg/{Args.env_name}/100/models/{checkpoint}/o_norm.pkl')
logger.load_module(sa_agent.g_normalizer, f'experiments/003_sa/{run_name}/bvn/bvn/train/ddpg/{Args.env_name}/100/models/{checkpoint}/g_norm.pkl')

# observations, actions, imgs = collect_obs_goal_action_images(fetch_env, steps=100, sample_freq=2, action_change_freq=100)
observations, actions, imgs = collect_obs_goal_action_images_with_policy(fetch_env, sa_agent, steps=100, sample_freq=2)
trajectory_length = len(observations)

more_obs, more_act = collect_random_obs_goal_action(fetch_env, 1000)
observations.extend(more_obs)
actions.extend(more_act)

state_actions, psi_embeds = get_state_action_embeds(observations, actions, sa_agent, sa_agent.critic.psi)
state_actions, phi_embeds = get_state_action_embeds(observations, actions, sa_agent, sa_agent.critic.phi)
# nearest_neighbor = check_nearest_neighbors(psi_embeds)
# print(nearest_neighbor)

plot_tsne_embeds(trajectory_length, psi_embeds, 'sanity_check_imgs/psi_tsne.png')
plot_tsne_embeds(trajectory_length, phi_embeds, 'sanity_check_imgs/phi_tsne.png', cmap='spring')

# plot_pca_embeds(trajectory_length, psi_embeds, 'sanity_check_imgs/psi_scatter.png')
# plot_pca_embeds(trajectory_length, phi_embeds, 'sanity_check_imgs/phi_scatter.png', cmap='spring')

goals, xi_embeds = get_goal_embeds(observations, sa_agent, sa_agent.critic.xi)
# plot_pca_embeds(1 * trajectory_length, xi_embeds, 'sanity_check_imgs/xi_scatter.png', cmap='gist_heat')

plot_tsne_embeds(trajectory_length, xi_embeds, 'sanity_check_imgs/xi_tsne.png', cmap='gist_heat')

################
plt.clf()
################

Args.critic_type = 'state_asym_metric'
Args.hid_size = 176
Args.metric_embed_dim = 16

bvn_agent = Agent(env_params, Args)
bvn_critic = bvn_agent.critic

run_name = 'bvn-saved1'
checkpoint = 'ep_0040'
logger.load_module(bvn_agent.critic, f'experiments/002_bvn/{run_name}/bvn/bvn/train/ddpg/{Args.env_name}/100/models/{checkpoint}/critic.pkl')
logger.load_module(bvn_agent.actor, f'experiments/002_bvn/{run_name}/bvn/bvn/train/ddpg/{Args.env_name}/100/models/{checkpoint}/actor.pkl')
logger.load_module(bvn_agent.o_normalizer, f'experiments/002_bvn/{run_name}/bvn/bvn/train/ddpg/{Args.env_name}/100/models/{checkpoint}/o_norm.pkl')
logger.load_module(bvn_agent.g_normalizer, f'experiments/002_bvn/{run_name}/bvn/bvn/train/ddpg/{Args.env_name}/100/models/{checkpoint}/g_norm.pkl')

# observations, actions, imgs = collect_obs_goal_action_images(fetch_env, steps=100, sample_freq=2, action_change_freq=100)
observations, actions, imgs = collect_obs_goal_action_images_with_policy(fetch_env, bvn_agent, steps=100, sample_freq=2)
trajectory_length = len(observations)

more_obs, more_act = collect_random_obs_goal_action(fetch_env, 1000)
observations.extend(more_obs)
actions.extend(more_act)

state_actions, f_embeds = get_state_action_embeds(observations, actions, bvn_agent, bvn_agent.critic.f)
# nearest_neighbor = check_nearest_neighbors(f_embeds)
# print(nearest_neighbor)

# plot_pca_embeds(trajectory_length, f_embeds, 'sanity_check_imgs/bvn_scatter.png')
plot_tsne_embeds(trajectory_length, f_embeds, 'sanity_check_imgs/bvn_tsne.png')
