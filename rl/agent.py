import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from torch.distributions import Normal, TransformedDistribution

from rl.normalizer import Normalizer
from rl.utils import net_utils
from rl.utils import torch_utils
from rl.utils import mpi_utils

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def target_policy_smooth(batch_action):
    """Add noises to actions for target policy smoothing."""
    noise = torch.clamp(0.2 * torch.randn_like(batch_action), -0.5, 0.5)
    return torch.clamp(batch_action + noise, -1, 1)

class TemperatureHolder(nn.Module):
    """Module that holds a temperature as a learnable value.
    Args:
        initial_log_temperature (float): Initial value of log(temperature).
    """

    def __init__(self, initial_log_temperature=0):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(initial_log_temperature, dtype=torch.float32)
        )

    def forward(self):
        """Return a temperature as a torch.Tensor."""
        return torch.exp(self.log_temperature)


class StochasticActor(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.act_limit = env_params['action_max']

        input_dim = env_params['obs'] + env_params['goal']
        self.net = net_utils.mlp(
            [input_dim] + [args.hid_size] * args.n_hids,
            activation=args.activ, output_activation=args.activ)
        self.mean = nn.Linear(args.hid_size, env_params['action'])
        self.logstd = nn.Linear(args.hid_size, env_params['action'])

    def gaussian_params(self, inputs):
        outputs = self.net(inputs)
        mean, logstd = self.mean(outputs), self.logstd(outputs)
        logstd = torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(logstd)
        return mean, std

    def forward(self, inputs, deterministic=False, with_logprob=True):
        mean, std = self.gaussian_params(inputs)
        pi_dist = Normal(mean, std)
        if deterministic:
            pi_action = mean
        else:
            pi_action = pi_dist.rsample()
        logp_pi = None
        if with_logprob:
            logp_pi = pi_dist.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)
        pi_action = torch.tanh(pi_action) * self.act_limit
        return pi_action, logp_pi


class Qfunc(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        input_dim = env_params['obs'] + env_params['goal'] + env_params['action']
        self.q_func = net_utils.mlp([input_dim] + [args.hid_size] * args.n_hids + [1], activation=args.activ)

    def forward(self, *args):
        q_value = self.q_func(torch.cat([*args], dim=-1))
        return torch.squeeze(q_value, -1)


class DoubleQfunc(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.q1 = Qfunc(env_params, args)
        self.q2 = Qfunc(env_params, args)

    def forward(self, *args):
        q1 = self.q1(*args)
        q2 = self.q2(*args)
        return q1, q2


class Actor(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.act_limit = env_params['action_max']

        input_dim = env_params['obs'] + env_params['goal']
        self.net = net_utils.mlp(
            [input_dim] + [args.hid_size] * args.n_hids,
            activation=args.activ, output_activation=args.activ)
        self.mean = nn.Linear(args.hid_size, env_params['action'])

    def forward(self, inputs):
        outputs = self.net(inputs)
        mean = self.mean(outputs)
        pi_action = torch.tanh(mean) * self.act_limit
        return pi_action


# def squashed_diagonal_gaussian_head(x, out_dim):
#     assert x.shape[-1] == out_dim * 2
#     mean, log_scale = torch.chunk(x, 2, dim=1)
#     log_scale = torch.clamp(log_scale, -20.0, 2.0)
#     var = torch.exp(log_scale * 2)
#     base_distribution = distributions.Independent(
#         distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
#     )
#     # cache_size=1 is required for numerical stability
#     return distributions.transformed_distribution.TransformedDistribution(
#         base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
#     )

# def sac_weights_init(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight)
#         torch.nn.init.zeros_(m.bias)

# class SquashedGaussianActor(nn.Module):
#     def __init__(self, env_params, args):
#         super().__init__()
#         self.act_limit = env_params['action_max']
#         self.act_size = env_params['action']

#         input_dim = env_params['obs'] + env_params['goal']
#         self.net = net_utils.mlp(
#             [input_dim] + [args.hid_size] * args.n_hids,
#             activation=args.activ, output_activation=args.activ)
#         self.mean_std = nn.Linear(args.hid_size, env_params['action'] * 2)

#         self.net.apply(sac_weights_init)
#         torch.nn.init.xavier_uniform_(self.mean_std.weight, gain=args.policy_output_scale)

#     def forward(self, inputs):
#         outputs = self.net(inputs)
#         outputs = self.mean_std(outputs)
#         dist = squashed_diagonal_gaussian_head(outputs, self.act_size)
#         return dist

class SACPolicy(nn.Module):

    def __init__(self, env_params, args):
        super().__init__()
        self.act_limit = env_params['action_max']
        self.act_size = env_params['action']

        input_dim = env_params['obs'] + env_params['goal']
        self.net = net_utils.mlp(
            [input_dim] + [args.hid_size] * args.n_hids,
            activation=args.activ, output_activation=args.activ)
        self.mean_std = nn.Linear(args.hid_size, env_params['action'] * 2)

    def forward(self, x, get_logprob=False, requires_act_grad=True):
        mu_logstd = self.mean_std(self.net(x))
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [torch_utils.TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample() if requires_act_grad else dist.sample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        else:
            logprob = None
        mean = torch.tanh(mu)
        return action, logprob, mean

class SACCritic(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.act_limit = env_params['action_max']

        input_dim = env_params['obs'] + env_params['goal'] + env_params['action']
        self.net = net_utils.mlp(
            [input_dim] + [args.hid_size] * args.n_hids + [1],
            activation=args.activ)

    def forward(self, pi_inputs, actions):
        q_inputs = torch.cat([pi_inputs, actions], dim=-1)
        q_values = self.net(q_inputs).squeeze()
        return q_values


class Critic(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.act_limit = env_params['action_max']

        input_dim = env_params['obs'] + env_params['goal'] + env_params['action']
        self.net = net_utils.mlp(
            [input_dim] + [args.hid_size] * args.n_hids + [1],
            activation=args.activ)

    def forward(self, pi_inputs, actions):
        q_inputs = torch.cat([pi_inputs, actions / self.act_limit], dim=-1)
        q_values = self.net(q_inputs).squeeze()
        return q_values

class StateAsymMetricCritic(nn.Module):
    '''
    Instead of phi(f(s,a), g) we use phi(s, g) to promote full-rank.
    '''
    def __init__(self, env_params, args):
        super().__init__()
        self.args = args
        self.act_limit = env_params['action_max']

        self.obs_dim = env_params['obs']
        self.act_dim = env_params['action']
        self.goal_dim = env_params['goal']

        embed_dim = args.metric_embed_dim
        self.f = net_utils.mlp(
            [env_params['obs'] + env_params['action']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ)
        self.phi = net_utils.mlp(
            [env_params['obs'] + env_params['goal']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ)
        
        if self.args.critic_reduce_type == 'concat':
            self.out = nn.Linear(embed_dim + embed_dim, 1)

    def forward(self, pi_inputs, actions):
        # NOTE: assume pi_inputs to be concatenated in the order [obs, goal]
        obses, goals = pi_inputs[:, :self.obs_dim], pi_inputs[:, self.obs_dim:]

        # f(s, a)
        f_inputs = torch.cat([obses, actions / self.act_limit], dim=-1)
        f_embeds = self.f(f_inputs)

        # \varphi(s, g)
        phi_inputs = torch.cat([obses, goals], dim=-1)
        phi_embeds = self.phi(phi_inputs)

        # ||f(s, a) - \varphi(f(s, a), g)||
        if self.args.critic_reduce_type == 'norm':
            embed_dist = torch.linalg.norm(f_embeds - phi_embeds, dim=-1, ord=self.args.metric_norm_ord)
            q_values = (-embed_dist).squeeze()
        elif self.args.critic_reduce_type == 'dot':
            n_batch, latent_dim = f_embeds.shape
            embed_dist = torch.bmm(f_embeds.view(n_batch, 1, latent_dim), phi_embeds.view(n_batch, latent_dim, 1)).view(n_batch)
            q_values = (embed_dist).squeeze()
        elif self.args.critic_reduce_type == 'concat':
            q_values = self.out(torch.cat([f_embeds, phi_embeds], dim=-1))

        return q_values - 1

    def reinit_phi(self):
        for layer in self.phi.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def reinit_f(self):
        for layer in self.f.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def freeze_phi(self):
        net_utils.set_requires_grad(self.phi, False)

    def freeze_f(self):
        net_utils.set_requires_grad(self.f, False)

class PhiAGMetricCritic(nn.Module):
    '''
    Instead of phi(f(s,a), g) we use phi(s, g) to promote full-rank.
    '''
    def __init__(self, env_params, args):
        super().__init__()
        self.args = args
        self.act_limit = env_params['action_max']

        self.obs_dim = env_params['obs']
        self.act_dim = env_params['action']
        self.goal_dim = env_params['goal']

        embed_dim = args.metric_embed_dim
        self.f = net_utils.mlp(
            [env_params['obs'] + env_params['action']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ)
        self.phi = net_utils.mlp(
            [env_params['goal'] + env_params['action']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ)
        
        if self.args.critic_reduce_type == 'concat':
            self.out = nn.Linear(embed_dim + embed_dim, 1)

    def forward(self, pi_inputs, actions):
        # NOTE: assume pi_inputs to be concatenated in the order [obs, goal]
        obses, goals = pi_inputs[:, :self.obs_dim], pi_inputs[:, self.obs_dim:]

        # f(s, a)
        f_inputs = torch.cat([obses, actions / self.act_limit], dim=-1)
        f_embeds = self.f(f_inputs)

        # \varphi(s, g)
        phi_inputs = torch.cat([goals, actions / self.act_limit], dim=-1)
        phi_embeds = self.phi(phi_inputs)

        # ||f(s, a) - \varphi(f(s, a), g)||
        if self.args.critic_reduce_type == 'norm':
            embed_dist = torch.linalg.norm(f_embeds - phi_embeds, dim=-1, ord=self.args.metric_norm_ord)
            q_values = (-embed_dist).squeeze()
        elif self.args.critic_reduce_type == 'dot':
            n_batch, latent_dim = f_embeds.shape
            embed_dist = torch.bmm(f_embeds.view(n_batch, 1, latent_dim), phi_embeds.view(n_batch, latent_dim, 1)).view(n_batch)
            q_values = -(embed_dist).squeeze()
        elif self.args.critic_reduce_type == 'concat':
            q_values = self.out(torch.cat([f_embeds, phi_embeds], dim=-1))

        return q_values

    def reinit_phi(self):
        for layer in self.phi.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def reinit_f(self):
        for layer in self.f.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def freeze_phi(self):
        net_utils.set_requires_grad(self.phi, False)

    def freeze_f(self):
        net_utils.set_requires_grad(self.f, False)

class AsymMetricCritic(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.args = args
        self.act_limit = env_params['action_max']

        # input_dim = env_params['obs'] + env_params['goal'] + env_params['action']
        self.obs_dim = env_params['obs']
        self.act_dim = env_params['action']
        self.goal_dim = env_params['goal']

        self.state_action_encoder_net = net_utils.mlp(
            [env_params['obs'] + env_params['action']] + [args.hid_size] * args.n_hids + [args.metric_embed_dim],
            activation=args.activ)
        self.goal_encoder_net = net_utils.mlp(
            [env_params['goal']] + [args.hid_size] * args.n_hids + [args.metric_embed_dim],
            activation=args.activ)

    def forward(self, pi_inputs, actions):
        # NOTE: assume pi_inputs to be concatenated in the order [obs, goal]
        obses, goals = pi_inputs[:, :self.obs_dim], pi_inputs[:, self.obs_dim:]

        # f(s, a)        
        f_inputs = torch.cat([obses, actions / self.act_limit], dim=-1)
        f_embeds = self.f(f_inputs)

        # \varphi(f(s,a), g)
        phi_inputs = torch.cat([f_embeds, goals], dim=-1)
        phi_embeds = self.phi(phi_inputs)
        # phi_embeds = self.phi_embed(f_embeds, goals)

        # ||f(s, a) - \varphi(f(s, a), g)||
        if self.args.critic_reduce_type == 'norm':
            embed_dist = torch.linalg.norm(f_embeds - phi_embeds, dim=-1, ord=self.args.metric_norm_ord)    
        elif self.args.critic_reduce_type == 'dot':
            n_batch, latent_dim = f_embeds.shape
            embed_dist = torch.bmm(f_embeds.view(n_batch, 1, latent_dim), phi_embeds.view(n_batch, latent_dim, 1)).view(n_batch)

        q_values = (-embed_dist).squeeze()

        return q_values

class SymMetricCritic(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.args = args
        self.act_limit = env_params['action_max']

        self.obs_dim = env_params['obs']
        self.act_dim = env_params['action']
        self.goal_dim = env_params['goal']

        embed_dim = args.metric_embed_dim
        self.f = net_utils.mlp(
            [env_params['obs'] + env_params['action']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ)
        self.phi = net_utils.mlp(
            [env_params['goal']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ)

    def forward(self, pi_inputs, actions):
        # NOTE: assume pi_inputs to be concatenated in the order [obs, goal]
        obses, goals = pi_inputs[:, :self.obs_dim], pi_inputs[:, self.obs_dim:]
        
        f_inputs = torch.cat([obses, actions / self.act_limit], dim=-1)
        f_embeds = self.f(f_inputs)

        goal_embeds = self.phi(goals)

        if self.args.critic_reduce_type == 'norm':
            state_action_to_goal_distance = torch.linalg.norm(f_embeds - goal_embeds, dim=-1, ord=self.args.metric_norm_ord)
        elif self.args.critic_reduce_type == 'dot':
            n_batch, latent_dim = f_embeds.shape
            state_action_to_goal_distance = torch.bmm(f_embeds.view(n_batch, 1, latent_dim), goal_embeds.view(n_batch, latent_dim, 1)).view(n_batch)

        q_values = (-state_action_to_goal_distance).squeeze()

        return q_values

class FSAGMetricCritic(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.args = args
        self.act_limit = env_params['action_max']

        self.obs_dim = env_params['obs']
        self.act_dim = env_params['action']
        self.goal_dim = env_params['goal']

        embed_dim = args.metric_embed_dim
        self.f = net_utils.mlp(
            [env_params['obs'] + env_params['action'] + env_params['goal']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ)
        self.phi = net_utils.mlp(
            [env_params['goal']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ)

    def forward(self, pi_inputs, actions):
        # NOTE: assume pi_inputs to be concatenated in the order [obs, goal]
        obses, goals = pi_inputs[:, :self.obs_dim], pi_inputs[:, self.obs_dim:]

        # f(s, a, g)
        f_inputs = torch.cat([obses, actions / self.act_limit, goals], dim=-1)
        f_embeds = self.f(f_inputs)        

        # \varphi(g)
        phi_embeds = self.phi(goals)

        # ||f(s, a, g) - \varphi(g)||
        if self.args.critic_reduce_type == 'norm':
            embed_dist = torch.linalg.norm(f_embeds - phi_embeds, dim=-1, ord=self.args.metric_norm_ord)
            q_values = (-embed_dist).squeeze()
        elif self.args.critic_reduce_type == 'dot':
            n_batch, latent_dim = f_embeds.shape
            embed_dist = torch.bmm(f_embeds.view(n_batch, 1, latent_dim), phi_embeds.view(n_batch, latent_dim, 1)).view(n_batch)
            q_values = (-embed_dist).squeeze()
        elif self.args.critic_reduce_type == 'concat':
            q_values = self.out(torch.cat([f_embeds, phi_embeds], dim=-1))

        q_values = (-embed_dist).squeeze()

        return q_values

class AutoEncSFCritic(nn.Module):

    def __init__(self, env_params, args):
        super().__init__()
        self.args = args
        self.act_limit = env_params['action_max']

        self.obs_dim = env_params['obs']
        self.act_dim = env_params['action']
        self.goal_dim = env_params['goal']

        self.latent_dim = embed_dim = args.metric_embed_dim
        self.phi = net_utils.mlp(
            [env_params['obs']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ
        )
        self.dec_phi = net_utils.mlp(
            [embed_dim] + [args.hid_size] * args.n_hids + [env_params['obs']],
            activation=args.activ
        )
        self.psi = net_utils.mlp(
            [embed_dim + env_params['action']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ)
        self.wsg = net_utils.mlp(
            [env_params['obs'] + env_params['goal']] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ)

    def forward(self, pi_inputs, actions):
        # NOTE: assume pi_inputs to be concatenated in the order [obs, goal]
        obses, goals = pi_inputs[:, :self.obs_dim], pi_inputs[:, self.obs_dim:]

        n_batch = pi_inputs.shape[0]

        # q = psi(s,a)^T w(s,g)
        phi_embeds = self.phi(obses).detach()
        psi_embeds = self.psi(torch.cat([phi_embeds, actions / self.act_limit], dim=-1))
        wsg_embeds = self.wsg(pi_inputs)
        
        return torch.bmm(psi_embeds.view(n_batch, 1, self.latent_dim), wsg_embeds.view(n_batch, self.latent_dim, 1)).view(n_batch).squeeze()


class USFACritic(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.args = args
        self.act_limit = env_params['action_max']

        self.obs_dim = env_params['obs']
        self.act_dim = env_params['action']
        self.goal_dim = env_params['goal']

        self.latent_dim = embed_dim = args.metric_embed_dim

        self.phi = net_utils.mlp(
            [self.obs_dim + self.act_dim] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ
        )
        self.psi = net_utils.mlp(
            [self.obs_dim + self.act_dim + self.goal_dim] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ
        )
        self.xi = net_utils.mlp(
            [self.goal_dim] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ
        )
    
    def forward_reward(self, pi_inputs, actions):
        obses, goals = pi_inputs[:, :self.obs_dim], pi_inputs[:, self.obs_dim:]

        n_batch = pi_inputs.shape[0]

        sa = torch.cat([obses, actions / self.act_limit], dim=-1)  # in the autoenc_sf implementation phi is passed into psi, why?
        phi_embeds = self.phi(sa)
        w_embeds = self.xi(goals)

        return torch.bmm(phi_embeds.view(n_batch, 1, self.latent_dim), w_embeds.view(n_batch, self.latent_dim, 1)).view(n_batch).squeeze()

    def forward(self, pi_inputs, actions):
        # NOTE: assume pi_inputs to be concatenated in the order [obs, goal]
        obses, goals = pi_inputs[:, :self.obs_dim], pi_inputs[:, self.obs_dim:]

        n_batch = pi_inputs.shape[0]

        sag = torch.cat([obses, actions / self.act_limit, goals], dim=-1)  # in the autoenc_sf implementation phi is passed into psi, why?
        psi_embeds = self.psi(sag)
        w_embeds = self.xi(goals)

        return torch.bmm(psi_embeds.view(n_batch, 1, self.latent_dim), w_embeds.view(n_batch, self.latent_dim, 1)).view(n_batch).squeeze()


class SACritic(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.args = args
        self.act_limit = env_params['action_max']

        self.obs_dim = env_params['obs']
        self.act_dim = env_params['action']
        self.goal_dim = env_params['goal']

        self.latent_dim = embed_dim = args.metric_embed_dim
        self.phi = net_utils.mlp(
            [self.obs_dim + self.act_dim] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ
        )
        # self.phi = net_utils.mlp(
        #     [self.obs_dim + self.act_dim] + [embed_dim],
        #     activation=args.activ
        # )
        self.psi = net_utils.mlp(
            [self.obs_dim + self.act_dim] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ
        )
        # self.xi = net_utils.mlp(
            # [self.goal_dim] + [args.hid_size] * args.n_hids + [embed_dim],
            # activation=args.activ
        # )
        self.xi = net_utils.mlp(
            [self.obs_dim + self.goal_dim] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ
        )
    
    def forward_reward(self, pi_inputs, actions):
        # NOTE: assume pi_inputs to be concatenated in the order [obs, goal]
        obses, goals = pi_inputs[:, :self.obs_dim], pi_inputs[:, self.obs_dim:]

        n_batch = pi_inputs.shape[0]

        # q = psi(s,a)^Tw(g)
        sa = torch.cat([obses, actions / self.act_limit], dim=-1)  # in the autoenc_sf implementation phi is passed into psi, why?
        sg = torch.cat([obses, goals], dim=-1)
        phi_embeds = self.phi(sa)
        # w_embeds = self.xi(goals)
        w_embeds = self.xi(sg)

        return torch.bmm(phi_embeds.view(n_batch, 1, self.latent_dim), w_embeds.view(n_batch, self.latent_dim, 1)).view(n_batch).squeeze()

    def forward(self, pi_inputs, actions):
        # NOTE: assume pi_inputs to be concatenated in the order [obs, goal]
        obses, goals = pi_inputs[:, :self.obs_dim], pi_inputs[:, self.obs_dim:]

        n_batch = pi_inputs.shape[0]

        # q = psi(s,a)^Tw(g)
        sa = torch.cat([obses, actions / self.act_limit], dim=-1)  # in the autoenc_sf implementation phi is passed into psi, why?
        sg = torch.cat([obses, goals], dim=-1)
        psi_embeds = self.psi(sa)
        # w_embeds = self.xi(goals)
        w_embeds = self.xi(sg)

        return torch.bmm(psi_embeds.view(n_batch, 1, self.latent_dim), w_embeds.view(n_batch, self.latent_dim, 1)).view(n_batch).squeeze()


class ASGCritic(nn.Module):
    def __init__(self, env_params, args):
        super().__init__()
        self.args = args
        self.act_limit = env_params['action_max']

        self.obs_dim = env_params['obs']
        self.act_dim = env_params['action']
        self.goal_dim = env_params['goal']

        self.latent_dim = embed_dim = args.metric_embed_dim
        self.a = net_utils.mlp(
            [self.act_dim] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ
        )
        self.sg = net_utils.mlp(
            [self.obs_dim + self.goal_dim] + [args.hid_size] * args.n_hids + [embed_dim],
            activation=args.activ
        )

    def forward(self, pi_inputs, actions):
        # NOTE: assume pi_inputs to be concatenated in the order [obs, goal]
        obses, goals = pi_inputs[:, :self.obs_dim], pi_inputs[:, self.obs_dim:]

        n_batch = pi_inputs.shape[0]

        a_embeds = self.a(actions / self.act_limit)
        sg_embeds = self.sg(pi_inputs)

        return torch.bmm(a_embeds.view(n_batch, 1, self.latent_dim), sg_embeds.view(n_batch, self.latent_dim, 1)).view(n_batch).squeeze()


class BaseAgent:
    def __init__(self, env_params, args, name='agent'):
        self.env_params = env_params
        self.args = args
        self._save_file = str(name) + '.pt'

    @staticmethod
    def to_2d(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x

    def to_tensor(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if self.args.cuda:
            x = x.cuda(self.device)
        return x

    @property
    def device(self):
        return torch.device(self.args.cuda_name if self.args.cuda else "cpu")

    def get_actions(self, obs, goal):
        raise NotImplementedError

    def get_pis(self, obs, goal):
        raise NotImplementedError

    def get_qs(self, obs, goal, actions, **kwargs):
        raise NotImplementedError

    def forward(self, obs, goal, *args, **kwargs):
        """ return q_pi, pi """
        raise NotImplementedError

    def target_update(self):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError

    def save(self, path):
        if mpi_utils.is_root():
            state_dict = self.state_dict()
            save_path = osp.join(path, self._save_file)
            torch.save(state_dict, save_path)

    def load(self, path):
        load_path = osp.join(path, self._save_file)
        state_dict = torch.load(load_path)
        self.load_state_dict(state_dict)


torch.save


class Agent(BaseAgent):
    def __init__(self, env_params, args, name='agent'):
        super().__init__(env_params, args, name=name)
        
        CriticCls = {'td': Critic, 
                    'asym_metric': AsymMetricCritic,
                    'sym_metric': SymMetricCritic,
                    'fsag_metric': FSAGMetricCritic,
                    'phi_ag_metric': PhiAGMetricCritic,
                    'state_asym_metric': StateAsymMetricCritic,
                    'autoenc_sf': AutoEncSFCritic,
                    'sa_metric': SACritic,
                    'usfa_metric': USFACritic,
                    'asg_metric': ASGCritic}[args.critic_type]

        self.actor = Actor(env_params, args)
        self.critic = CriticCls(env_params, args)

        # if mpi_utils.use_mpi():
        #     mpi_utils.sync_networks(self.actor)
        #     mpi_utils.sync_networks(self.critic)

        self.actor_targ = Actor(env_params, args)
        self.critic_targ = CriticCls(env_params, args)

        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic_targ.load_state_dict(self.critic.state_dict())

        net_utils.set_requires_grad(self.actor_targ, allow_grad=False)
        net_utils.set_requires_grad(self.critic_targ, allow_grad=False)

        if self.args.cuda:
            self.cuda()

        self.o_normalizer = Normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_normalizer = Normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)

    def cuda(self):
        self.actor.cuda(self.device)
        self.critic.cuda(self.device)
        self.actor_targ.cuda(self.device)
        self.critic_targ.cuda(self.device)  

    def _preprocess_inputs(self, obs, goal):
        # add conditional here
        obs = self.to_2d(obs)
        goal = self.to_2d(goal)
        if self.args.clip_inputs:
            obs = np.clip(obs, -self.args.clip_obs, self.args.clip_obs)
            goal = np.clip(goal, -self.args.clip_obs, self.args.clip_obs)
        return obs, goal

    def _process_inputs(self, obs, goal):
        if self.args.normalize_inputs:
            obs = self.o_normalizer.normalize(obs)
            goal = self.g_normalizer.normalize(goal)
        return self.to_tensor(np.concatenate([obs, goal], axis=-1))

    def get_actions(self, obs, goal):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        with torch.no_grad():
            actions = self.actor(inputs).cpu().numpy().squeeze()
        return actions

    def get_pis(self, obs, goal):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        pis = self.actor(inputs)
        return pis

    def get_qs(self, obs, goal, actions, **kwargs):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        actions = self.to_tensor(actions)
        return self.critic(inputs, actions, **kwargs)

    def forward(self, obs, goal, q_target=False, pi_target=False):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        q_net = self.critic_targ if q_target else self.critic
        a_net = self.actor_targ if pi_target else self.actor
        pis = a_net(inputs)
        return q_net(inputs, pis), pis
        
    def target_update(self):
        net_utils.target_soft_update(source=self.actor, target=self.actor_targ, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic, target=self.critic_targ, polyak=self.args.polyak)

    def normalizer_update(self, obs, goal):
        obs, goal = self._preprocess_inputs(obs, goal)
        self.o_normalizer.update(obs)
        self.g_normalizer.update(goal)
        self.o_normalizer.recompute_stats()
        self.g_normalizer.recompute_stats()

    def state_dict(self):
        return {'actor': self.actor.state_dict(),
                'actor_targ': self.actor_targ.state_dict(),
                'critic': self.critic.state_dict(),
                'critic_targ': self.critic_targ.state_dict(),
                'o_normalizer': self.o_normalizer.state_dict(),
                'g_normalizer': self.g_normalizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_targ.load_state_dict(state_dict['actor_targ'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_targ.load_state_dict(state_dict['critic_targ'])
        self.o_normalizer.load_state_dict(state_dict['o_normalizer'])
        self.g_normalizer.load_state_dict(state_dict['g_normalizer'])

class TD3Agent(Agent):
    def __init__(self, env_params, args, name='agent'):
        super().__init__(env_params, args, name=name)
        
        CriticCls = {'td': Critic, 
                    'asym_metric': AsymMetricCritic,
                    'sym_metric': SymMetricCritic,
                    'fsag_metric': FSAGMetricCritic,
                    'state_asym_metric': StateAsymMetricCritic}[args.critic_type]

        self.actor = Actor(env_params, args)
        self.critic = CriticCls(env_params, args)
        self.critic2 = CriticCls(env_params, args)

        # if mpi_utils.use_mpi():
        #     mpi_utils.sync_networks(self.actor)
        #     mpi_utils.sync_networks(self.critic)

        self.actor_targ = Actor(env_params, args)
        self.critic_targ = CriticCls(env_params, args)
        self.critic2_targ = CriticCls(env_params, args)

        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic_targ.load_state_dict(self.critic.state_dict())
        self.critic2_targ.load_state_dict(self.critic2.state_dict())

        net_utils.set_requires_grad(self.actor_targ, allow_grad=False)
        net_utils.set_requires_grad(self.critic_targ, allow_grad=False)
        net_utils.set_requires_grad(self.critic2_targ, allow_grad=False)

        if self.args.cuda:
            self.cuda()

        self.o_normalizer = Normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_normalizer = Normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)

    def cuda(self):
        self.actor.cuda(self.device)
        self.critic.cuda(self.device)
        self.critic2.cuda(self.device)
        self.actor_targ.cuda(self.device)
        self.critic_targ.cuda(self.device)
        self.critic2_targ.cuda(self.device)

    def get_qs(self, obs, goal, actions, critic_id):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        actions = self.to_tensor(actions)
        if critic_id == 0:
            return self.critic(inputs, actions)
        elif critic_id == 1:
            return self.critic2(inputs, actions)
        else:
            raise NotImplemented()

    def forward(self, obs, goal, critic_id, q_target=False, pi_target=False, smooth=False):       
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        if critic_id == 0:
            q_net = self.critic_targ if q_target else self.critic
        elif critic_id == 1:
            q_net = self.critic2_targ if q_target else self.critic2
        else:
            raise NotImplemented()
        a_net = self.actor_targ if pi_target else self.actor
        pis = a_net(inputs)
        if smooth and self.args.smooth_targ_policy:
            pis = target_policy_smooth(pis)
        return q_net(inputs, pis), pis
        
    def target_update(self):
        net_utils.target_soft_update(source=self.actor, target=self.actor_targ, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic, target=self.critic_targ, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic2, target=self.critic2_targ, polyak=self.args.polyak)

    def state_dict(self):
        return {'actor': self.actor.state_dict(),
                'actor_targ': self.actor_targ.state_dict(),
                'critic': self.critic.state_dict(),
                'critic2': self.critic2.state_dict(),
                'critic_targ': self.critic_targ.state_dict(),
                'critic2_targ': self.critic2_targ.state_dict(),
                'o_normalizer': self.o_normalizer.state_dict(),
                'g_normalizer': self.g_normalizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_targ.load_state_dict(state_dict['actor_targ'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_targ.load_state_dict(state_dict['critic_targ'])
        self.critic2.load_state_dict(state_dict['critic2'])
        self.critic2_targ.load_state_dict(state_dict['critic2_targ'])
        self.o_normalizer.load_state_dict(state_dict['o_normalizer'])
        self.g_normalizer.load_state_dict(state_dict['g_normalizer'])


class SACAgent(Agent):
    def __init__(self, env_params, args, name='agent'):
        super().__init__(env_params, args, name=name)

        self.min_action = (
            np.zeros(env_params['action_space'].shape, dtype=env_params['action_space'].dtype) + -1
        )
        self.max_action = (
            np.zeros(env_params['action_space'].shape, dtype=env_params['action_space'].dtype) + 1
        )        
        
        CriticCls = {'td': SACCritic, 
                    }[args.critic_type]

        self.actor = SACPolicy(env_params, args)
        self.critic = CriticCls(env_params, args)
        self.critic2 = CriticCls(env_params, args)

        # if mpi_utils.use_mpi():
        #     mpi_utils.sync_networks(self.actor)
        #     mpi_utils.sync_networks(self.critic)

        self.critic_targ = CriticCls(env_params, args)
        self.critic2_targ = CriticCls(env_params, args)

        self.critic_targ.load_state_dict(self.critic.state_dict())
        self.critic2_targ.load_state_dict(self.critic2.state_dict())

        net_utils.set_requires_grad(self.critic_targ, allow_grad=False)
        net_utils.set_requires_grad(self.critic2_targ, allow_grad=False)

        self.target_entropy = -env_params['action']
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()        

        if self.args.cuda:
            self.cuda()

        self.o_normalizer = Normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_normalizer = Normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        
    def cuda(self):
        self.actor.cuda(self.device)
        self.critic.cuda(self.device)
        self.critic2.cuda(self.device)
        self.critic_targ.cuda(self.device)
        self.critic2_targ.cuda(self.device)
        self.log_alpha.cuda(self.device)

    def get_qs(self, obs, goal, actions, critic_id):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        actions = self.to_tensor(actions)
        if critic_id == 0:
            return self.critic(inputs, actions)
        elif critic_id == 1:
            return self.critic2(inputs, actions)
        else:
            raise NotImplemented()

    def _rescale_action(self, action):
        assert np.all(np.greater_equal(action, self.min_action)), (
            action,
            self.min_action,
        )
        assert np.all(np.less_equal(action, self.max_action)), (action, self.max_action)
        low = self.min_action
        high = self.max_action
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = np.clip(action, low, high)
        return action

    def get_actions(self, obs, goal, deterministic=False):
        # with torch.no_grad():
        #     action, _, mean = self.policy(torch.Tensor(state).view(1,-1).to(device))
        # if deterministic:
        #     return mean.squeeze().cpu().numpy()
        # return np.atleast_1d(action.squeeze().cpu().numpy())

        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
        with torch.no_grad():
            actions, _, mean = self.actor(inputs)

        return self._rescale_action(mean.squeeze().cpu().numpy().squeeze()) if deterministic else self._rescale_action(np.atleast_1d(actions.cpu().numpy().squeeze()))

    def get_pis(self, obs, goal):
        raise NotImplemented()

    def forward(self, obs, goal, q_target=False, requires_act_grad=True):
        obs, goal = self._preprocess_inputs(obs, goal)
        inputs = self._process_inputs(obs, goal)
    
        q_net1 = self.critic_targ if q_target else self.critic
        q_net2 = self.critic2_targ if q_target else self.critic2

        action_batch, logprob_batch, _ = self.actor(inputs, get_logprob=True, requires_act_grad=requires_act_grad)       

        return q_net1(inputs, action_batch), q_net2(inputs, action_batch), logprob_batch
        
    def target_update(self):
        net_utils.target_soft_update(source=self.critic, target=self.critic_targ, polyak=self.args.polyak)
        net_utils.target_soft_update(source=self.critic2, target=self.critic2_targ, polyak=self.args.polyak)

    def state_dict(self):
        return {'actor': self.actor.state_dict(),                
                'critic': self.critic.state_dict(),
                'critic2': self.critic2.state_dict(),
                'critic_targ': self.critic_targ.state_dict(),
                'critic2_targ': self.critic2_targ.state_dict(),
                'o_normalizer': self.o_normalizer.state_dict(),
                'g_normalizer': self.g_normalizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])    
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_targ.load_state_dict(state_dict['critic_targ'])
        self.critic2.load_state_dict(state_dict['critic2'])
        self.critic2_targ.load_state_dict(state_dict['critic2_targ'])
        self.o_normalizer.load_state_dict(state_dict['o_normalizer'])
        self.g_normalizer.load_state_dict(state_dict['g_normalizer'])