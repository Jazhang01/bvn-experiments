from rl.agent import Critic
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

import os.path as osp
from rl.utils import mpi_utils, net_utils


def to_numpy(x):
    return x.detach().float().cpu().numpy()


def dict_to_numpy(tensor_dict):
    return {
        k: to_numpy(v) for k, v in tensor_dict.items()
    }

class Learner:
    def __init__(
            self,
            agent,
            args,
            name='learner',
    ):
        self.agent = agent
        from ml_logger import logger
        self.logger = logger
        self.args = args

        self.optim_q = Adam(agent.critic.parameters(), lr=args.lr_critic)
        self.optim_pi = Adam(agent.actor.parameters(), lr=args.lr_actor)

        self._save_file = str(name) + '.pt'      

    def usfa_critic_loss_old(self, batch):
        ### get items from batch
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())
        r_ag_ag2 = self.agent.to_tensor(batch['r_ag_ag2'].flatten())
        r_future_ag = self.agent.to_tensor(batch['r_future_ag'].flatten())

        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())
        ### get items from batch

        ### calculate the predicted reward as phi^Tw
        pi_inputs = self.agent._process_inputs(*self.agent._preprocess_inputs(o, bg))
        action = self.agent.to_tensor(a)
        pred_r = self.agent.critic.forward_reward(pi_inputs, action)
        ### calculate the predicted reward as phi^Tw

        ### calculate q target
        q_next, _ = self.agent.forward(o2, bg, q_target=True, pi_target=True)
        if self.args.update_with_predict_r:
            q_targ = pred_r + self.args.gamma * q_next.detach()
        else:
            q_targ = r + self.args.gamma * q_next.detach()
        q_targ = torch.clamp(q_targ, -self.args.clip_return, 0.0)
        ### calculate q target

        ### experiment: sf loss
        assert len(o.shape) == 2
        n_batch, obs_dim = o.shape

        obs_goal = self.agent._process_inputs(*self.agent._preprocess_inputs(o, bg))
        nextobs_goal = self.agent._process_inputs(*self.agent._preprocess_inputs(o2, bg))

        obs = obs_goal[:, :obs_dim]
        next_obs = nextobs_goal[:, :obs_dim]
        goal = obs_goal[:, obs_dim:]
        
        next_action = self.agent.actor_targ(nextobs_goal)
        
        sa = torch.cat([obs, action / self.agent.critic.act_limit], dim=-1)
        sag = torch.cat([obs, action / self.agent.critic.act_limit, goal], dim=-1)
        sag2 = torch.cat([next_obs, next_action / self.agent.critic_targ.act_limit, goal], dim=-1)

        psi = self.agent.critic.psi(sag)
        phi = self.agent.critic_targ.phi(sa)
        next_psi = self.agent.critic_targ.psi(sag2)
        target_psi = phi + self.args.gamma * next_psi
        
        loss_sf = F.mse_loss(psi, target_psi.detach())
        ### experiment: sf loss

        ### calculate the critic loss
        q_bg = self.agent.get_qs(o, bg, a)
        loss_q = (q_bg - q_targ).pow(2).mean()
        loss_r = (pred_r - r).pow(2).mean()
        loss_critic = loss_q + loss_r
        ### calculate the critic loss

        ### what is this, i don't know
        q_ag2 = self.agent.get_qs(o, ag2, a) # not sure what this is
        loss_ag2 = q_ag2.pow(2).mean() # not sure what this is

        q_future = self.agent.get_qs(o, future_ag, a) # not sure what this is
        ### what is this, i don't know

        with torch.no_grad():
            self.logger.store(loss_q=loss_q.item(),
                            loss_ag2=loss_ag2.item(),
                            loss_critic=loss_critic.item(),
                            q_targ=q_targ.mean().item(),
                            q_bg=q_bg.mean().item(),
                            q_ag2=q_ag2.mean().item(),
                            q_future=q_future.mean().item(),
                            offset=offset.mean().item(),
                            loss_r=loss_r.mean().item(),
                            loss_sf=loss_sf,
                            r=r, r_ag_ag2=r_ag_ag2, r_future_ag=r_future_ag)

        return loss_critic

    def sa_critic_loss_old(self, batch):
        ### get items from batch
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        r_ag_ag2, r_future_ag = batch['r_ag_ag2'], batch['r_future_ag']

        # bg_extra = np.random.permutation(batch['bg'])
        # bg_extra = torch.distributions.normal.Normal(self.agent.to_tensor(bg),
        #         self.agent.to_tensor([1])).sample().cpu().numpy()

        r = self.agent.to_tensor(r.flatten())
        r_ag_ag2 = self.agent.to_tensor(r_ag_ag2.flatten())
        r_future_ag = self.agent.to_tensor(r_future_ag.flatten())
        offset = self.agent.to_tensor(offset.flatten())
        ### get items from batch

        ### calculate the predicted reward
        action = self.agent.to_tensor(a)
        pi_inputs = self.agent._process_inputs(*self.agent._preprocess_inputs(o, bg))
        pred_r_bg = self.agent.critic.forward_reward(pi_inputs, action)
        ### calculate the predicted reward

        ### EXTRA: calculate the predicted reward as phi^Tw
        # pi_inputs_extra = self.agent._process_inputs(*self.agent._preprocess_inputs(o, bg_extra))
        # pred_r_extra = self.agent.critic.forward_reward(pi_inputs_extra, action)
        ### EXTRA: calculate the predicted reward as phi^Tw

        ### calculate q target
        q_next_bg, _ = self.agent.forward(o2, bg, q_target=True, pi_target=True)
        if self.args.update_with_predict_r:
            q_targ_bg = pred_r_bg + self.args.gamma * q_next_bg.detach()
        else:
            q_targ_bg = r + self.args.gamma * q_next_bg.detach()
        q_targ_bg = torch.clamp(q_targ_bg, -self.args.clip_return, 0.0)
        ### calculate q target

        ### EXTRA: calculate q target
        # q_next_extra, _ = self.agent.forward(o2, bg_extra, q_target=True, pi_target=True)
        # q_targ_extra = pred_r_extra.detach() + self.args.gamma * q_next_extra.detach()
        # q_targ_extra = torch.clamp(q_targ_extra, -self.args.clip_return, 0.0)
        ### EXTRA: calculate q target

        ### experiment: sf loss
        assert len(o.shape) == 2
        n_batch, obs_dim = o.shape

        obs_goal = self.agent._process_inputs(*self.agent._preprocess_inputs(o, bg))
        nextobs_goal = self.agent._process_inputs(*self.agent._preprocess_inputs(o2, bg))

        obs = obs_goal[:, :obs_dim]
        next_obs = nextobs_goal[:, :obs_dim]
        goal = obs_goal[:, obs_dim:]
        
        next_action = self.agent.actor_targ(nextobs_goal)

        sa = torch.cat([obs, action / self.agent.critic.act_limit], dim=-1)
        sa2 = torch.cat([next_obs, next_action / self.agent.critic_targ.act_limit], dim=-1)

        psi = self.agent.critic.psi(sa)
        phi = self.agent.critic_targ.phi(sa)
        next_psi = self.agent.critic_targ.psi(sa2)
        target_psi = phi + self.args.gamma * next_psi
        
        loss_sf = F.mse_loss(psi, target_psi.detach())
        ### experiment: sf loss

        ### calculate the critic loss
        q_bg = self.agent.get_qs(o, bg, a)
        loss_q = (q_bg - q_targ_bg).pow(2).mean()
        loss_r = (pred_r_bg - r).pow(2).mean()
        loss_critic = loss_q + loss_r
        ### calculate the critic loss

        ### EXTRA: critic loss
        # q_bg_extra = self.agent.get_qs(o, bg_extra, a)

        # q_bg = torch.cat([q_bg, q_bg_extra])
        # q_targ = torch.cat([q_targ, q_targ_extra])
        ### EXTRA: critic loss

        ### extra information
        q_ag2 = self.agent.get_qs(o, ag2, a)
        loss_ag2 = q_ag2.pow(2).mean()

        q_future = self.agent.get_qs(o, future_ag, a)
        ### extra information

        with torch.no_grad():
            self.logger.store(loss_q=loss_q.item(),
                            loss_ag2=loss_ag2.item(),
                            loss_critic=loss_critic.item(),
                            q_targ=q_targ_bg.mean().item(),
                            q_bg=q_bg.mean().item(),
                            q_ag2=q_ag2.mean().item(),
                            q_future=q_future.mean().item(),
                            offset=offset.mean().item(),
                            loss_r=loss_r.mean().item(),
                            loss_sf=loss_sf,
                            r=r, r_ag_ag2=r_ag_ag2, r_future_ag=r_future_ag)

        return loss_critic

    def calc_loss_q(self, o, a, o2, g, r=None, predict_reward=False, backprop_predict_r=False):
        assert predict_reward or r is not None
        assert not (backprop_predict_r and not predict_reward)

        if predict_reward:
            action = self.agent.to_tensor(a)
            pi_inputs = self.agent._process_inputs(*self.agent._preprocess_inputs(o, g))
            r = self.agent.critic.forward_reward(pi_inputs, action)
        
        # q target
        q_next, _ = self.agent.forward(o2, g, q_target=True, pi_target=True)
        if backprop_predict_r:
            q_targ = r + self.args.gamma * q_next.detach()
        else:
            q_targ = r.detach() + self.args.gamma * q_next.detach()
        q_targ = torch.clamp(q_targ, -self.args.clip_return, 0.0)

        # q loss
        q = self.agent.get_qs(o, g, a)
        loss_q = (q - q_targ).pow(2).mean()

        return loss_q
    
    def calc_loss_r(self, o, a, g, r):
        action = self.agent.to_tensor(a)
        pi_inputs = self.agent._process_inputs(*self.agent._preprocess_inputs(o, g))
        pred_r = self.agent.critic.forward_reward(pi_inputs, action)

        loss_r = (pred_r - r).pow(2).mean()
        
        return loss_r

    def usfa_critic_loss(self, batch):
        ### get items from batch
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())
        r_ag_ag2 = self.agent.to_tensor(batch['r_ag_ag2'].flatten())
        r_future_ag = self.agent.to_tensor(batch['r_future_ag'].flatten())

        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())
        ### get items from batch

        loss_q_bg = self.calc_loss_q(o, a, o2, bg, r, predict_reward=self.args.update_with_predict_r,
                                                      backprop_predict_r=self.args.update_with_predict_r)
        loss_r = self.calc_loss_r(o, a, bg, r)

        loss_critic = loss_q_bg + loss_r
        
        with torch.no_grad():
            self.logger.store(loss_q=loss_q_bg.item(),
                            # loss_q_future_ag=loss_q_future_ag.item(),
                            loss_critic=loss_critic.item(),
                            offset=offset.mean().item(),
                            loss_r=loss_r.mean().item(),
                            r=r, r_ag_ag2=r_ag_ag2, r_future_ag=r_future_ag)

        return loss_critic

    
    def sa_critic_loss(self, batch):
        ### get items from batch
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        r_ag_ag2, r_future_ag = batch['r_ag_ag2'], batch['r_future_ag']

        r = self.agent.to_tensor(r.flatten())
        r_ag_ag2 = self.agent.to_tensor(r_ag_ag2.flatten())
        r_future_ag = self.agent.to_tensor(r_future_ag.flatten())
        offset = self.agent.to_tensor(offset.flatten())
        ### get items from batch

        ### extra goals
        bg_permute = np.random.permutation(batch['bg'])
        bg_sample = torch.distributions.normal.Normal(self.agent.to_tensor(bg),
                self.agent.to_tensor([1])).sample().cpu().numpy()
        ### extra goals

        loss_q_bg = self.calc_loss_q(o, a, o2, bg, r, predict_reward=self.args.update_with_predict_r, 
                                                      backprop_predict_r=self.args.update_with_predict_r)
        loss_q_future_ag = self.calc_loss_q(o, a, o2, future_ag, r=None, predict_reward=True, backprop_predict_r=False)
        loss_r = self.calc_loss_r(o, a, bg, r)

        loss_critic = loss_q_bg + loss_r
        # loss_critic = loss_q_bg + loss_q_future_ag + loss_r

        with torch.no_grad():
            self.logger.store(loss_q=loss_q_bg.item(),
                            loss_q_future_ag=loss_q_future_ag.item(),
                            loss_critic=loss_critic.item(),
                            offset=offset.mean().item(),
                            loss_r=loss_r.mean().item(),
                            r=r, r_ag_ag2=r_ag_ag2, r_future_ag=r_future_ag)

        return loss_critic


    def critic_loss(self, batch):
        if self.args.critic_type == 'sa_metric':
            return self.sa_critic_loss(batch)
        if self.args.critic_type == 'usfa_metric':
            return self.usfa_critic_loss(batch)

        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())
        r_ag_ag2 = self.agent.to_tensor(batch['r_ag_ag2'].flatten())
        r_future_ag = self.agent.to_tensor(batch['r_future_ag'].flatten())

        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())

        with torch.no_grad():
            q_next, _ = self.agent.forward(o2, bg, q_target=True, pi_target=True)
            q_targ = r + self.args.gamma * q_next
            q_targ = torch.clamp(q_targ, -self.args.clip_return, 0.0)

        q_bg = self.agent.get_qs(o, bg, a)
        loss_q = (q_bg - q_targ).pow(2).mean()

        q_ag2 = self.agent.get_qs(o, ag2, a)
        loss_ag2 = q_ag2.pow(2).mean()

        q_future = self.agent.get_qs(o, future_ag, a)

        loss_critic = loss_q

        extra_logs = {}
        if self.args.critic_type == 'autoenc_sf':
            n_batch = o.shape[0]            
            obs_goal = self.agent._process_inputs(*self.agent._preprocess_inputs(o, bg))
            obs = obs_goal[:, :o.shape[-1]]
            
            phi_obs = self.agent.critic.phi(obs)            
            loss_phi = (self.agent.critic.dec_phi(phi_obs) - obs).pow(2).mean()
            
            latent_dim = phi_obs.shape[-1]
            wsg = self.agent.critic.wsg(obs_goal)
            phi_dot_wsg = torch.bmm(phi_obs.view(n_batch, 1, latent_dim), wsg.view(n_batch, latent_dim, 1)).view(n_batch).squeeze()
            loss_wsg = (phi_dot_wsg - r).pow(2).mean()
            loss_critic += (loss_phi + loss_wsg)
            extra_logs['loss_phi'] = loss_phi.mean().item()
            extra_logs['loss_wsg'] = loss_wsg.mean().item()

        with torch.no_grad():
            self.logger.store(loss_q=loss_q.item(),
                            loss_ag2=loss_ag2.item(),
                            loss_critic=loss_critic.item(),
                            q_targ=q_targ.mean().item(),
                            q_bg=q_bg.mean().item(),
                            q_ag2=q_ag2.mean().item(),
                            q_future=q_future.mean().item(),
                            offset=offset.mean().item(),
                            **extra_logs,
                            r=r, r_ag_ag2=r_ag_ag2, r_future_ag=r_future_ag)

        return loss_critic

    optim_gred = None

    def gred_update(self, observation, goal_next, goal_far):
        loss_gred = self.gred_loss(observation, goal_next, goal_far)
        # self.optim_pi.zero_grad()
        # loss_gred.backward()
        # self.optim_pi.step()

        self.optim_gred = self.optim_gred or Adam(self.agent.actor.parameters(), lr=self.args.lr_actor)

        self.optim_gred.zero_grad()
        loss_gred.backward()
        self.optim_gred.step()

        self.logger.store(loss_gred=loss_gred.item(), )

    def gred_loss(self, observation, goal_next, goal_far, action=None):
        """Implement the loss here.
        todo: remove actions (to use online relabeling)
        """
        goal_far = self.agent.to_tensor(goal_far)
        o = self.agent.to_tensor(observation)
        if action is None:
            with torch.no_grad():
                a = self.agent.get_pis(o, goal_next)
        else:
            a = self.agent.to_tensor(action)

        # HER has a behavior cloning term. We further supplant
        #   with goals that are farther.
        pi = self.agent.get_pis(o, goal_far)
        # loss_bc = (pi_future - a).pow(2).mean()
        return F.smooth_l1_loss(pi, a, )

    def actor_loss(self, batch):
        o, a, bg = batch['ob'], batch['a'], batch['bg']
        ag, ag2, future_ag = batch['ag'], batch['ag2'], batch['future_ag']

        a = self.agent.to_tensor(a)

        q_pi, pi = self.agent.forward(o, bg)
        action_l2 = (pi / self.agent.actor.act_limit).pow(2).mean()
        loss_actor = (- q_pi).mean() + self.args.action_l2 * action_l2

        # loss_actor = (- q_pi).mean()

        # pi_future = self.agent.get_pis(o, future_ag)
        # loss_bc = (pi_future - a).abs().mean()

        with torch.no_grad():
            self.logger.store(loss_actor=loss_actor.item(),
                              action_l2=action_l2.item(),
                            #   loss_bc=loss_bc.item(),
                              q_pi=q_pi.mean().cpu().numpy(), )

        return loss_actor

    def update(self, batch):
        loss_critic = self.critic_loss(batch)
        self.optim_q.zero_grad()
        loss_critic.backward()
        # if mpi_utils.use_mpi():
        #     mpi_utils.sync_grads(self.agent.critic, scale_grad_by_procs=True)
        self.optim_q.step()

        for i in range(self.args.n_actor_optim_steps):
            loss_actor = self.actor_loss(batch)
            self.optim_pi.zero_grad()
            loss_actor.backward()
            # if mpi_utils.use_mpi():
            #     mpi_utils.sync_grads(self.agent.actor, scale_grad_by_procs=True)
            self.optim_pi.step()

    def target_update(self):
        self.agent.target_update()

    def state_dict(self):
        return dict(
            q_optim=self.optim_q.state_dict(),
            pi_optim=self.optim_pi.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.optim_q.load_state_dict(state_dict['q_optim'])
        self.optim_pi.load_state_dict(state_dict['pi_optim'])

    def save(self, path):
        if mpi_utils.is_root():
            state_dict = self.state_dict()
            save_path = osp.join(path, self._save_file)
            torch.save(state_dict, save_path)

    def load(self, path):
        load_path = osp.join(path, self._save_file)
        state_dict = torch.load(load_path)
        self.load_state_dict(state_dict)

class TD3Learner(Learner):
    def __init__(
            self,
            agent,
            args,
            name='learner',
    ):
        super().__init__(agent, args, name)
        self.optim_q2 = Adam(agent.critic2.parameters(), lr=args.lr_critic)

    def critic_loss(self, batch):
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())
        r_ag_ag2 = self.agent.to_tensor(batch['r_ag_ag2'].flatten())
        r_future_ag = self.agent.to_tensor(batch['r_future_ag'].flatten())

        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())

        with torch.no_grad():
            q1_next, _ = self.agent.forward(o2, bg, critic_id=0, q_target=True, pi_target=True, smooth=True)
            q1_targ = r + self.args.gamma * q1_next
            q1_targ = torch.clamp(q1_targ, -self.args.clip_return, 0.0)

            q2_next, _ = self.agent.forward(o2, bg, critic_id=1, q_target=True, pi_target=True, smooth=True)
            q2_targ = r + self.args.gamma * q2_next
            q2_targ = torch.clamp(q2_targ, -self.args.clip_return, 0.0)

            q_targ = torch.min(q1_targ, q2_targ)

        q1_bg = self.agent.get_qs(o, bg, a, critic_id=0)
        # NOTE: both q function uses the same target
        loss_q1 = (q1_bg - q_targ).pow(2).mean()
        q2_bg = self.agent.get_qs(o, bg, a, critic_id=1)
        loss_q2 = (q2_bg - q_targ).pow(2).mean()

        q1_ag2 = self.agent.get_qs(o, ag2, a, critic_id=0)
        loss_q1_ag2 = q1_ag2.pow(2).mean()
        q2_ag2 = self.agent.get_qs(o, ag2, a, critic_id=1)
        loss_q2_ag2 = q2_ag2.pow(2).mean()

        q1_future = self.agent.get_qs(o, future_ag, a, critic_id=0)
        q2_future = self.agent.get_qs(o, future_ag, a, critic_id=1)

        loss_critic1 = loss_q1
        loss_critic2 = loss_q2

        with torch.no_grad():
            self.logger.store(loss_q=loss_q1.item(),
                            loss_q1_ag2=loss_q1_ag2.item(),
                            loss_q2_ag2=loss_q2_ag2.item(),
                            loss_critic1=loss_critic1.item(),
                            loss_critic2=loss_critic2.item(),
                            q_targ=q_targ.mean().item(),
                            q1_bg=q2_bg.mean().item(),
                            q2_bg=q2_bg.mean().item(),
                            q1_ag2=q1_ag2.mean().item(),
                            q2_ag2=q2_ag2.mean().item(),
                            q1_future=q1_future.mean().item(),
                            q2_future=q2_future.mean().item(),
                            offset=offset.mean().item(),
                            r=r, r_ag_ag2=r_ag_ag2, r_future_ag=r_future_ag)

        return loss_critic1, loss_critic2

    optim_gred = None

    def actor_loss(self, batch):        
        o, a, bg = batch['ob'], batch['a'], batch['bg']
        ag, ag2, future_ag = batch['ag'], batch['ag2'], batch['future_ag']

        a = self.agent.to_tensor(a)

        q_pi, pi = self.agent.forward(o, bg, critic_id=0)
        action_l2 = (pi / self.agent.actor.act_limit).pow(2).mean()
        loss_actor = (- q_pi).mean() + self.args.action_l2 * action_l2

        # loss_actor = (- q_pi).mean()

        # pi_future = self.agent.get_pis(o, future_ag)
        # loss_bc = (pi_future - a).abs().mean()

        with torch.no_grad():
            self.logger.store(loss_actor=loss_actor.item(),
                            action_l2=action_l2.item(),
                            # loss_bc=loss_bc.item(),
                            q_pi=q_pi.mean().cpu().numpy(), )

        return loss_actor

    def update(self, batch):
        loss_critic1, loss_critic2 = self.critic_loss(batch)
        self.optim_q.zero_grad()
        loss_critic1.backward()
        # if mpi_utils.use_mpi():
        #     mpi_utils.sync_grads(self.agent.critic, scale_grad_by_procs=True)
        self.optim_q.step()

        self.optim_q2.zero_grad()
        loss_critic2.backward()
        # if mpi_utils.use_mpi():
        #     mpi_utils.sync_grads(self.agent.critic, scale_grad_by_procs=True)
        self.optim_q2.step()

        for i in range(self.args.n_actor_optim_steps):
            loss_actor = self.actor_loss(batch)
            self.optim_pi.zero_grad()
            loss_actor.backward()
            # if mpi_utils.use_mpi():
            #     mpi_utils.sync_grads(self.agent.actor, scale_grad_by_procs=True)
            self.optim_pi.step()

    def target_update(self):
        self.agent.target_update()

    def state_dict(self):
        return dict(
            q_optim=self.optim_q.state_dict(),
            pi_optim=self.optim_pi.state_dict(),
            q2_optim=self.optim_q2.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.optim_q.load_state_dict(state_dict['q_optim'])
        self.optim_pi.load_state_dict(state_dict['pi_optim'])
        self.optim_q2.load_state_dict(state_dict['q2_optim'])


class SACLearner(Learner):
    def __init__(
            self,
            agent,
            args,
            name='learner',
    ):
        super().__init__(agent, args, name)
        self.optim_q2 = Adam(agent.critic2.parameters(), lr=args.lr_critic)
        self.temp_optimizer = torch.optim.Adam([self.agent.log_alpha], lr=args.temperature_optimizer_lr)

    def critic_loss(self, batch):
        o, a, o2, r, bg = batch['ob'], batch['a'], batch['o2'], batch['r'], batch['bg']
        r = self.agent.to_tensor(r.flatten())
        r_ag_ag2 = self.agent.to_tensor(batch['r_ag_ag2'].flatten())
        r_future_ag = self.agent.to_tensor(batch['r_future_ag'].flatten())

        ag, ag2, future_ag, offset = batch['ag'], batch['ag2'], batch['future_ag'], batch['offset']
        offset = self.agent.to_tensor(offset.flatten())
        
        with torch.no_grad():
            q1_next, q2_next, logprob_next = self.agent.forward(o2, bg, q_target=True, requires_act_grad=False) 
            q_next = torch.min(q1_next, q2_next)
            q_targ = torch.clamp(r + self.args.gamma * (q_next - self.agent.alpha * logprob_next), -self.args.clip_return, 0.0)            

        q1_bg = self.agent.get_qs(o, bg, a, critic_id=0)
        loss_q1 = 0.5 * (q1_bg - q_targ).pow(2).mean()
        
        q2_bg = self.agent.get_qs(o, bg, a, critic_id=1)
        loss_q2 = 0.5 * (q2_bg - q_targ).pow(2).mean()
        # import ipdb; ipdb.set_trace()
        # q1_ag2 = self.agent.get_qs(o, ag2, a, critic_id=0)
        # loss_q1_ag2 = q1_ag2.pow(2).mean()
        
        # q2_ag2 = self.agent.get_qs(o, ag2, a, critic_id=1)
        # loss_q2_ag2 = q2_ag2.pow(2).mean()

        # q1_future = self.agent.get_qs(o, future_ag, a, critic_id=0)
        # q2_future = self.agent.get_qs(o, future_ag, a, critic_id=1)

        loss_critic1 = loss_q1
        loss_critic2 = loss_q2

        with torch.no_grad():
            self.logger.store(loss_q=loss_q1.item(),
                            # loss_q1_ag2=loss_q1_ag2.item(),
                            # loss_q2_ag2=loss_q2_ag2.item(),
                            loss_critic1=loss_critic1.item(),
                            loss_critic2=loss_critic2.item(),
                            q_targ=q_targ.mean().item(),
                            q1_bg=q2_bg.mean().item(),
                            q2_bg=q2_bg.mean().item(),
                            # q1_ag2=q1_ag2.mean().item(),
                            # q2_ag2=q2_ag2.mean().item(),
                            # q1_future=q1_future.mean().item(),
                            # q2_future=q2_future.mean().item(),
                            offset=offset.mean().item(),
                            r=r, r_ag_ag2=r_ag_ag2, r_future_ag=r_future_ag)

        return loss_critic1, loss_critic2

    optim_gred = None

    def actor_loss(self, batch):        
        o, a, bg = batch['ob'], batch['a'], batch['bg']
        ag, ag2, future_ag = batch['ag'], batch['ag2'], batch['future_ag']

        a = self.agent.to_tensor(a)
        
        q1, q2, logprob = self.agent.forward(o, bg, q_target=False) 
        q = torch.min(q1, q2)
        ent = self.agent.alpha * logprob
        loss_actor = torch.mean(ent - q)
        temp_loss = -self.agent.log_alpha * (logprob.detach() + self.agent.target_entropy).mean()

        with torch.no_grad():
            self.logger.store(loss_actor=loss_actor.item(),
                            # action_l2=action_l2.item(),
                            # loss_bc=loss_bc.item(),
                            entropy_term=ent.detach.cpu().numpy(),
                            temperature=self.agent.alpha.item(),
                            q_pi=q.mean().cpu().numpy(),)
    
        return loss_actor, temp_loss

    def update(self, batch):
        
        loss_critic1, loss_critic2 = self.critic_loss(batch)
        self.optim_q.zero_grad()
        loss_critic1.backward()
        # if mpi_utils.use_mpi():
        #     mpi_utils.sync_grads(self.agent.critic, scale_grad_by_procs=True)
        self.optim_q.step()

        self.optim_q2.zero_grad()
        loss_critic2.backward()
        # if mpi_utils.use_mpi():
        #     mpi_utils.sync_grads(self.agent.critic, scale_grad_by_procs=True)
        self.optim_q2.step()

        net_utils.set_requires_grad(self.agent.critic, False)
        net_utils.set_requires_grad(self.agent.critic2, False)

        loss_actor, loss_temp = self.actor_loss(batch)
        self.optim_pi.zero_grad()
        loss_actor.backward()
        # if mpi_utils.use_mpi():
        #     mpi_utils.sync_grads(self.agent.actor, scale_grad_by_procs=True)
        self.optim_pi.step()

        net_utils.set_requires_grad(self.agent.critic, True)
        net_utils.set_requires_grad(self.agent.critic2, True)

        self.temp_optimizer.zero_grad()
        loss_temp.backward()
        self.temp_optimizer.step()

        self.agent.alpha = self.agent.log_alpha.exp()


    def target_update(self):
        self.agent.target_update()

    def state_dict(self):
        return dict(
            q_optim=self.optim_q.state_dict(),
            pi_optim=self.optim_pi.state_dict(),
            q2_optim=self.optim_q2.state_dict(),
            temp_optim=self.temp_optimizer.state_dict()
        )

    def load_state_dict(self, state_dict):
        self.optim_q.load_state_dict(state_dict['q_optim'])
        self.optim_pi.load_state_dict(state_dict['pi_optim'])
        self.optim_q2.load_state_dict(state_dict['q2_optim'])
        self.temp_optimizer.load_state_dict(state_dict['temp_optim'])
