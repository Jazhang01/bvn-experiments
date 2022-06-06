import os.path as osp
from networkx.generators import stochastic

import numpy as np
import sys, os
import torch

from rl import Args, LfGR
from rl.algo.mega import simple_min_kernel_density_model, OMEGA
from rl.algo.object_relabel import relabel
from rl.replays.her import sample_her_transitions
from rl.replays.lfgr import visualize_graph, GraphicalReplay
from rl.utils import mpi_utils
from rl.utils.torch_utils import mode_of_distribution


class BaseAlgo:
    lfgr: GraphicalReplay = None

    def __init__(self, *, env, test_env=None, env_params, args, agent, replay, learner, reward_func,
                 name='algo'):
        self.env = env
        self.test_env = test_env
        self.env_params = env_params
        self.args = args

        self.agent = agent
        self.replay = replay
        self.learner = learner

        self.reward_func = reward_func

        from ml_logger import logger
        self.logger = logger

        self.logger.start('start')
        self.total_timesteps = 0

        self.env_steps = 0
        self.opt_steps = 0

        self.num_envs = 1
        if hasattr(self.env, 'num_envs'):
            self.num_envs = getattr(self.env, 'num_envs')

        # self.n_mpi = mpi_utils.get_size()
        self.n_mpi = 1
        self._save_file = str(name) + '.pt'

        if args.resume_ckpt:
            self.load_checkpoint(args.resume_ckpt)

    def run_eval(self, video_path=None):
        import gym
        env = self.test_env or self.env
        env_id = (Args.test_env_name or Args.env_name).split(':')[-1]
        env_spec = gym.envs.registry.env_specs[env_id]

        total_success_count = 0
        total_trial_count = 0
        frames = []
        for n_test in range(self.args.n_test_rollouts):
            info = None
            observation = env.reset()
            ob = observation['observation']
            bg = observation['desired_goal']
            ag = observation['achieved_goal']
            ag_origin = ag.copy()
            if Args.record_video and video_path and n_test == 0:
                img = env.render("rgb_array", width=200, height=200)
                frames.append(img)
            for timestep in range(env_spec.max_episode_steps):
                a = self.agent.get_actions(ob, bg)
                observation, _, _, info = env.step(a)
                ob = observation['observation']
                bg = observation['desired_goal']
                ag = observation['achieved_goal']
                ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
                self.logger.store_key_value('test/AgChangeRatio', np.mean(ag_changed))

                if video_path and n_test == 0:
                    img = env.render("rgb_array", width=200, height=200)
                    frames.append(img)

            for per_env_info in [info] if isinstance(info, dict) else info:
                total_trial_count += 1
                if per_env_info['is_success']:
                    total_success_count += 1

        if video_path and Args.record_video:
            frames = np.array(frames)
            frames = np.concatenate(frames.transpose([1, 0, 2, 3, 4]))
            self.logger.save_video(frames, video_path)

        success_rate = total_success_count / total_trial_count
        # if mpi_utils.use_mpi():
        #     success_rate = mpi_utils.global_mean(np.array([success_rate]))[0]

        return success_rate

    def save_checkpoint(self, checkpoint):
        raise NotImplementedError

    def load_checkpoint(self, checkpoint):
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

    def save_all(self):
        self.save()
        self.agent.save()
        self.replay.save()
        self.learner.save()

    def load_all(self, path):
        self.load(path)
        self.agent.load(path)
        self.replay.load(path)
        self.learner.load(path)


class Algo(BaseAlgo):
    def __init__(self, *, env, test_env=None, env_params, args, agent, replay, learner, reward_func,
                 name='algo'):
        super().__init__(env=env, test_env=test_env, env_params=env_params, args=args, agent=agent, replay=replay,
                         learner=learner, reward_func=reward_func, name=name)

    def get_actions(self, ob, bg, a_max=1.0, act_randomly=False):
        act = self.agent.get_actions(ob, bg)
        if self.args.noise_eps > 0.0:
            act += self.args.noise_eps * a_max * np.random.randn(*act.shape)
            act = np.clip(act, -a_max, a_max)
        if self.args.random_eps > 0.0:
            a_rand = np.random.uniform(low=-a_max, high=a_max, size=act.shape)
            mask = np.random.binomial(1, self.args.random_eps, self.num_envs)
            if self.num_envs > 1:
                mask = np.expand_dims(mask, -1)
            act += mask * (a_rand - act)
        if act_randomly:
            act = np.random.uniform(low=-a_max, high=a_max, size=act.shape)
        return act

    def agent_optimize(self, epoch):
        logger = self.logger
        batch_size = self.args.batch_size

        # the DDPG agent does not have hard/soft mode.
        logger.start('train')
        for n_train in range(self.args.n_batches):
            batch = self.replay.sample(batch_size=batch_size)

            if Args.object_relabel:
                batch = relabel(batch)

            self.opt_steps += 1
            self.learner.update(batch)

            if self.opt_steps % self.args.target_update_freq == 0:
                self.learner.target_update()

        self.logger.store(dt_trainIter=logger.since('train') / self.args.n_batches)

    def collect_experience(self, epoch, act_randomly=False, train_agent=True):
        ob_list, ag_list, bg_list, a_list = [], [], [], []
        observation = self.env.reset()
        ob = observation['observation']
        ag = observation['achieved_goal']
        bg = observation['desired_goal']

        if train_agent and OMEGA.use_omega:
            batch = self.replay.sample(OMEGA.n)
            goal_pool = batch['bg' if OMEGA.use_behavior_goal else 'ag']
            bg = simple_min_kernel_density_model(k=Args.n_workers, goal_pool=goal_pool)

        ag_origin = ag.copy()
        a_max = self.env_params['action_max']

        for timestep in range(self.env_params['max_timesteps']):
            act = self.get_actions(ob, bg, a_max=a_max, act_randomly=act_randomly)
            ob_list.append(ob.copy())
            ag_list.append(ag.copy())
            bg_list.append(bg.copy())
            a_list.append(act[None, :].copy() if len(act.shape) == 1 else act.copy())

            observation, _, _, info = self.env.step(act)
            ob = observation['observation']
            ag = observation['achieved_goal']
            ag_changed = np.abs(self.reward_func(ag_origin, ag, None))
            # note: used to characterize the number of achieved_goals that moves during
            #   the episode. Lower on harder exploration problems.
            self.logger.store_key_value('train/AgChangeRatio', np.mean(ag_changed))
            self.total_timesteps += self.num_envs * self.n_mpi

            # Looks like all implementations now run optimization inline.
            for _ in range(self.num_envs):
                self.env_steps += 1
                if train_agent and self.logger.every(Args.optimize_every, 'core/optim'):
                    self.agent_optimize(epoch)

            if LfGR.use_lfgr and (epoch or 0) >= LfGR.start:
                self.lfgr.populate_relabel_buffer('backward', self.agent.get_actions)
                self.logger.store_metrics(plan_buffer_len=len(self.lfgr.plan_buffer))

            if LfGR.use_lfgr and len(self.lfgr.plan_buffer) > LfGR.batch_size:
                for _ in range(self.num_envs):
                    if train_agent and self.logger.every(LfGR.optimize_every, 'core/optim_gred'):
                        gred_batch = self.lfgr.plan_buffer.sample(LfGR.batch_size)
                        self.learner.gred_update(gred_batch['o'], gred_batch['g_next'], gred_batch['g_far'])

        ob_list.append(ob.copy())
        ag_list.append(ag.copy())

        experience = dict(ob=ob_list, ag=ag_list, bg=bg_list, a=a_list)
        experience = {k: np.array(v) for k, v in experience.items()}
        if experience['ob'].ndim == 2:
            experience = {k: np.expand_dims(v, 0) for k, v in experience.items()}
        else:
            experience = {k: np.swapaxes(v, 0, 1) for k, v in experience.items()}

        bg_achieve = self.reward_func(bg, ag, None) + 1.

        self.logger.store_key_value("train/success", bg_achieve)

        if LfGR.use_lfgr:
            for ag, ob, a in zip(experience['ag'], experience['ob'], experience['a']):
                if epoch and epoch >= LfGR.start and \
                        self.logger.every(LfGR.store_interval, 'lfgr/store'):
                    self.lfgr.store(achieved_goals=ag, states=ob, actions=a, k=LfGR.k_steps)

        self.replay.store(experience)
        self.update_normalizer(experience)

    def update_normalizer(self, buffer):
        if not self.args.freeze_norm:
            transitions = sample_her_transitions(
                buffer=buffer, reward_func=self.reward_func,
                batch_size=self.env_params['max_timesteps'] * self.num_envs,
                future_p=self.args.future_p)
            self.agent.normalizer_update(obs=transitions['ob'], goal=transitions['bg'])

    def run(self):
        for n_init_rollout in range(self.args.n_initial_rollouts // self.num_envs):
            self.collect_experience(None, act_randomly=True, train_agent=False)

        logger = self.logger
        for epoch in range(self.args.n_epochs + 1):
            if mpi_utils.is_root():
                logger.print('Epoch %d: Iter (out of %d)=' % (epoch, self.args.n_cycles), end=' ', flush=True)
                sys.stdout.flush()

            for n_iter in range(self.args.n_cycles):
                if mpi_utils.is_root():
                    logger.print("%d" % n_iter, end=' ' if n_iter < self.args.n_cycles - 1 else '\n', flush=True)
                    sys.stdout.flush()
                    sys.stderr.flush()

                logger.start('rollout')
                for n_rollout in range(self.args.num_rollouts_per_mpi):
                    self.collect_experience(epoch, train_agent=True)

                logger.store(dt_rollout=logger.since('rollout') / self.args.num_rollouts_per_mpi)

            env_name = Args.test_env_name or Args.env_name
            success_rate = self.run_eval(video_path=f"videos/epoch_{epoch:04d}/{env_name.split(':')[-1]}_agent.mp4")
            if epoch % Args.record_video_freq:
                logger.remove(f"videos/epoch_{epoch - 10:04d}")

            if mpi_utils.is_root():
                logger.print(f'Epoch {epoch} eval success rate {success_rate:.3f}')

            key_values = {
                'epoch': epoch,
                'test/success': success_rate,
                'timesteps': self.total_timesteps,
                'time': logger.since('start'),
                'env_steps': self.env_steps,
                'opt_steps': self.opt_steps,
                'replay_size': self.replay.current_size,
                'replay_fill': self.replay.current_size / self.replay.size,
                'dt_epoch': logger.split('epoch')
            }
            if LfGR.use_lfgr:
                key_values["lfgr/steps"] = self.lfgr.optim_step
                if LfGR.start and epoch >= LfGR.start and epoch % LfGR.visualization_interval == 0:
                    visualize_graph(self.lfgr.graph, f"figures/graph_{epoch:04d}.png")

            logger.log_metrics_summary(key_values=key_values, default_stats='mean')

            if Args.checkpoint_freq and epoch % Args.checkpoint_freq == 0:
                self.save_checkpoint(f"models/ep_{epoch:04d}")
                logger.remove(f"models/{epoch - Args.checkpoint_freq:04d}_cp") 

    def state_dict(self):
        return dict(total_timesteps=self.total_timesteps)

    def load_state_dict(self, state_dict):
        self.total_timesteps = state_dict['total_timesteps']

    def save_checkpoint(self, checkpoint):
        self.logger.save_module(self.agent.actor, checkpoint + "/actor.pkl")
        self.logger.save_module(self.agent.critic, checkpoint + "/critic.pkl")
        self.logger.save_module(self.agent.o_normalizer, checkpoint + "/o_norm.pkl")
        self.logger.save_module(self.agent.g_normalizer, checkpoint + "/g_norm.pkl")

    def load_checkpoint(self, checkpoint):
        self.logger.load_module(self.agent.actor, checkpoint + "/actor.pkl")
        self.logger.load_module(self.agent.critic, checkpoint + "/critic.pkl")
        self.logger.load_module(self.agent.actor_targ, checkpoint + "/actor.pkl")
        self.logger.load_module(self.agent.critic_targ, checkpoint + "/critic.pkl")
        self.logger.load_module(self.agent.o_normalizer, checkpoint + "/o_norm.pkl")
        self.logger.load_module(self.agent.g_normalizer, checkpoint + "/g_norm.pkl")

        if self.args.freeze_phi:
            self.agent.critic.freeze_phi()

        if self.args.freeze_f:
            self.agent.critic.freeze_f()

        if self.args.reinit_phi:
            self.agent.critic.reinit_phi()
        
        if self.args.reinit_f:
            self.agent.critic.reinit_f()

class SACAlgo(Algo):

    def get_actions(self, ob, bg, a_max=1, act_randomly=False):
        return self.agent.get_actions(ob, bg, deterministic=(not act_randomly))

class OfflineAlgo(Algo):

    def run(self):
        logger = self.logger
        print('Working dir: {}'.format(os.getcwd()))
        home = os.path.expanduser("~")
        offline_replay_path = f'{home}/offline_dataset/{self.args.env_name}/mixed_replay.pkl'
        print('Load replay buffer from {}'.format(offline_replay_path))
        self.replay.load_state_dict(torch.load(offline_replay_path))

        print('Compute normalization')
        self.agent.normalizer_update(obs=self.replay.buffers['ob'], goal=self.replay.buffers['bg'])

        for epoch in range(self.args.n_epochs + 1):
            if mpi_utils.is_root():
                logger.print('Epoch %d: Iter (out of %d)=' % (epoch, self.args.n_cycles), end=' ', flush=True)
                sys.stdout.flush()

            for n_iter in range(self.args.n_cycles):                
                if mpi_utils.is_root():
                    logger.print("%d" % n_iter, end=' ' if n_iter < self.args.n_cycles - 1 else '\n', flush=True)
                    sys.stdout.flush()
                    sys.stderr.flush()                
                for _ in range(self.num_envs):
                    self.agent_optimize(epoch)                
            
            env_name = Args.test_env_name or Args.env_name
            success_rate = self.run_eval(video_path=f"videos/epoch_{epoch:04d}/{env_name.split(':')[-1]}_agent.mp4")
            if epoch % 10:
                logger.remove(f"videos/epoch_{epoch - 10:04d}")

            if mpi_utils.is_root():
                logger.print(f'Epoch {epoch} eval success rate {success_rate:.3f}')

            key_values = {
                'epoch': epoch,
                'test/success': success_rate,              
                'time': logger.since('start'),
                'opt_steps': self.opt_steps,
            }      

            logger.log_metrics_summary(key_values=key_values, default_stats='mean')

            if Args.checkpoint_freq and epoch % Args.checkpoint_freq == 0:
                self.save_checkpoint(f"models/ep_{epoch:04d}")
                logger.remove(f"models/{epoch - Args.checkpoint_freq:04d}_cp")
