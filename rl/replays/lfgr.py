from collections import defaultdict, deque
from functools import partial

import numpy as np
import torch
from torch import optim
import torch.nn as nn

from graph_search import dijkstra
from sparse_graphs.asym_graph import AsymMesh
from rl.replays.linear_buffer import IndexBuffer, to_float, BaseBuffer
from rl.utils.torch_utils import torchify, _torchify


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.alpha


def l2(a, b):
    return np.linalg.norm(a - b, axis=-1)


class L2(nn.Module):
    def forward(self, a, b):
        return torch.norm(a - b, dim=-1)


def rev_hinge_loss(x, y: float):
    d = x - torch.ones_like(x, requires_grad=False) * y
    return - (d * (d < 0).float()).mean()


class Cache:
    def __init__(self, default_factory=lambda: None):
        self._data = defaultdict(default_factory)

    def __getitem__(self, item):
        if np.issubdtype(type(item), np.integer):
            return self._data[item]
        elif isinstance(item, list):
            return [self[i] for i in item]

    def __setitem__(self, key, value):
        if isinstance(key, list):
            for k, v in zip(key, value):
                self._data[k] = v

    def __len__(self):
        return self._data

    def clear(self):
        return self._data.clear()


class GraphicalReplay:
    optim = None
    graph: AsymMesh

    def __init__(self, goal_dim, reward_fn, n=10_000, goal_buffer_n=500,
                 *, plan_buffer_n, d_max, obs_bucket_size):
        """
        Assume we have access to the true reward R(g, g') that takes the
        form of an indicator function $(d(g, g') < ε) - 1$. We use this
        reward to train the local metric
        \[                     d(g, g') = - R(g, g')                   \]
        by relabeling randomly sampled pairs in the graph.

        :param goal_dim:
        :param n:
        :param kwargs:
        """
        from rl import LfGR

        self.reward_fn = reward_fn
        self.embed_fn = Scale()
        self.kernel_fn = L2()
        self.goal_pairs = IndexBuffer(goal_buffer_n, "goal")
        self.graph = AsymMesh(n=n, k=int(3 * goal_dim), dim=goal_dim,
                              img_dim=[goal_dim],
                              embed_fn=_torchify(self.embed_fn, with_eval=True),
                              kernel_fn=l2,  # use numpy to speed things up.
                              d_max=d_max)
        # self.bucket_size = obs_bucket_size
        if LfGR.use_state_bucket:
            factory = partial(deque, maxlen=obs_bucket_size)
            self.state_buffer = Cache(factory)
        self.plan_buffer = BaseBuffer(plan_buffer_n)
        from ml_logger import logger
        self.logger = logger

    @torchify(input_only=True)
    def d(self, g, g_prime):
        zs = self.embed_fn(g)
        zs_2 = self.embed_fn(g_prime)
        return self.kernel_fn(zs, zs_2)

    def sample_tasks(self, batch_size):
        inds = np.random.choice(len(self.graph), batch_size)
        return self.graph.indices[inds]

    optim_step = 0

    def fit(self, batch_size, lr, steps=None):
        """
        fit the embedding function to emulate the reward function, with a hinge loss.

        :param batch_size: batch size
        :param lr: learning rate
        :param steps: number of optimizations to run
        """
        self.optim = self.optim or optim.Adam(self.embed_fn.parameters(), lr=lr)
        l1 = nn.SmoothL1Loss()

        # use while else or random sampling (with replacement).
        for (g, g_prime, r), g_shuffle in zip(
                self.goal_pairs.sample_all(batch_size, 'goal@t', 'goal@t_next', 'r', proc=to_float),
                self.goal_pairs.sample_all(batch_size, 'goal@t', proc=to_float)
        ):
            self.optim_step += 1
            d_hat = self.d(g, g_prime)
            d_shuffle = self.d(g, g_shuffle)
            # two = torch.full_like(d_shuffle, 2, requires_grad=False)
            # these are 1 dynamic distance.
            loss = l1(d_hat, - r.float())  # + rev_hinge_loss(d_shuffle, two).mean()
            self.logger.store(r=r.mean().item(), d_hat=d_hat.mean().item(),
                              d_shuffle=d_shuffle.mean().item(), loss=loss.item(),
                              k_steps=self.k_steps, prefix="lfgr/")

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if steps and self.logger.every(steps, 'lfgr/fit'):
                break

    k_steps = 1

    def store(self, achieved_goals, states, actions, k=4):
        """

        :param states: [H + 1, dim_s]
        :param achieved_goals: [H + 1, dim_goal]
        :param actions: [H, dim_a]
        :param k: If k == 1, the step wise distance might be too
            small for the reward function to pick up. As a result the
            value might always be zero.

            This is an artifact of using a binary reward. If the reward
            is dense (dense in the sense that it is the stepwise distance,
            NOT in the sense of a shaped reward) this should not be an
            issue.

            When labeling the rewards, we can choose from two strategies:
            1. we can scale the relabeled reward by k. Thus 1 step is -1,
            4 steps becomes -4. We would also need to change the `d_min`
            during graph construction to make sure that the edges are
            significant in length, therefore enforcing sparsity (in
            coverage).
            2. Alternatively we can keep the reward to always -1. This way
            we do not need to scale `d_min`.

            Adaptive k_steps does not work well because the graph can be
            too sparse. I have not tried to change d_min, but it should help
            a great deal.
        """
        from rl import LfGR

        if k == "adaptive":
            k = self.k_steps
            # you want to make sure that this is close to -1, or at least 50%.
            labeled_rewards = self.reward_fn(achieved_goals[:-k], achieved_goals[k:])
            # make a linear assumption. Not necessarily correct
            while k > 1 and (- labeled_rewards.mean()) * (k - 1) / k > 0.7:
                k -= 1
                labeled_rewards = self.reward_fn(achieved_goals[:-k], achieved_goals[k:])
            while (-labeled_rewards.mean()) < 0.7 and k < 10:
                k += 1
                labeled_rewards = self.reward_fn(achieved_goals[:-k], achieved_goals[k:])
            self.k_steps = k
        else:
            labeled_rewards = self.reward_fn(achieved_goals[:-k], achieved_goals[k:])

        end, H = self.goal_pairs.end, len(achieved_goals) - k
        self.goal_pairs.extend(achieved_goals,
                               t=end + np.arange(H, dtype=int),
                               t_next=end + np.arange(k, H + k, dtype=int),
                               r=labeled_rewards)
        self.fit(32, 1e-1, steps=1)

        self.graph.sparse_extend(achieved_goals[:-1].copy(), d_min=LfGR.d_min,
                                 o=states[:-1].copy(), a=actions.copy())

        # add additional states to each goal
        # fixit: list could contain states for different goals if goal index overflows.
        if LfGR.use_state_bucket:
            zs = self.graph.embed_fn(achieved_goals)
            ds = self.graph.to_goal(zs_2=zs)
            for center, s in zip(self.graph.indices[ds.argmin(axis=0)], states):
                self.state_buffer[center].append(s)

        self.graph.update_edges()

    def populate_relabel_buffer(self, mode, policy):
        from rl import LfGR
        graph = self.graph
        k = LfGR.relabel_k_steps

        try:
            start, goal = self.sample_tasks(2)
            path, ds = dijkstra(graph, start, goal)
            assert path is not None
            assert len(path) > (k + 1)
        except:
            return

        if mode == "forward":
            pass
        elif mode == "backward":
            # this is a hyper parameter
            ind, ind_next = path[:-k], path[k:]
            g_next = np.stack(graph._meta['images'][ind_next])

            if LfGR.use_state_bucket:
                obs = self.state_buffer[ind]
                # the buffer is irregularly shaped.
                for o, g in zip(obs, g_next):
                    o = np.stack(o)
                    new_shape = [*o.shape[:-1], 1]
                    g = np.tile(g[None, ...], new_shape)
                    # todo: save logits. or remove local_metric completely (double self-distillation)
                    # a_local = policy(o, g)
                    g_far = graph._meta['images'][path[-1:] * len(g)]
                    self.plan_buffer.extend(o=o, g_next=g, g_far=g_far)  # a=a_local)
            else:
                o = np.stack(graph._meta['o'][ind])
                # todo: save logits. or remove local_metric completely (double self-distillation)
                # a_local = policy(o, g_next)
                g_far = graph._images[path[-1:] * len(g_next)]
                self.plan_buffer.extend(o=o, g_next=g_next, g_far=g_far)  # , a=a_local)

        else:
            raise NotImplementedError(f"{mode} is bad.")

    def sample(self, batch_size):
        """
        sample from he plan_buffer

        :param batch_size:
        :return: o, g, a
        """
        return self.plan_buffer.sample(batch_size)


def visualize_graph(graph, filename=f"figures/graph.png"):
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from ml_logger import logger

    plt.figure()
    graph.update_zs()
    graph.dedupe_(d_min=1.0)
    graph.update_edges()

    plt.gca().set_aspect('equal')
    for i, j in tqdm(graph.edges, desc="sparse"):
        a, b = graph._meta['images'][[i, j]]
        plt.plot([a[0], b[0]], [a[1], b[1]], color="red", linewidth=0.4)
    logger.savefig(filename)
