import numpy as np

class Replay:
    def __init__(self, obs_shape, goal_shape, action_shape, capacity, horizon, gamma=1):
        self.gamma = gamma

        self.capacity = capacity
        self.horizon = horizon
        self.size = self.capacity // self.horizon

        self.current_size = 0
        self.n_transitions_stored = 0

        self.buffers = dict(ob=np.zeros((self.size, self.horizon, *obs_shape)),
                            ag=np.zeros((self.size, self.horizon, goal_shape)),
                            bg=np.zeros((self.size, self.horizon, goal_shape)),
                            a=np.zeros((self.size, self.horizon, action_shape)),
                            r=np.zeros((self.size, self.horizon, 1)),
                            q=np.zeros((self.size, self.horizon, 1)))
        
        self.idx = 0
        self.full = False

    def store(self, trajectory):
        np.copyto(self.buffers['ob'][self.idx], trajectory['observations'])
        np.copyto(self.buffers['ag'][self.idx], trajectory['achieved_goals'])
        desired_goal = np.expand_dims(trajectory['achieved_goals'][-1], 
                                      axis=0).repeat(self.horizon, axis=0)
        np.copyto(self.buffers['bg'][self.idx], desired_goal)
        # np.copyto(self.buffers['bg'][self.idx], trajectory['desired_goals'])
        np.copyto(self.buffers['a'][self.idx], trajectory['actions'])
        np.copyto(self.buffers['r'][self.idx], trajectory['rewards'])

        # calculate q value estimate from rewards in trajectory
        q = self.buffers['q'][self.idx]
        r = self.buffers['r'][self.idx]

        q[-1] = r[-1]
        for i in range(self.horizon - 1, 0, -1):
            q[i - 1] = r[i - 1] + self.gamma * q[i]
        # calculate q value estimate from rewards in trajectory

        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.current_size = self.size if self.full else self.idx
    
    def sample(self, batch_size, **kwargs):
        o2_buffer = self.buffers['ob'][:, 1:, :]
        a2_buffer = self.buffers['a'][:, 1:, :]

        traj_idxes = np.random.randint(0, self.current_size, size=batch_size)
        tran_idxes = np.random.randint(0, self.horizon - 1, size=batch_size)  # -1 because otherwise no next observation

        batch = {key: self.buffers[key][traj_idxes, tran_idxes].copy() for key in self.buffers.keys()}
        batch['o2'] = o2_buffer[traj_idxes, tran_idxes].copy()
        batch['a2'] = a2_buffer[traj_idxes, tran_idxes].copy()

        return batch
