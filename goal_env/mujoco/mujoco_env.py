from gym.envs.mujoco import mujoco_env


class MujocoEnv(mujoco_env.MujocoEnv):
    def render_callback(self):
        pass
