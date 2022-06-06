from gym.envs.registration import register

robots = ['Point', 'Ant']
task_types = ['Maze', 'Maze-train']  # , 'Push', 'Fall', 'Block', 'BlockMaze']
all_name = [x + y for x in robots for y in task_types]
for name_t in all_name:

    if 'Point' in name_t:
        # training is 50 steps
        max_timestep = 50  # 200
    elif 'Ant' in name_t:
        # training is 100 steps
        max_timestep = 50  # 200
    else:
        raise RuntimeError(f"{name_t} is not supported")

    if 'train' in name_t:
        goal_args = [[-4., -4.], [20, 20]]
        random_start = True
    else:
        goal_args = [[0., 16.], [1e-3, 16 + 1e-3]]
        random_start = False
        # give test more time
        max_timestep *= 4  # 500

    # v1 is the one we use in the main paper
    register(
        id=name_t + '-v1',
        entry_point='goal_env.mujoco.create_maze_env:create_maze_env',
        kwargs={'env_name': name_t, 'goal_args': goal_args, 'maze_size_scaling': 4, 'random_start': random_start},
        max_episode_steps=max_timestep,
    )
