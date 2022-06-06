from gym.envs.registration import register

from goal_env.fetch.bin import BinEnv


kw = dict(max_episode_steps=50, )
# # original gym envs
# for action in ['reach', 'push', 'pick-place', 'slide']:
#     register(  # Same as FetchPickAndPlace, with a bin.
#         id=f"{action.title().replace('-', '')}-v0",
#         entry_point=GymFetchEnv, kwargs=dict(action=action, ), **kw)
# # Fetch

# ------------------------ Finalized ------------------------
# Bin Environments Bin + object, no lid
for mode in ['Near', 'Far', 'Left', 'Right', 'Train', 'Test']:
    # Keep this for backward compatability
    if mode == 'Train':
        goal_sampling_type = 'near'
    elif mode == 'Test':
        goal_sampling_type = 'far'
    else:
        goal_sampling_type = mode.lower()
    
    kwargs = {
        'reward_type': 'sparse',
        'goal_sampling_type': goal_sampling_type,
    }
    register(id=f'Bin-pick-{mode}-v0', entry_point=BinEnv,
            kwargs=dict(action="pick", obs_keys=['object0', 'bin@pos'], far_goal_threshold=0.2, **kwargs), **kw)
    register(id=f'Bin-place-{mode}-v0', entry_point=BinEnv,
            kwargs=dict(action="place+air", obs_keys=['object0', 'bin@pos'], far_goal_threshold=0.24), **kw)
