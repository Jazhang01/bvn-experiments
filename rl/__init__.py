import sys

from params_proto.neo_proto import ParamsProto, Flag, Proto, PrefixProto


class Args(PrefixProto):
    """Soft-actor Critic Implementation with SOTA Performance
    """
    debug = True if "pydevd" in sys.modules else False
    record_video = False
    record_video_freq = 10
    train_type = 'online'
    agent_type = 'ddpg'

    # For SAC
    policy_output_scale = 1.0
    initial_temperature = 1.0
    temperature_optimizer_lr = 3e-4
    entropy_target = 'action_size'

    # For TD3
    smooth_targ_policy = True

    # Transfer options
    reinit_phi = False
    freeze_phi = False
    reinit_f = False
    freeze_f = False
    freeze_norm = False

    # Critic type
    critic_type = 'td'
    # norm, dot
    critic_reduce_type = 'norm'
    
    metric_embed_dim = 16
    
    # For `norm` reduce type
    metric_norm_ord = 2
    
    # Actor
    n_actor_optim_steps = 1

    # experimental features
    object_relabel = False

    env_name = "FetchReach-v1"
    test_env_name = None

    seed = 123
    load_seed = 123
    save_dir = "experiments/"
    ckpt_name = ""
    resume_ckpt = ""

    n_workers = 2 if debug else 12
    cuda = Flag("cuda tend to be slower.")
    num_rollouts_per_mpi = 1

    n_epochs = 200
    n_cycles = 10
    optimize_every = 2
    n_batches = 1

    hid_size = 256
    n_hids = 3
    activ = "relu"
    noise_eps = 0.1
    random_eps = 0.2

    buffer_size = 2500000
    future_p = 0.8
    batch_size = 1024

    clip_inputs = Flag("to turn on input clipping")
    clip_obs = Proto(200, dtype=float)

    normalize_inputs = Flag("to normalize the inputs")
    clip_range = Proto(5, dtype=float)

    gamma = 0.98
    clip_return = Proto(50, dtype=float)

    action_l2 = 0.01
    lr_actor = 0.001
    lr_critic = 0.001

    polyak = 0.995
    target_update_freq = 10
    checkpoint_freq = 10

    n_initial_rollouts = 1 if debug else 100
    n_test_rollouts = 15
    demo_length = 20
    play = Flag()


class LfGR(PrefixProto):
    # reporting
    use_lfgr = False
    start = 0 if Args.debug else 10
    store_interval = 10
    visualization_interval = 10

    plan_buffer_size = 5_000

    k_steps = 4
    relabel_k_steps = 2

    d_max = 2
    d_min = 0.8

    optimize_every = 2
    relabel = "backward"
    batch_size = 32

    use_state_bucket = True
    obs_bucket_size = 32


def main(deps=None):
    from rl.launcher import launch

    algo = launch(deps)
    algo.run()
