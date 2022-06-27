from params_proto.neo_hyper import Sweep

from rl import Args, main

from jaynes import jaynes
from ml_logger import logger
from params_proto.neo_hyper import Sweep

import goal_env.robotics

if __name__ == '__main__':
    from experiments import RUN, instr

    with Sweep(RUN, Args) as sweep:
        Args.cuda = False
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

        Args.update_with_predict_r = False
        
        Args.future_p = 0.8
        Args.do_relabel_filter = False
        Args.resume_ckpt = '/home/jason/bvn/experiments/003_sa/bvn-saved1/bvn/bvn/train/ddpg/FetchPush-v1/100/models/ep_0100'
        # Args.resume_ckpt = ""

        with sweep.product:
            with sweep.zip:
                # reaching is easy, so skip for now
                Args.env_name = ['FetchPickAndPlace-v1']
                # Args.test_env_name = ['FetchPickAndPlace-v1']
                Args.n_workers = [16]
                Args.n_epochs = [200]

                # Args.env_name = ['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1', 'FetchSlide-v1']
                # Args.n_workers = [2, 8, 16, 20]
                # Args.n_epochs = [50, 150, 200, 500]

            Args.seed = [100]
            # Args.seed = [100, 200, 300, 400, 500]

    @sweep.each
    def fn(RUN, Args, *_):
        RUN.job_name = ("../debug/" if Args.debug else "") + f"{Args.agent_type}/{Args.env_name.split(':')[-1]}/{Args.seed}"
        RUN.prefix = f"{RUN.project}/{RUN.project}/bvn/train/{RUN.job_name}"

    for i, deps in sweep.items():
        jaynes.config("local", runner=dict(n_cpu=Args.n_workers + 2))
        thunk = instr(main, deps)
        jaynes.run(thunk)
        
    jaynes.listen()
