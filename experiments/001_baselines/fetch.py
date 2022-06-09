from params_proto.neo_hyper import Sweep

from rl import Args, main

from jaynes import jaynes
from params_proto.neo_hyper import Sweep

if __name__ == '__main__':
    from experiments import RUN, instr

    with Sweep(RUN, Args) as sweep:
        Args.cuda = True
        Args.record_video=True
        Args.clip_inputs = True
        Args.normalize_inputs = True
        Args.agent_type = 'ddpg'    # todo: there is a problem with using 'sac' and 'td3' and 'cuda' together. parent class calls child class' method.

        with sweep.product:
            with sweep.zip:
                Args.env_name = ['FetchPickAndPlace-v1']
                Args.n_workers = [2]
                Args.n_epochs = [250]

                # Args.env_name = ['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1', 'FetchSlide-v1']
                # Args.n_workers = [2, 2, 2, 2]  # [2, 8, 16, 20]
                # Args.n_epochs = [50, 150, 200, 500]

            Args.seed = [100, 200, 300, 400, 500]            

    @sweep.each
    def fn(RUN, Args, *_):
        RUN.job_name = ("../debug/" if Args.debug else "") + f"{Args.agent_type}/{Args.env_name.split(':')[-1]}/{Args.seed}"
        RUN.prefix = f"{RUN.project}/{RUN.project}/baselines/train/{RUN.job_name}"

    for i, deps in sweep.items():     
        jaynes.config("local", runner=dict(n_cpu=Args.n_workers + 2))
        thunk = instr(main, deps)
        jaynes.run(thunk)        

    jaynes.listen()
