from params_proto.neo_hyper import Sweep

from rl import Args, main

from jaynes import jaynes
from ml_logger import logger
from params_proto.neo_hyper import Sweep

if __name__ == '__main__':
    from experiments import RUN, instr

    with Sweep(RUN, Args) as sweep:
        Args.clip_inputs = True
        Args.normalize_inputs = True
        Args.agent_type = 'ddpg'

        with sweep.product:
            with sweep.zip:
                Args.env_name = ['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1', 'FetchSlide-v1']
                Args.n_workers = [2, 8, 16, 20]
                Args.n_epochs = [50, 150, 200, 500]

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
