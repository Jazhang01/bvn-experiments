import os
import wandb

class WandBLogger(object):
    def __init__(self, project_name, run_name, run_group, mode='online'):
        os.environ['WANDB_API_KEY'] = 'd86def89f79a04e9c8a9c794a54288a3af1aa335'

        run = wandb.init(project=project_name, 
                        entity='jazhang',
                        name=run_name, 
                        group=run_group,
                        mode=mode)

    def log(self, info, step):
        wandb.log(info, step)
    
    def finish(self):
        wandb.finish()