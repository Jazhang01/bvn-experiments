import os
import wandb
from ml_logger import logger

projects = ['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1']
experiments = ['001_baselines', '002_bvn', '003_sa']

group = 'baseline'
experiment = '001_baselines'
project = 'FetchPushDense-v1'
seed = 200
run_name = f'{group}-{seed}'

# group = 'sa'
# experiment = '003_sa'
# project = 'FetchPushDense-v1'
# seed = 100
# run_name = f'{group}-{seed}-no-her-detached-xi-from-q-loss'

# group = 'bvn'
# experiment = '002_bvn'
# project = 'FetchPushDense-v1'
# seed = 100
# run_name = f'{group}-{seed}'

# group = 'usfa'
# experiment = '004_usfa'
# project = 'FetchPushDense-v1'
# seed = 100
# run_name = f'{group}-{seed}-detached-xi-from-q-loss'

pkl_path = f'{experiment}/bvn-dense-her/bvn/baselines/train/ddpg/{project}/{seed}/metrics.pkl'
metrics = logger.load_pkl(pkl_path)

print(metrics)

os.environ['WANDB_API_KEY'] = 'd86def89f79a04e9c8a9c794a54288a3af1aa335'
wandb.init(project=project,
           entity='jazhang',
           name=run_name,
           group=group,
           mode='online')
for item in metrics:
    if 'epoch' in item:
        wandb.log(item, step=item['epoch'])
wandb.finish()
