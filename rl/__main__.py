from rl import main

import os
from ml_logger import logger

logger.configure(f"{os.environ['HOME']}/runs/",
                 prefix=f"{os.environ['USER']}/latent-planning/{logger.now('%Y/%m-%d/%H.%M.%S-%f')}",
                 register_experiment=True)
main()
