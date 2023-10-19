import os
import sys
from typing import Any
sys.path.append(os.path.abspath('.')) # CDS_pretraining
sys.path.append(os.path.abspath('..')) # project root
from collections import OrderedDict
import itertools
import submitit
import copy
import gen_cub

NUM_JOBS = 5

class Trainer:
    def __init__(self, args) -> None:
        self.args = copy.deepcopy(args)

    def __call__(self) -> Any:
        gen_cub.main(self.args)
        
if __name__ == '__main__':
    dataset = 'cub'
    domains = ['Real', 'Painting']

    for src, tgt in itertools.permutations(domains, 2):
        for job_idx in range(NUM_JOBS):
            scenario = f'{src[0]}2{tgt[0]}'
            root_dir = '/gpfs/u/home/LMTM/LMTMsmms/scratch/data/synthetic-cdm/synthetic_data/elite_global'
            expt_name = f'elite_global_{scenario}_job_{job_idx}'
            
            args = gen_cub.parse_args([])
            args.dataset = dataset
            args.source = src
            args.target = tgt
            args.root_dir = root_dir
            args.batch_size = 12
            args.num_jobs = NUM_JOBS
            args.job_idx = job_idx
            
            trainer = Trainer(args)
            num_gpus_per_node = 1
            os.makedirs(f'expts/all_expts/{expt_name}/slurm', exist_ok=True)
            executor = submitit.AutoExecutor(folder=f'expts/all_expts/{expt_name}/slurm', slurm_max_num_timeout=30)
            addnl_params = {
                'gres': f'gpu:{num_gpus_per_node}',
                'mail_type': 'FAIL',
                'mail_user' : 'samarthm@bu.edu',
            }
            executor.update_parameters(
                name=expt_name,
                mem_gb=40,
                tasks_per_node=num_gpus_per_node,
                cpus_per_task=10,
                timeout_min=360,
                # slurm_partition='el8',
                slurm_signal_delay_s=120,
                slurm_additional_parameters=addnl_params,
            )
            
            print('Job Name :', expt_name)
            # loop for submitting
            import time
            submitted = False
            num_tries = 0
            while not submitted and num_tries < 10:
                try:
                    job = executor.submit(trainer)
                    submitted = True
                except Exception:
                    print('Submission failed. Trying again in 5 seconds')
                    time.sleep(5)
                    submitted = False
                    num_tries += 1
            if num_tries >= 10:
                raise Exception(f'Failed to submit job {expt_name}')
            
            print("Submitted job_id:", job.job_id)
            