import os
import sys
sys.path.append(os.path.abspath('.')) # CDS_pretraining
sys.path.append(os.path.abspath('..')) # project root
from collections import OrderedDict
import itertools
from pathlib import Path
from utils_qsub import get_qsub_options
import itertools

PROJECT = 'ivc-ml'
NUM_JOBS = 4
dataset = 'cub'
domains = ['Real', 'Painting']

for src, tgt in itertools.permutations(domains, 2):
    if src == 'Real':
        num_jobs = 4
    else:
        num_jobs = 2
    for job_idx in range(num_jobs):
        root_dir = '/projectnb/ivc-ml/samarth/projects/synthetic/data/synthetic-cdm/synthetic_data/elite_global_no_bird'
        scenario = f'{src[0]}2{tgt[0]}'
        
        expt_name = f'elite_global_no_bird_{scenario}_job_{job_idx}'
        proc_arr = ['qsub']
        proc_arr.extend(get_qsub_options(
            qsub_name=expt_name,
            outfile=Path(root_dir) / dataset / scenario / f'qsub_log_{job_idx}.txt',
            project=PROJECT,
            duration='6:00:00',
            # gpu_c='4.5',
            gpu_memory='20G',
        ))

        proc_arr += ['python', 'gen_data.py']
        proc_arr += ['--dataset', dataset]
        proc_arr += ['--source', src]
        proc_arr += ['--target', tgt]
        proc_arr += ['--root_dir', root_dir]
        proc_arr += ['--batch_size', '12']
        proc_arr += ['--num_jobs', str(num_jobs)]
        proc_arr += ['--job_idx', str(job_idx)]
        
        print('Job name : ', expt_name)
        print('Command :', ' '.join(proc_arr))
        os.system(' '.join(proc_arr))