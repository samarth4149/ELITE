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
NUM_JOBS = 5
dataset = 'domainnet'
domains = ['sketch', 'clipart', 'painting']

for src, tgt in itertools.permutations(domains, 2):
    for job_idx in range(NUM_JOBS):
        strength = 0.5
        root_dir = f'/projectnb/ivc-ml/samarth/projects/synthetic/data/synthetic-cdm/synthetic_data/elite_global_img2img/strength_{strength}'
        scenario = f'{src[0]}2{tgt[0]}'
        expt_name = f'elite_global_img2img_strength_{strength}_{dataset}_{scenario}_job_{job_idx}'
        proc_arr = ['qsub']
        proc_arr.extend(get_qsub_options(
            qsub_name=expt_name,
            outfile=Path(root_dir) / dataset / scenario / f'qsub_log_{job_idx}.txt',
            project=PROJECT,
            duration='11:50:00',
            # gpu_c='4.5',
            gpu_memory='20G',
        ))

        proc_arr += ['python', 'gen_data.py']
        proc_arr += ['--dataset', dataset]
        proc_arr += ['--source', src]
        proc_arr += ['--target', tgt]
        proc_arr += ['--root_dir', root_dir]
        proc_arr += ['--batch_size', '12']
        proc_arr += ['--num_jobs', str(NUM_JOBS)]
        proc_arr += ['--job_idx', str(job_idx)]
        
        proc_arr += ['--strength', str(strength)]
        
        print('Job name : ', expt_name)
        print('Command :', ' '.join(proc_arr))
        os.system(' '.join(proc_arr))
        