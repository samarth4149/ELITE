from pathlib import Path
import shutil
import os

def get_qsub_options(
    qsub_name,
    outfile,
    project='ivc-ml',
    duration='12:00:00',
    mem_per_core='6G',
    gpu_c='4.5',
    gpu_memory='16G',
    log_overwrite=True,
    num_workers=3,):

    if not Path(outfile).parent.exists():
        print(f'Creating path for outfile : {outfile}')
        Path(outfile).parent.mkdir(parents=True)
        
    if log_overwrite and Path(outfile).exists():
        print(f'Overwriting {outfile}')
        os.remove(outfile)
    
    options = [ 
        '-V',
        '-b', 'y', # To say that "command is executable"
        '-N', qsub_name,
        '-j', 'y',
        '-P', project,
        '-o', str(outfile),
        '-pe', f'omp {num_workers}',
        f'-l h_rt={duration}',
        f'-l mem_per_core={mem_per_core}',
        f'-l gpus=1',
        f'-l gpu_c={gpu_c}',
        f'-l gpu_memory={gpu_memory}',
        '-m', 'beas',
    ]   
    return options