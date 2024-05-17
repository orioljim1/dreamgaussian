import os
import glob
import argparse
import time

time.sleep(4)

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='data', type=str, help='Directory where processed images are stored')
parser.add_argument('--out', default='logs', type=str, help='Directory where obj files will be saved')
parser.add_argument('--gpu', default=0, type=int, help='ID of GPU to use')
parser.add_argument('--elevation', default=0, type=int, help='Elevation angle of view in degrees')
parser.add_argument('--config', default='configs', type=str, help='Path to config directory, which contains image.yaml')
args = parser.parse_args()

files = glob.glob(f'{args.dir}/*_rgba.png')
configs_dir = args.config

# check if image.yaml exists
if not os.path.exists(os.path.join(configs_dir, 'image.yaml')):
    raise FileNotFoundError(
        f'image.yaml not found in {configs_dir} directory. Please check if the directory is correct.'
    )

# create output directories if not exists
out_dir = args.out
os.makedirs(out_dir, exist_ok=True)

#make dirs for depth and no depth


for file in files:
    name = os.path.basename(file).replace("_rgba.png", "")

    os.makedirs(out_dir + f'/{name}', exist_ok=True)
    out_dir_depth = out_dir + f'/{name}' + '/depth'
    out_dir_default = out_dir + f'/{name}' + '/default'
    os.makedirs(out_dir_depth,  exist_ok=True)
    os.makedirs(out_dir_default,  exist_ok=True)
    print(f'======== processing w depth {name} ========')
    # first stage
    #opt.outdir
    os.system(f'python -m debugpy --listen 5678 main.py '
              f'--config {configs_dir}/image.yaml '
              f'input={file} '
              f'save_path={name} '
              f'outdir={out_dir_depth} '
              f'use_depth=True')
    
    print(f'======== processing default {name} ========')

    os.system(f'python -m debugpy --listen 5678 main.py '
              f'--config {configs_dir}/image.yaml '
              f'input={file} '
              f'save_path={name} '
              f'outdir={out_dir_default} '
              f'use_depth=False')
    


    # second stage
    '''
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python main2.py '
              f'--config {configs_dir}/image.yaml '
              f'input={file} '
              f'save_path={name} elevation={args.elevation}')
    '''