import os
import glob
import argparse
import time
import json
import ast
import subprocess

time.sleep(4)

def get_last_line(log_data):
    # Split the string into lines
    lines = log_data.strip().splitlines()
    # Return the last line if lines exist, otherwise return an empty string
    return lines[-1] if lines else ''


def save_string_to_file(text, file_path):
    """
    Save a string to a text file.

    Parameters:
    - text: The string to be saved.
    - file_path: The path (including filename) where the text will be saved.
    """
    try:
        with open(file_path, 'w') as file:
            file.write(text)
        print(f"Successfully saved to {file_path}")
    except IOError as e:
        print(f"Error saving to {file_path}: {e}")

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='data', type=str, help='Directory where processed images are stored')
parser.add_argument('--out', default='logs', type=str, help='Directory where obj files will be saved')
parser.add_argument('--gpu', default=0, type=int, help='ID of GPU to use')
parser.add_argument('--elevation', default=0, type=int, help='Elevation angle of view in degrees')
parser.add_argument('--config', default='configs', type=str, help='Path to config directory, which contains image.yaml')
args = parser.parse_args()

lambda_to_test = [0,1,10,50,100,500,1000,10000]

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



def read_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        return {}
results_dict = read_json_file(out_dir+'/Results.json')
for file in files:
    name = os.path.basename(file).replace("_rgba.png", "")

    #results_dict[name] = {}
    
    out_dir_file = out_dir + f'/{name}'
    if name not in results_dict:
        results_dict[name] = read_json_file(out_dir_file+'/image_generation_results.json')
    os.makedirs(out_dir_file, exist_ok=True)

    for lambda_i in lambda_to_test:
        out_dir_generation = out_dir_file + f'/{lambda_i}'
        print(f'======== processing lambda {lambda_i} ========')
        if os.path.exists(out_dir_generation):
              continue
        os.makedirs(out_dir_generation,  exist_ok=True)
        
        # first stage
        #opt.outdir
        command_main = (f'python main.py '
                f'--config {configs_dir}/image.yaml '
                f'input={file} '
                f'save_path={name} '
                f'outdir={out_dir_generation} '
                f'use_depth=True '
                f'dloss_lambda=0 '
                f'canny_lambda={lambda_i} '
                f'text_to_img=False')

        result = subprocess.run(command_main, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        scores = get_last_line(result.stdout) #pnsr, ssim, lpips
        # Get the default system encoding
        scores = ast.literal_eval(scores)
        scores = {"PNSR": scores[0], "SSIM": scores[1], "LPIPS": scores[2]}
        
        command = f'python -m kiui.cli.clip_sim ' \
          f'{file} ' \
          f'{out_dir_generation + "/" + name}_mesh.obj' 
        
        print(f'========Evaluating CLIP score of the lambda {lambda_i} generation ========')
        process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        CLIP_score = get_last_line(process.stdout.decode())
        print("Clip score:", CLIP_score)

        scores["CLIP"] = float(CLIP_score)

        with open(out_dir_generation + '/scores.json', 'w') as f:
            json.dump(scores, f)

        results_dict[name][lambda_i] = scores
        with open(out_dir_file+'/image_generation_results.json', 'w') as f:
                    json.dump(results_dict[name], f)
    
with open(out_dir+'/Results.json', 'w') as f:
                json.dump(results_dict, f)