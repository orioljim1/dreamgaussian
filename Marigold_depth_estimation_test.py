import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from Marigold.marigold import MarigoldPipeline


if torch.cuda.is_available():
            device = torch.device("cuda")
else:
    device = torch.device("cpu")
    logging.warning("CUDA is not available. Running on CPU will be slow.")

dtype = torch.float32
variant = None
match_input_res = True

pipe = MarigoldPipeline.from_pretrained(
        "prs-eth/marigold-lcm-v1-0", variant=variant, torch_dtype=dtype
    )

try:
    pipe.enable_xformers_memory_efficient_attention()
except ImportError:
    pass  # run without xformers

pipe = pipe.to(device)

rgb_path = "./data/charmander_rgba.png"
output_dir = "./test_path/output"

# Output directories
output_dir_color = os.path.join(output_dir, "depth_colored")
output_dir_tif = os.path.join(output_dir, "depth_bw")
output_dir_npy = os.path.join(output_dir, "depth_npy")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_color, exist_ok=True)
os.makedirs(output_dir_tif, exist_ok=True)
os.makedirs(output_dir_npy, exist_ok=True)
logging.info(f"output dir = {output_dir}")

# -------------------- Inference and saving --------------------
with torch.no_grad():
    os.makedirs(output_dir, exist_ok=True)


    input_image = Image.open(rgb_path)

    # Predict depth
    pipe_out = pipe(
        input_image,
        denoising_steps=4,
        ensemble_size=5,
        processing_res=768,
        match_input_res=match_input_res,
        batch_size=1,
        color_map="Spectral",
        show_progress_bar=True,
        resample_method="bilinear",
        seed=None,
    )

    depth_pred: np.ndarray = pipe_out.depth_np
    depth_colored: Image.Image = pipe_out.depth_colored

    # Save as npy
    rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
    pred_name_base = rgb_name_base + "_pred"
    npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
    if os.path.exists(npy_save_path):
        logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
    np.save(npy_save_path, depth_pred)

    # Save as 16-bit uint png
    depth_to_save = (depth_pred * 65535.0).astype(np.uint16)
    png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
    if os.path.exists(png_save_path):
        logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
    Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

    # Colorize
    colored_save_path = os.path.join(
        output_dir_color, f"{pred_name_base}_colored.png"
    )
    if os.path.exists(colored_save_path):
        logging.warning(
            f"Existing file: '{colored_save_path}' will be overwritten"
        )
    depth_colored.save(colored_save_path)