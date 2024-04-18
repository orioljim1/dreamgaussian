# Marigold script to get depth values and colored image

import argparse
import logging
import os
from glob import glob
import cv2

import numpy as np
import torch
from PIL import Image

from Marigold.marigold import MarigoldPipeline

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    return device

def setup_pipeline(device, dtype=torch.float32, variant=None):
    pipe = MarigoldPipeline.from_pretrained(
        "prs-eth/marigold-lcm-v1-0", variant=variant, torch_dtype=dtype
    )
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass  # run without xformers

    return pipe.to(device)

def get_depth_prediction(input_image,pipe):
    with torch.no_grad():
        #input_image = Image.open(rgb_path)

        
        if isinstance(input_image,np.ndarray):
            scaled_data = (input_image * 255).astype(np.uint8)
            cv2.imwrite("./export_cv_pre.png", scaled_data)

            input_image = Image.fromarray(scaled_data)

            c = input_image.split()  # Split into individual channels
            # Merge the channels back but in the order of RGB
            input_image = Image.merge("RGB", (c[2], c[1], c[0]))
            input_image.save("test33_conversion.png")
        
        match_input_res = True
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
        return depth_pred, depth_colored

def get_depth(img):
    device = setup_device()
    pipe = setup_pipeline(device)
    return get_depth_prediction(img, pipe)