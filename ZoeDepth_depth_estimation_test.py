import os
import sys
import torch
#from zoedepth.utils.misc import pil_to_batched_tensor
from ZoeDepth.zoedepth.utils.misc import pil_to_batched_tensor
from ZoeDepth.zoedepth.utils.misc import save_raw_16bit
from ZoeDepth.zoedepth.utils.misc import colorize
from PIL import Image

model_zoe = None
if model_zoe is None:
    model_zoe = torch.hub.load("./ZoeDepth", "ZoeD_N", source="local", pretrained=True).to('cuda')


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("*Cuda avaliable* ", torch.cuda.is_available())
zoe = model_zoe.to(DEVICE)

# Local file
image = Image.open("./data/charmander.png").convert("RGB")  # load
#print("Image:", image)
depth_numpy = zoe.infer_pil(image)  # as numpy

#depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

#depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor


# Tensor 
X = pil_to_batched_tensor(image).to(DEVICE)
depth_tensor = zoe.infer(X)
#source_depth = model_zoe.infer_pil(image.convert("RGB"))
#target=depthmap.copy()

depth = zoe.infer_pil(image)

# Save raw
fpath = "./test_path/charmander_output.png"
save_raw_16bit(depth_numpy, fpath)

# Colorize output

colored = colorize(depth_numpy)

# save colored output
fpath_colored = "./test_path/charmander_output_colored.png"
Image.fromarray(colored).save(fpath_colored)