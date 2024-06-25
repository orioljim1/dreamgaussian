from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
import torch
from PIL import Image

# Path to your safetensors model
safetensors_model_path = "./models/sd3_medium_incl_clips_t5xxlfp8_2.safetensors"

# Load the safetensors model
state_dict = load_file(safetensors_model_path)

# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Load the state dict into the model
pipe.unet.load_state_dict(state_dict, strict=False)

# Move the model to GPU (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

image = pipe(
    "A simple 3d render of a entirely fully visible dinosaur with a funny hat",
    negative_prompt="bad quality, poor quality, disfigured, jpg, bad anatomy, missing limbs, missing fingers",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]
image.save("generated_image2.png")

# Display the image
image.show()
'''
# Define your prompt
prompt = "a high resulusion female character with long, flowing hair that appears to be made of ethereal, swirling patterns resembling the Northern Lights or Aurora Borealis. The background is dominated by deep blues and purples, creating a mysterious and dramatic atmosphere. The character's face is serene, with pale skin and striking features. She wears a dark-colored outfit with subtle patterns. The overall style of the artwork is reminiscent of fantasy or supernatural genres"

# Generate an image
with torch.no_grad():
    image = pipe(prompt).images[0]

# Save the image
image.save("generated_image2.png")

# Display the image
image.show()
'''