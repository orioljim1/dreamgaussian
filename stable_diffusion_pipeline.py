import random
import sys
import torch
from diffusers import DiffusionPipeline, TCDScheduler
from huggingface_hub import hf_hub_download

class StableDiffusionPipeline:
    def __init__(self, base_model_id="stabilityai/stable-diffusion-xl-base-1.0", repo_name="ByteDance/Hyper-SD", num_inference_steps=12, device="cuda"):
        self.base_model_id = base_model_id
        self.repo_name = repo_name
        self.num_inference_steps = num_inference_steps
        self.device = device

        plural = "s" if num_inference_steps > 1 else ""
        self.ckpt_name = f"Hyper-SDXL-{num_inference_steps}step{plural}-CFG-lora.safetensors"

        self.pipe = DiffusionPipeline.from_pretrained(self.base_model_id, torch_dtype=torch.float16, variant="fp16").to(self.device)
        self.pipe.load_lora_weights(hf_hub_download(self.repo_name, self.ckpt_name))
        self.pipe.fuse_lora()
        self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)

    def generate_image(self, prompt, guidance_scale=5.0, eta=0.5, save_path="output.jpg"):
        seed = random.randint(0, sys.maxsize)
        generator = torch.Generator(self.device).manual_seed(seed)

        images = self.pipe(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            generator=generator
        ).images

        print(f"Prompt:\t{prompt}\nSeed:\t{seed}")
        
        images[0].save(save_path)
        images[0].show()

        return images[0]

# Example usage:
if __name__ == "__main__":
    sd_pipeline = StableDiffusionPipeline()
    prompt = "A simple 3d render of an entirely fully visible dinosaur with a funny hat"
    sd_pipeline.generate_image(prompt)
