import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    a = torch.tensor([1.0, 2.0], device=device)
    print(a)
else:
    print("CUDA is not available")


def latents_to_png(self, latents):
    decoded_latent = self.decode_latents(latents)
    decoded_latent = decoded_latent.squeeze()
    decoded_latent = decoded_latent.permute(1,2,0) #[H,W,C]
    return decoded_latent

from PIL import Image
def save_tensor_as_png(tensor, filename):
    """
    Saves a tensor of shape [H, W, C] where C=3 (for RGB images) to a PNG file.
    
    Args:
    tensor (torch.Tensor): Input tensor with shape [256, 256, 3].
    filename (str): Path to save the PNG image.
    
    Returns:
    None
    """
    
    tensor = tensor.squeeze()
    tensor = tensor.permute(1,2,0)

    # Ensure the tensor is on CPU and convert to PIL image
    if tensor.is_cuda:
        tensor = tensor.cpu()  # Move tensor to CPU if it's on GPU

    # Normalize the tensor to 0-255 and convert to 'uint8'
    if tensor.max() <= 1.0:
        tensor = tensor.mul(255).byte()  # Scale to 0-255 if max is 1.0 or less
    elif tensor.dtype != torch.uint8:
        tensor = tensor.byte()  # Convert to uint8 if not already
    
    # Convert to PIL Image (assuming tensor is in HWC format and 'uint8')
    image = Image.fromarray(tensor.numpy())
    
    # Save the image as a PNG file
    image.save(filename, 'PNG')
    print(f'Image saved as {filename}')