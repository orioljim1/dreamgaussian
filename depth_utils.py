import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize_depth(depth):
    return (depth-depth.min())/(depth.max()-depth.min())

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def nearMean_map(array, mask, kernelsize=3):
    """ array: (H,W) / mask: (H,W) """
    cnt_map = torch.ones_like(array)

    nearMean_map = conv((array * mask)[None,None])
    cnt_map = conv((cnt_map * mask)[None,None])
    nearMean_map = (nearMean_map / (cnt_map+1e-8)).squeeze()
        
    return nearMean_map

def image2canny(image, thres1, thres2, isEdge1=True):
    """ image: (H, W, 3)"""
    canny_mask = torch.from_numpy(cv2.Canny((image.detach().cpu().numpy()*255.).astype(np.uint8), thres1, thres2)/255.)
    if not isEdge1:
        canny_mask = 1. - canny_mask
    return canny_mask.float()


def optimize_depth(source, target, mask, depth_weight, prune_ratio=0.001):
    """
    Arguments
    =========
    source: np.array(h,w)
    target: np.array(h,w)
    mask: np.array(h,w):
        array of [True if valid pointcloud is visible.]
    depth_weight: np.array(h,w):
        weight array at loss.
    Returns
    =======
    refined_source: np.array(h,w)
        literally "refined" source.
    loss: float
    """
    source = torch.from_numpy(source).cuda()
    target = torch.from_numpy(target).cuda()
    mask = torch.from_numpy(mask).cuda()
    depth_weight = torch.from_numpy(depth_weight).cuda()

    # Prune some depths considered "outlier"     
    with torch.no_grad():
        target_depth_sorted = target[target>1e-7].sort().values
        min_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*prune_ratio)]
        max_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*(1.0-prune_ratio))]

        mask2 = target > min_prune_threshold
        mask3 = target < max_prune_threshold
        mask = torch.logical_and( torch.logical_and(mask, mask2), mask3)

    source_masked = source[mask]
    target_masked = target[mask]
    depth_weight_masked = depth_weight[mask]
    # tmin, tmax = target_masked.min(), target_masked.max()

    # # Normalize
    # target_masked = target_masked - tmin 
    # target_masked = target_masked / (tmax-tmin)

    scale = torch.ones(1).cuda().requires_grad_(True)
    shift = (torch.ones(1) * 0.5).cuda().requires_grad_(True)

    optimizer = torch.optim.Adam(params=[scale, shift], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))
    loss = torch.ones(1).cuda() * 1e5

    iteration = 1
    loss_prev = 1e6
    loss_ema = 0.0
    
    while abs(loss_ema - loss_prev) > 1e-5:
        source_hat = scale*source_masked + shift
        loss = torch.mean(((target_masked - source_hat)**2)*depth_weight_masked)

        # penalize depths not in [0,1]
        loss_hinge1 = loss_hinge2 = 0.0
        if (source_hat<=0.0).any():
            loss_hinge1 = 2.0*((source_hat[source_hat<=0.0])**2).mean()
        # if (source_hat>=1.0).any():
        #     loss_hinge2 = 0.3*((source_hat[source_hat>=1.0])**2).mean() 
        
        loss = loss + loss_hinge1 + loss_hinge2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        iteration+=1
        if iteration % 1000 == 0:
            print(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
            loss_prev = loss.item()
        loss_ema = loss.item() * 0.2 + loss_ema * 0.8

    loss = loss.item()
    print(f"loss ={loss:10.5f}")

    with torch.no_grad():
        refined_source = (scale*source + shift) 
    torch.cuda.empty_cache()
    return refined_source.cpu().numpy(), loss

def export_depth_image(tensor, path, H, W):

    buffer_image = tensor
    buffer_image = buffer_image.repeat(3, 1, 1)
    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

    buffer_image = F.interpolate(
        buffer_image.unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    buffer_image = (
        buffer_image.permute(1, 2, 0)
        .contiguous()
        .clamp(0, 1)
        .contiguous()
        .detach()
        .cpu()
        .numpy()
    )
    plt.imsave(path, buffer_image)


def clean_background(rgba_image, rgb_image):
    
    height, width, _ = rgba_image.shape
    cv2.imwrite("./rgba_image.png", rgba_image)
    cv2.imwrite("./rgb_image_pre.png", rgb_image)

    # Iterate over all pixels
    for y in range(height):
        for x in range(width):
            r, g, b, a = rgba_image[y, x]  # Extract RGBA values
            # Set RGB to black if alpha is 0
            print(a)
            if a == 0:
                rgb_image[y, x] = 0 # Set RGB pixel to black
    cv2.imwrite("./rgb_image_pos.png", rgb_image)
    return rgb_image     

'''
### depth supervised loss
depth = render_pkg["depth"]
if usedepth and viewpoint_cam.original_depth is not None:
    depth_mask = (viewpoint_cam.original_depth>0) # render_pkg["acc"][0]
    gt_maskeddepth = (viewpoint_cam.original_depth * depth_mask).cuda()
    if args.white_background: # for 360 datasets ...
        gt_maskeddepth = normalize_depth(gt_maskeddepth)
        depth = normalize_depth(depth)

    deploss = l1_loss(gt_maskeddepth, depth*depth_mask) * 0.5
    loss = loss + deploss

## depth regularization loss (canny)
        if usedepthReg and iteration>=0: 
            depth_mask = (depth>0).detach()
            nearDepthMean_map = nearMean_map(depth, viewpoint_cam.canny_mask*depth_mask, kernelsize=3)
            loss = loss + l2_loss(nearDepthMean_map, depth*depth_mask) * 1.0
 
self.canny_mask = image2canny(self.original_image.permute(1,2,0), 50, 150, isEdge1=False).detach().to(self.data_device)

'''