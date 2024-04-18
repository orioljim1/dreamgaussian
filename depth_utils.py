import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import cv2
import numpy as np

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