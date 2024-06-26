import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F

import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize
from PIL import Image
from mg import get_depth
import json
import sys
import io

from depth_utils  import lpips, psnr, ssim, normalize_depth, normalize_depth5, l1_loss, l2_loss, nearMean_map, image2canny, optimize_depth, export_depth_image, normalize_depth_map, clean_background, save_tensor_as_png, save_tensor_as_png2, load_camera_poses
from stable_diffusion_pipeline import StableDiffusionPipeline


class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "42"
        self.seed_everything()

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        #depth
        self.model_zoe = None
        self.original_depth = None
        self.canny_mask = None
        self.input_og_depth_torch = None
        self.input_image_rgba = None
        self.losses_data = {"rgb": [], "alpha": [], "depth": [], "canny":[]}
        self.breakk = False

        #Text-to-Image
        self.text_to_img = None

        self.saved_cameras = load_camera_poses("./camera_poses")

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input) #load image here.
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

        # override if provide a checkpoint
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load)            
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        print("We're seeding "*10)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):
        
        print(f"[INFO] Prepare_train ")

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # default camera
        if self.opt.mvdream or self.opt.imagedream:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(self.opt.elevation, 90, self.opt.radius)
        else:
            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None
        #self.enable_zero123 = False #to deactivate sds

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            elif self.opt.imagedream:
                print(f"[INFO] loading ImageDream...")
                from guidance.imagedream_utils import ImageDream
                self.guidance_sd = ImageDream(self.device)
                print(f"[INFO] loaded ImageDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] self.enable_zero123", self.enable_zero123)
            print(f"[INFO] self.opt.stable_zero123", self.opt.stable_zero123)
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            if self.opt.stable_zero123:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/stable-zero123-diffusers')
            else:
                self.guidance_zero123 = Zero123(self.device, model_key='ashawkey/zero123-xl-diffusers')
            print(f"[INFO] loaded zero123!")

        # input image
        if self.input_img is not None or self.opt.text_to_img:

            if self.opt.text_to_img:
                #infer image with stable diffusion
                self.sd_pipeline = StableDiffusionPipeline()
                #prompt = "A simple 3d render of an entirely fully visible dinosaur with a funny hat"
                self.input_img = self.sd_pipeline.generate_image(self.opt.prompt)

                if self.bg_remover is None:
                    self.bg_remover = rembg.new_session()
                    self.input_img = rembg.remove(self.input_img, session=self.bg_remover)
                    self.input_img = np.array(self.input_img)
                    #self.input_img = np.array(self.input_img).astype(np.float32)

                    self.input_img = cv2.resize(self.input_img, (self.W, self.H), interpolation=cv2.INTER_AREA)                
                    self.input_img = self.input_img.astype(np.float32) / 255.0
                    self.input_image_rgba = self.input_img.copy()
                    self.input_mask = self.input_img[..., 3:]
                    # white bg
                    self.input_img = self.input_img[..., :3] * self.input_mask + (1 - self.input_mask)

                    #self.input_img = self.input_img.float()
                               

            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            #we need tensor
            self.canny_mask = self.input_img_torch.clamp(0.0, 1.0).to(self.device).squeeze()
            self.canny_mask = image2canny(self.canny_mask.permute(1,2,0), 50, 150, isEdge1=False).detach().to(self.device)
            save_tensor_as_png(self.canny_mask, "./iter_canny.png")

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        # prepare embeddings
        with torch.no_grad():

            if self.enable_sd:
                if self.opt.imagedream:
                    self.guidance_sd.get_image_text_embeds(self.input_img_torch, [self.prompt], [self.negative_prompt])
                else:
                    self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                print("reaching self.enable_zero123", self.enable_zero123)
                self.guidance_zero123.get_img_embeds(self.input_img_torch)

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0

            ### known view
            if self.input_img_torch is not None and not self.opt.imagedream:
                cur_cam = self.fixed_cam
                out = self.renderer.render(cur_cam)

                # rgb loss
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                rgb_loss = 10000 * (step_ratio if self.opt.warmup_rgb_loss else 1) * F.mse_loss(image, self.input_img_torch)
                loss = loss + rgb_loss#+ 10000 * (step_ratio if self.opt.warmup_rgb_loss else 1) * F.mse_loss(image, self.input_img_torch)
                self.losses_data["rgb"].append(f"{rgb_loss.item():.4f}")

                # mask loss
                mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                alpha_loss = 1000 * (step_ratio if self.opt.warmup_rgb_loss else 1) * F.mse_loss(mask, self.input_mask_torch)
                loss = loss + alpha_loss #1000 * (step_ratio if self.opt.warmup_rgb_loss else 1) * F.mse_loss(mask, self.input_mask_torch)
                self.losses_data["alpha"].append(f"{alpha_loss.item():.4f}")
                
                depth = out["depth"]#.unsqueeze(0)
                #depth = normalize_depth(depth)
                dloss ={ "item":0.0}
                if(self.step >= 0 and self.opt.use_depth and False):
                    #print("using depth!")
                    #depth loss
                    dloss =  self.opt.dloss_lambda * (step_ratio if self.opt.warmup_rgb_loss else 1) * F.mse_loss(depth, self.input_og_depth_torch)
                    #loss = loss + dloss
                    #print("Dloss: ", f"{dloss.item():.4f}" )
                    self.losses_data["depth"].append(f"{dloss.item():.4f}")
                else:
                    if(self.step == 400):
                        a = 2
                    #self.losses_data["depth"].append(0.0)

                ### depth supervised loss
                
                usedepth = True
                if usedepth and self.input_og_depth_torch is not None:
                    depth_mask = (self.input_og_depth_torch>0) 
                    gt_maskeddepth = (self.input_og_depth_torch * depth_mask)
                    deploss =  self.opt.dloss_lambda * l1_loss(gt_maskeddepth, depth*depth_mask)
                    self.losses_data["depth"].append(f"{deploss.item():.4f}")

                    loss = loss + deploss

                ## depth regularization loss (canny)
                usedepthReg = True
                if usedepthReg and self.step>=0 and self.opt.canny_lambda > 0: 
                    depth_mask = (depth>0).detach()
                    nearDepthMean_map = nearMean_map(depth.squeeze(), (self.canny_mask*depth_mask).squeeze(), kernelsize=3)
                    canny_loss = self.opt.canny_lambda * l2_loss(nearDepthMean_map, depth*depth_mask) * 1000.0
                    loss = loss + canny_loss
                    self.losses_data["canny"].append(f"{canny_loss.item():.4f}")


            ### novel view (manual batch)
            render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
            min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)

            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                poses.append(pose)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam, bg_color=bg_color)

                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)

                # enable mvdream training
                if self.opt.mvdream or self.opt.imagedream:
                    for view_i in range(1, 4):
                        pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                        poses.append(pose_i)

                        cur_cam_i = MiniCam(pose_i, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                        # bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
                        out_i = self.renderer.render(cur_cam_i, bg_color=bg_color)

                        image = out_i["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        images.append(image)
                    
            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

            # import kiui
            # print(hor, ver)
            # kiui.vis.plot_image(images)

            # guidance loss
            if self.enable_sd:
                if self.opt.mvdream or self.opt.imagedream:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio=step_ratio if self.opt.anneal_timestep else None)
                else:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio=step_ratio if self.opt.anneal_timestep else None)

            if self.enable_zero123:
                
                loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio=step_ratio if self.opt.anneal_timestep else None, default_elevation=self.opt.elevation)
            
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # densify and prune
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter) #this breaks if no sds is used D:

                if self.step % self.opt.densification_interval == 0:
                    self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
                
                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

        # dynamic train steps (no need for now)
        # max allowed train time per-frame is 500 ms
        # full_t = t / self.train_steps * 16
        # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        #     self.train_steps = train_steps

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if self.breakk == True:
            self.breakk = False
            self.save_model(mode='geo+tex')
            #self.compute_metrics()

        # should update image
        if self.need_update:
            # render image

            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            out = self.renderer.render(cur_cam, self.gaussain_scale_factor)

            buffer_image = out[self.mode]  # [3, H, W]

            if self.mode in ['depth', 'alpha', "depth_2"]:
                if self.mode == 'depth': buffer_image = buffer_image.squeeze(0)
                buffer_image = buffer_image.repeat(3, 1, 1)
                if self.mode == 'depth' or self.mode == "depth_2":
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            buffer_image = F.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            self.buffer_image = (
                buffer_image.permute(1, 2, 0)
                .contiguous()
                .clamp(0, 1)
                .contiguous()
                .detach()
                .cpu()
                .numpy()
            )

            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio)
                    + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!



    def render_img(self, camera_values):
        cur_cam = MiniCam(
            np.array(camera_values['pose'], dtype= np.float32),
            self.W,
            self.H,
            np.float64(camera_values['fovy']),
            np.float64(camera_values["fovx"]),
            np.float64(camera_values["near"]),
            np.float64(camera_values["far"]),
        )

        out = self.renderer.render(cur_cam, self.gaussain_scale_factor)

        buffer_image = out[self.mode]  # [3, H, W]

        if self.mode in ['depth', 'alpha', "depth_2"]:
            buffer_image = buffer_image.repeat(3, 1, 1)
            if self.mode == 'depth' or self.mode == "depth_2":
                buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

        buffer_image = F.interpolate(
            buffer_image.unsqueeze(0),
            size=(self.H, self.W),
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
        return buffer_image
    

    def render_multiple_images(self, save_path ):
        
        save_path = save_path +'/renders'
        os.makedirs(save_path, exist_ok=True)

        for cam_pose in self.saved_cameras:
            img = self.render_img(cam_pose)
            files = os.listdir(save_path)
            next_number = len(files) + 1
            img = (img * 255).astype(np.uint8)
            img = img[..., ::-1].copy() #bgr to rgb
            cv2.imwrite(f'{save_path}/camera_pose{next_number}.png', img)

    def compute_metrics(self):
        
        '''
        camera_values = self.saved_cameras[0]

        cur_cam = MiniCam(
            np.array(camera_values['pose'], dtype= np.float32),
            self.W,
            self.H,
            np.float64(camera_values['fovy']),
            np.float64(camera_values["fovx"]),
            np.float64(camera_values["near"]),
            np.float64(camera_values["far"]),
        )
        '''
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        fixed_cam = MiniCam(
            pose,
            self.input_img.shape[0],
            self.input_img.shape[1],
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )
        out = self.renderer.render(fixed_cam, self.gaussain_scale_factor)

        image = out[self.mode]  # [3, H, W]

        input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).to(self.device)
        #input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
        psnr_test = psnr(image, input_img_torch ).mean().item()
        ssim_test = ssim(image, input_img_torch).mean().item()
        lpips_test = lpips(image, input_img_torch).mean().item()
        print([psnr_test, ssim_test, lpips_test])

    def predict_depth_Zoe(self, img):
        from ZoeDepth.zoedepth.utils.misc import colorize

        #depth estimation 
        if self.model_zoe is None:
            self.model_zoe = torch.hub.load("./ZoeDepth", "ZoeD_NK", source="local", pretrained=True).to('cuda')
        
        depth_pred =  self.model_zoe.infer_pil(img)
        #depth_pred = (depth_pred * 255).astype(np.uint8)
        #depth_pred = depth_pred[..., ::-1].copy() #bgr to rgb
        clean_img = clean_background(self.input_image_rgba, depth_pred)
        #cv2.imwrite("./clean_image.png", clean_img)
        return clean_img
        ''' 
        # Colorize output
        print("[Info] Coloring depth")
        colored = colorize(self.source_depth)

        # save colored output
        fpath_colored = "./test_path/test_colored2.png"
        Image.fromarray(colored).save(fpath_colored)
        '''

    def predict_depth_Marigold(self, img):
        rgb_path = "./data/charmander_rgba.png"

        depth_pred, depth_colored = get_depth(img)
        return depth_pred
        '''
        # Save as npy
        output_dir = "./test_path/output/new"
        rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
        pred_name_base = rgb_name_base + "_pred"
        # Output directories
        output_dir_color = os.path.join(output_dir, "depth_colored")
        output_dir_tif = os.path.join(output_dir, "depth_bw")
        output_dir_npy = os.path.join(output_dir, "depth_npy")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir_color, exist_ok=True)
        os.makedirs(output_dir_tif, exist_ok=True)
        os.makedirs(output_dir_npy, exist_ok=True)
        # Save as 16-bit uint png
        depth_to_save = (depth_pred * 65535.0).astype(np.uint16)
        png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")

        Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

        # Colorize
        colored_save_path = os.path.join(
            output_dir_color, f"{pred_name_base}_colored.png"
        )
        depth_colored.save(colored_save_path)
        '''
    
    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        #cv2.imwrite("./test_export33.png", img)
       
        img = img.astype(np.float32) / 255.0
        #depth_marigold = self.predict_depth_Marigold(img)
        self.input_image_rgba = img.copy()
        
        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)

        
        print("[Info] we're reaching img save point")
        depth_zoe = self.predict_depth_Zoe(self.input_img)
        
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

       
        #depht used 
        self.original_depth = depth_zoe
        self.input_og_depth_torch = torch.from_numpy(depth_zoe).unsqueeze(0).unsqueeze(0).to(self.device)
        if (self.input_og_depth_torch.max() > 1):
            self.input_og_depth_torch = torch.clamp(self.input_og_depth_torch - 1, min=0)

        self.input_og_depth_torch = F.interpolate(self.input_og_depth_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
        self.input_og_depth_torch = normalize_depth5(self.input_og_depth_torch)

        # Save the processed image
        # Ensure image is converted back to 8-bit per channel for saving
        

        # load prompt
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f'[INFO] load prompt from {file_prompt}...')
            with open(file_prompt, "r") as f:
                self.prompt = f.read().strip()

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024):
        os.makedirs(self.opt.outdir, exist_ok=True)
        original_stdout = sys.stdout
        # Reconfigure sys.stdout to use UTF-8 encoding
        sys.stdout = io.TextIOWrapper(original_stdout.buffer, encoding='utf-8')

        if mode == 'geo':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.ply')
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)

        elif mode == 'geo+tex':
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_mesh.' + self.opt.mesh_format)
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                
                cur_out = self.renderer.render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]

                # enhance texture quality with zero123 [not working well]
                # if self.opt.guidance_model == 'zero123':
                #     rgbs = self.guidance.refine(rgbs, [ver], [hor], [0])
                    # import kiui
                    # kiui.vis.plot_image(rgbs)
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + '_model.ply')
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")
                
                # overlay stuff
                with dpg.group(horizontal=True):

                    def callback_toggle_overlay_input_img(sender, app_data):
                        self.overlay_input_img = not self.overlay_input_img
                        self.need_update = True

                    dpg.add_checkbox(
                        label="overlay image",
                        default_value=self.overlay_input_img,
                        callback=callback_toggle_overlay_input_img,
                    )

                    def callback_set_overlay_input_img_ratio(sender, app_data):
                        self.overlay_input_img_ratio = app_data
                        self.need_update = True

                    dpg.add_slider_float(
                        label="ratio",
                        min_value=0,
                        max_value=1,
                        format="%.1f",
                        default_value=self.overlay_input_img_ratio,
                        callback=callback_set_overlay_input_img_ratio,
                    )

                # prompt stuff
            
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )

                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    def callback_save(sender, app_data, user_data):
                        self.save_model(mode=user_data)

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=callback_save,
                        user_data='model',
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_button(
                        label="geo",
                        tag="_button_save_mesh",
                        callback=callback_save,
                        user_data='geo',
                    )
                    dpg.bind_item_theme("_button_save_mesh", theme_button)

                    dpg.add_button(
                        label="geo+tex",
                        tag="_button_save_mesh_with_tex",
                        callback=callback_save,
                        user_data='geo+tex',
                    )
                    dpg.bind_item_theme("_button_save_mesh_with_tex", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                            with open('./metrics_data/losses_depth_map_fixed.json', 'w') as f:
                                json.dump(self.losses_data, f)
                            print("result saved at: ","./metrics_data/losses_depth_map_fixed.json" )
                        else:
                            self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")
                            

                    # dpg.add_button(
                    #     label="init", tag="_button_init", callback=self.prepare_train
                    # )
                    # dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha", "depth_2"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

                def callback_set_gaussain_scale(sender, app_data):
                    self.gaussain_scale_factor = app_data
                    self.need_update = True

                dpg.add_slider_float(
                    label="gaussain scale",
                    min_value=0,
                    max_value=1,
                    format="%.2f",
                    default_value=self.gaussain_scale_factor,
                    callback=callback_set_gaussain_scale,
                )
            
            
            with dpg.collapsing_header(label="Camera stuff", default_open=True):
                def callback_campera_pose(sender, app_data):
                    camera_parameters = {
                    "pose": self.cam.pose,

                    "fovy":self.cam.fovy,
                    "fovx":self.cam.fovx,
                    "near":self.cam.near,
                    "far": self.cam.far
                    }

                    def default_converter(o):
                        if isinstance(o, np.ndarray):
                            return o.tolist()
                    print(camera_parameters)
                    files = os.listdir("./camera_poses")
                    next_number = len(files) + 1
                    with open(f'./camera_poses/camera_pose{next_number}.json', 'w') as f:
                        json.dump(camera_parameters, f, default=default_converter)
                    
                dpg.add_button(
                        label="Save camera pose", tag="scp", callback=callback_campera_pose
                    )
                dpg.add_button(
                        label="Render views!", tag="rvs", callback=self.render_multiple_images("./saved_renders")
                )
                dpg.add_button(
                        label="Compute metrics!", tag="cm", callback=self.compute_metrics
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
            with open(self.opt.outdir+'/losses.json', 'w') as f:
                json.dump(self.losses_data, f)
            print("Json saveda at:", self.opt.outdir+'/losses.json')
            # do a last prune
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        # save
        self.render_multiple_images(self.opt.outdir)
        self.save_model(mode='model')
        self.compute_metrics()
        self.save_model(mode='geo+tex')
        #Print metrics
        self.compute_metrics()
        #self.compute_metrics()
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters)
