import inspect
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from diffusers import ModelMixin
from tqdm import tqdm
import torch.nn as nn

from .pvn_model import PVNModel
from einops.layers.torch import Rearrange
import numpy as np
import sys
import os
sys.path.append('.')
from eeg_data_process.extract_eeg_feature import VideoImageEEGClassifyColor3

def get_custom_betas(beta_start: float, beta_end: float, warmup_frac: float = 0.3, num_train_timesteps: int = 1000):
    """Custom beta schedule"""
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    warmup_frac = 0.3
    warmup_time = int(num_train_timesteps * warmup_frac)
    warmup_steps = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    warmup_time = min(warmup_time, num_train_timesteps)
    betas[:warmup_time] = warmup_steps[:warmup_time]
    return betas

class EEGTo3DDiffusionModel(ModelMixin):
    
    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        point_cloud_model_embed_dim: int,
        in_channels:int,
        out_channels=3,
        sub='sub13',
        generate_type='color',
        retri_pretrain_model='',
        **kwargs,  # projection arguments
    ):
        super().__init__(**kwargs)

        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {}
        if beta_schedule == 'custom':
            scheduler_kwargs.update(dict(trained_betas=get_custom_betas(beta_start=beta_start, beta_end=beta_end)))
        else:
            scheduler_kwargs.update(dict(beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule))
        self.schedulers_map = {
            'ddpm': DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            'ddim': DDIMScheduler(**scheduler_kwargs, clip_sample=False), 
            'pndm': PNDMScheduler(**scheduler_kwargs), 
        }
        self.scheduler = self.schedulers_map['ddpm']  # this can be changed for inference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.video_clip_mlp = nn.Linear(1024, 64)
        # self.point_clip_mlp = nn.Linear(768, 64)
        self.generate_type = generate_type
        # self.meta_eeg_video = VideoImageEEG(num_channels=64, sequence_length=600, sequence_length2=250, num_latents=1024)
        ss_dir = {
            'color': 'best-color.pth',
            'shape': 'best-retri.pth'
        }        
        self.meta_eeg_video = VideoImageEEGClassifyColor3(num_channels=64, sequence_length=600, sequence_length2=250, num_latents=1024)
        if os.path.exists(f"{retri_pretrain_model}/{ss_dir[generate_type]}"):
            self.meta_eeg_video.load_state_dict(torch.load(f"{retri_pretrain_model}/{ss_dir[generate_type]}", map_location='cpu'))
        else:
            print('not load!')
        for name, param in self.meta_eeg_video.named_parameters():
            param.requires_grad = False
        # for name, param in self.meta_eeg_point.named_parameters():
        #     param.requires_grad = False
        
        self.mse_loss_fn = nn.MSELoss()
        self.alpha = 0.99

        # Create point cloud model for processing point cloud at each diffusion step
        self.point_cloud_model = PVNModel(
            embed_dim=point_cloud_model_embed_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )

    def compute_eeg_loss(self, pc, eeg_data, eeg_data2, shape_c=None, fea_list=None, labels=None):
        x_0 = pc
        B, N, D = x_0.shape
        if shape_c is not None:
            noise_std = 0.02
            if np.random.random() > 0.5:
                shape_c = shape_c + torch.randn_like(shape_c) * noise_std
            shape_condition = shape_c
        # Sample random noise
        noise = torch.randn_like(x_0)

        # Sample random timesteps for each point_cloud
        timestep = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=self.device, dtype=torch.long)

        # Add noise to points
        x_t = self.scheduler.add_noise(x_0, noise, timestep)
        if shape_c is not None:
            x_t_input = [shape_condition, x_t]
        else: 
            x_t_input = [x_t]
        # import pdb;pdb.set_trace()

        eeg_features1, shape_result, eeg_features2, color_result = self.meta_eeg_video(eeg_data, eeg_data2)
        
        color_result_soft = color_result.softmax(1)
        
        if shape_c is not None:
            c1_features = eeg_features2
        else:
            c1_features = eeg_features1
        
        
        # import pdb;pdb.set_trace()
        # c1_features = self.video_clip_mlp(c1).unsqueeze(1).expand(-1, N, -1)
        noise_std2 = 0.004
        if np.random.random() > 0.5:
            c1_features = c1_features + torch.randn_like(c1_features) * noise_std2
        c1_features = c1_features.unsqueeze(1).expand(-1, N, -1)
        
        x_t_input.append(c1_features)
        if shape_c is not None:
            x_t_input.append(color_result_soft.unsqueeze(1).expand(-1, N, -1))
        x_t_input = torch.cat(x_t_input, dim=2)
        # Forward
        noise_pred = self.point_cloud_model(x_t_input, timestep)
        if noise_pred.shape[-1] == 6:
            noise_pred = noise_pred[..., 3:]
        # Loss
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def get_acc(self, logit_scale, eeg_features, img_features_all, labels):
        logits_img = logit_scale * eeg_features @ img_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1) # (n_batch, ) \in {0, 1, ..., n_cls-1}

        total = predicted.shape[0]
        correct = (predicted == labels).sum().item()
        return correct, total, (predicted == labels)
    
    def retrieval_test(self, eeg_data, eeg_data2, fea_list, labels):
        eeg_features1, shape_result, eeg_features2, color_result = self.meta_eeg_video(eeg_data, eeg_data2)
        if self.generate_type == 'color':
            result = color_result
            color_result_soft = result.softmax(1)
            predicted = torch.argmax(color_result_soft, dim=1)
            total = predicted.shape[0]
            # print(predicted)
            correct = (predicted == labels).sum().item()
            acc_list = (predicted == labels)
        elif self.generate_type == 'shape':
            result = shape_result
            correct, total, acc_list = self.get_acc(self.meta_eeg_video.logit_scale, eeg_features1, fea_list['video_features_all'].to(eeg_features1.device), labels)
        return (correct, total, acc_list)

    @torch.no_grad()
    def generate_eeg(self, num_points, eeg_data, eeg_data2, shape_c=None, fea_list=None, labels=None,
        scheduler: Optional[str] = 'ddpm',
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0, 
        return_sample_every_n_steps: int = -1,
        disable_tqdm: bool = False,
    ):

        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

        # Get the size of the noise
        N = num_points
        B = eeg_data.shape[0]
        D = 3
        device = eeg_data.device
        
        # Sample noise
        x_t = torch.randn(B, N, D, device=device)

        # Set timesteps
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}
        # import pdb;pdb.set_trace()
        # eeg_video_fea, c1 = self.meta_eeg_video(eeg_data, 0)
        # eeg_point_fea, c2 = self.meta_eeg_point(eeg_data, 0)
        video_acc_count, total, acc_list = self.retrieval_test(eeg_data, eeg_data2, fea_list, labels)
        eeg_features1, shape_result, eeg_features2, color_result = self.meta_eeg_video(eeg_data, eeg_data2)
        
        color_result_soft = color_result.softmax(1)
        
        if shape_c is not None:
            c1 = eeg_features2
        else:
            c1 = eeg_features1
        
        
        all_outputs = [x_t]
        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(scheduler.timesteps.to(device), desc=f'Sampling ({x_t.shape})', disable=disable_tqdm)
        for i, t in enumerate(progress_bar):
            if shape_c is not None:
                x_t_input = [shape_c, x_t]
            else: 
                x_t_input = [x_t]
            # import pdb;pdb.set_trace()
            # c1_features = self.video_clip_mlp(c1).unsqueeze(1).expand(-1, N, -1)
            c1_features = c1.unsqueeze(1).expand(-1, N, -1)
            x_t_input.append(c1_features)
            if shape_c is not None:
                x_t_input.append(color_result_soft.unsqueeze(1).expand(-1, N, -1))
            
            x_t_input = torch.cat(x_t_input, dim=2)

            # Forward
            noise_pred = self.point_cloud_model(x_t_input, t.reshape(1).expand(B))
            
            if noise_pred.shape[-1] == 6:
                noise_pred = noise_pred[..., 3:]
                
            # Step
            x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample
            
            # Append to output list if desired
            if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                all_outputs.append(x_t)
        output = x_t
        return output, (video_acc_count, total, acc_list)
    
    def forward(self, pc, eeg_data, eeg_data2, shape_c=None, mode: str = 'train', fea_list=None, labels=None, **kwargs):
        if mode == 'train':
            return self.compute_eeg_loss(pc, eeg_data, eeg_data2, shape_c, fea_list, labels) 
        elif mode == 'sample':
            return self.generate_eeg(8192, eeg_data, eeg_data2, shape_c, fea_list, labels, **kwargs) 
        elif mode == 'test_retrieval':
            return self.retrieval_test(eeg_data, eeg_data2, fea_list, labels) 
        else:
            raise NotImplementedError()
    
    