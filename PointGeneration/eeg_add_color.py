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

from .pvn_model import PVNModel, PointCloudTransformerModel
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
        in_channels:int,
        point_cloud_model_embed_dim=64,
        out_channels=3,
        point_cloud_model_layers=1,
        sub='sub10',
        model_type='',
        pretrain_model='',
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
        self.model_type = model_type
        self.scheduler = self.schedulers_map['ddpm']  # this can be changed for inference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.meta_eeg_video = VideoImageEEGClassifyColor3(num_channels=64, sequence_length=600, sequence_length2=250, num_latents=1024)
        if os.path.exists(f"{pretrain_model}/best-color.pth"):
            print('load!')
            self.meta_eeg_video.load_state_dict(torch.load(f"{pretrain_model}/best-color.pth", map_location='cpu'))
        else:
            print('not load!')
        
        for name, param in self.meta_eeg_video.named_parameters():
            param.requires_grad = False

        self.point_cloud_model = PointCloudTransformerModel(
            num_layers=point_cloud_model_layers,
            embed_dim=point_cloud_model_embed_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )

    def compute_loss(self, pc, c1, c2, shape_c=None, noise_std=0.0):
        # import pdb;pdb.set_trace()
        x_0 = pc
        if noise_std != 0 and np.random.random() > 0.5:
            shape_c = shape_c + torch.randn_like(shape_c) * noise_std
        x_t_input = [shape_c]
        N = pc.shape[1]
        # import pdb;pdb.set_trace()
        # c1_features, _ = self.meta_eeg_video(c1, c2)
        # import pdb;pdb.set_trace()
        if self.model_type == 'static':
            eeg_features1, shape_result, eeg_features2, color_result = self.meta_eeg_video(c2, 0)
        elif self.model_type == 'dynamic':
            eeg_features1, shape_result, eeg_features2, color_result = self.meta_eeg_video(c1, 0)
        else:
            eeg_features1, shape_result, eeg_features2, color_result = self.meta_eeg_video(c1, c2)
        
        color_result_soft = color_result.softmax(1)
        c1_features = eeg_features2

        noise_std2 = 0.004
        if np.random.random() > 0.5:
            c1_features = c1_features + torch.randn_like(c1_features) * noise_std2
            
        c1_features = torch.cat([c1_features, color_result_soft], dim=1)
        # c1_features = self.video_clip_mlp(c1).unsqueeze(1).expand(-1, N, -1)
        c1_features = c1_features.unsqueeze(1).expand(-1, N, -1)
        # c2_features = self.point_clip_mlp(c2).unsqueeze(1).expand(-1, N, -1)
        
        x_t_input.append(c1_features)
        # x_t_input.append(c2_features)
        x_t_input = torch.cat(x_t_input, dim=2)
        # import pdb;pdb.set_trace()
        # Forward
        pred_colors = self.point_cloud_model(x_t_input)
        # Loss
        loss = F.mse_loss(pred_colors, x_0)
        return loss
    
    @torch.no_grad()
    def generate(self, num_points, c1, c2, shape_c=None):
        x_t_input = [shape_c]
        # Get the size of the noise
        N = num_points

        # c1_features, _ = self.meta_eeg_video(c1, c2)
        if self.model_type == 'static':
            eeg_features1, shape_result, eeg_features2, color_result = self.meta_eeg_video(c2, 0)
        elif self.model_type == 'dynamic':
            eeg_features1, shape_result, eeg_features2, color_result = self.meta_eeg_video(c1, 0)
        else:
            eeg_features1, shape_result, eeg_features2, color_result = self.meta_eeg_video(c1, c2)
        
        color_result_soft = color_result.softmax(1)
        c1_features = eeg_features2
        
        c1_features = torch.cat([c1_features, color_result_soft], dim=1)
        c1_features = c1_features.unsqueeze(1).expand(-1, N, -1)

        x_t_input.append(c1_features)
        x_t_input = torch.cat(x_t_input, dim=2)
        pred_colors = self.point_cloud_model(x_t_input)
        return pred_colors

    def retrieval_test(self, eeg_data, eeg_data2, fea_list, labels):
        if self.model_type == 'static':
            eeg_features1, shape_result, eeg_features2, color_result = self.meta_eeg_video(eeg_data2, 0)
        elif self.model_type == 'dynamic':
            eeg_features1, shape_result, eeg_features2, color_result = self.meta_eeg_video(eeg_data, 0)
        else:
            eeg_features1, shape_result, eeg_features2, color_result = self.meta_eeg_video(eeg_data, eeg_data2)
        color_result_soft = color_result.softmax(1)
        predicted = torch.argmax(color_result_soft, dim=1)
        total = predicted.shape[0]
        # print(predicted)
        correct = (predicted == labels).sum().item()
        acc_list = (predicted == labels)
        return (correct, total, acc_list)

    def forward(self, pc, c1, c2, shape_c=None, mode: str = 'train', noise_std=0.0, fea_list=None, labels=None, **kwargs):
        if mode == 'train':
            return self.compute_loss(pc, c1, c2, shape_c, noise_std) 
        elif mode == 'sample':
            return self.generate(8192, c1, c2, shape_c) 
        elif mode == 'test_retrieval':
            return self.retrieval_test(c1, c2, fea_list, labels) 
        else:
            raise NotImplementedError()
    
    