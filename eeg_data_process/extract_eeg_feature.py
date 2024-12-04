import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
import math
from torch.nn import functional as F
import numpy as np
import sys
sys.path.append('.')
from eeg_data_process.clip_loss import ClipLoss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=600):#600 1500
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        # self.pe = pe
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
        x = x + pe
        return x

class EEGAttention(nn.Module): ###时间维度上的attention
    def __init__(self, channel, d_model, nhead, max_len=600):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, src):
        src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)  # Change shape back to [batch_size, channel, time_length]

class ConvBlock(nn.Module):
    def __init__(self, num_channels, num_features):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        self.norm3 = nn.LayerNorm(num_features)
        self.residual_conv = nn.Conv1d(num_channels, num_features, kernel_size=1)

    def forward(self, x):
        # print(f'ConvBlock input shape: {x.shape}')
        residual = self.residual_conv(x)
        # residual = x
        # print(f'residual shape: {residual.shape}')
        
        x = F.gelu(self.conv1(x))
        x = self.norm1(x)
        # print(f'After first convolution shape: {x.shape}')
                
        x = F.gelu(self.conv2(x))
        x = self.norm2(x)
        # print(f'After second convolution shape: {x.shape}')
        
        x = F.gelu(self.conv3(x))
        x = self.norm3(x)
        # print(f'After third convolution shape: {x.shape}')
        
        x += residual
        # print(f'ConvBlock output shape: {x.shape}')
        return x

class MLPHead(nn.Module):
    def __init__(self, in_features, num_latents, dropout_rate=0.25):
        super(MLPHead, self).__init__()

        self.layer1 = nn.Sequential(
            Rearrange('B C L->B L C'),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_latents),
            nn.GELU(),
            nn.Dropout(dropout_rate), 
            Rearrange('B L C->B (C L)'),
        )
    def forward(self, x):
        # print(f'MLPHead input shape: {x.shape}')
        x = self.layer1(x)
        # print(f'After first layer of MLPHead shape: {x.shape}')
        return x

class CustomTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CustomTransformerLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, Q, K, V):
        # Q, K, V shape: (seq_length, batch_size, embed_dim)
        Q = Q.permute(2, 0, 1)
        K = K.permute(2, 0, 1)
        V = V.permute(2, 0, 1)
        attn_output, attn_output_weights = self.multihead_attn(Q, K, V)
        Q = self.norm1(Q + attn_output)
        ff_output = F.relu(self.linear1(Q))
        ff_output = self.dropout(ff_output)
        ff_output = self.linear2(ff_output)
        output = self.norm2(Q + ff_output)

        return output.permute(1, 2, 0)

class VideoImageEEGClassifyColor3(nn.Module):
    def __init__(self, num_channels, sequence_length, sequence_length2, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1, cls_num=72):
        super(VideoImageEEGClassifyColor3, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)

        self.static_attention = EEGAttention(num_channels, num_channels, nhead=1)
        self.static_linear = nn.Linear(sequence_length2, sequence_length2)
        self.dynamic_linear = nn.Linear(sequence_length, sequence_length2)

        self.dynamic_static = CustomTransformerLayer(num_channels, num_heads=1)

        self.conv_blocks = nn.Sequential(*[ConvBlock(num_channels, sequence_length2) for _ in range(num_blocks)],
                                         Rearrange('B C L->B L C'))
        self.linear_projection = nn.Sequential(
                                            Rearrange('B L C->B C L'),
                                            nn.Linear(sequence_length2, num_latents),
                                            Rearrange('B C L->B L C'))
        self.temporal_aggregation = nn.Linear(sequence_length2, 1)
        self.clip_head = MLPHead(num_latents, num_latents)
        self.class_head = nn.Linear(num_latents, 6)
        self.clip_head2 = MLPHead(num_latents, num_latents)
        self.class_head2 = nn.Linear(num_latents, cls_num)
        # self.mse_head = MLPHead(num_latents, num_latents)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.01))
        self.loss_func = ClipLoss()
        
    def forward(self, x, x2):
        # import pdb;pdb.set_trace()
        dyn = self.attention_model(x)
        dyn = self.dynamic_linear(dyn)
        stc = self.static_attention(x2)
        stc = self.static_linear(stc)
        x = self.dynamic_static(stc, dyn, dyn)
        # x = self.dynamic_static(dyn, stc, dyn)
        # stc_dyn = torch.cat([stc, dyn], dim=-1)
        # x = self.dynamic_static(stc_dyn)[..., 250:]

        x = self.conv_blocks(x)
        
        x = self.linear_projection(x)
        # print(f'After linear projection shape: {x.shape}')
        # import pdb;pdb.set_trace()
        x_tem = self.temporal_aggregation(x)
        # print(f'After temporal aggregation shape: {x.shape}')

        clip_out = self.clip_head2(x_tem)
        cls_result = self.class_head2(clip_out)
        clip_out2 = self.clip_head(x_tem)
        cls_result2 = self.class_head(clip_out2)

        return clip_out, cls_result, clip_out2, cls_result2

    