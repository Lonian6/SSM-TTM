import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.fft

# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model
# from timm.models.vision_transformer import _cfg
import math
import numpy as np
from mamba_ssm import Mamba, Mamba2
from mamba_ssm.ops.triton.layer_norm import RMSNorm

from inspect import isfunction
from math import ceil, floor, log, pi, log2
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
from packaging import version

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many
from torch import Tensor, einsum
# from torch.backends.cuda import sdp_kernel
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F
# from dac.nn.layers import Snake1d

class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with `scale`).
    """
    def __init__(self, *args, lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group

class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonally the residual outputs close to 0, with a learnt scale.

    Args:
        channels (int): Number of channels.
        init (float): Initial scale.
        channel_last (bool): If True, expect `[*, C]` shaped tensors, otherwise, `[*, C, T]`.
        device (torch.device or str, optional): Device on which to initialize the module.
        dtype (torch.dtype, optional): dtype to use to initialize the module.
    """
    def __init__(self, channels: int, init: float = 1e-4, channel_last: bool = True, device=None, dtype=None):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(
            torch.full((channels,), init, requires_grad=True, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x

# Layers
def create_sin_embedding(positions: torch.Tensor, dim: int, max_period: float = 10000,
                            dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create sinusoidal positional embedding, with shape `[B, T, C]`.

    Args:
        positions (torch.Tensor): LongTensor of positions.
        dim (int): Dimension of the embedding.
        max_period (float): Maximum period of the cosine/sine functions.
        dtype (torch.dtype or str): dtype to use to generate the embedding.
    Returns:
        torch.Tensor: Sinusoidal positional embedding.
    """
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

# Mask
def causal_mask(q: Tensor, k: Tensor) -> Tensor:
    b, i, j, device = q.shape[0], q.shape[-2], k.shape[-2], q.device
    mask = ~torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
    mask = repeat(mask, "n m -> b n m", b=b)
    return mask

def get_prefix_mask(seq: Tensor, condition: Tensor) -> Tensor:
    b, i, j, device = seq.shape[0], seq.shape[-2], condition.shape[-2], seq.device
    # print(b, i, j)
    mask = ~torch.ones((i, i), dtype=torch.bool, device=device).triu(j+1)
    
    mask = repeat(mask, "n m -> b n m", b=b)
    return mask

def add_mask(sim: Tensor, mask: Tensor) -> Tensor:
    b, ndim = sim.shape[0], mask.ndim
    if ndim == 3:
        mask = rearrange(mask, "b n m -> b 1 n m")
    if ndim == 2:
        mask = repeat(mask, "n m -> b 1 n m", b=b)
    max_neg_value = -torch.finfo(sim.dtype).max
    sim = sim.masked_fill(~mask, max_neg_value)
    return sim

T = TypeVar("T")

def exists(val: Optional[T]) -> T:
    return val is not None

def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d

# Modules
class MambaLayer(nn.Module):
    '''
    Mamba sub layer w/ add&norm and residual structure
    '''
    def __init__(self, d_model=1024, p=0.2, d_state=128, d_conv=4, expand=2, norm_epsilon=1e-5):
        super().__init__()
        # self.norm = nn.LayerNorm(d_model, eps=norm_epsilon)
        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(p)
        # self.drop = DropPath(p) if p > 0. else nn.Identity()
        self.mamba = Mamba2(
            d_model=d_model,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )
        
    def forward(self, x):
        # print('x',x.shape)
        B, L, C = x.shape
        x = x + self.drop(self.mamba(self.norm(x)))
        return x

class FFLayer(nn.Module):
    '''
    normal MLP
    H -> 2H -> H
    '''
    def __init__(self, d_model=1024, p=0.2):
        super().__init__()
        # self.dim = dim
        # self.norm = nn.LayerNorm(d_model)
        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(p)
        # self.drop = DropPath(p) if p > 0. else nn.Identity()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            # nn.ReLU(), # simba v23 v25
            nn.GELU(), # simba text_v9
            nn.Dropout(p),
            nn.Linear(d_model*2, d_model),
        )
        # self.feed_forward = EinFFT(d_model)
    def forward(self, x):
        # print('x',x.shape)
        B, L, C = x.shape
        x = x + self.drop(self.feed_forward(self.norm(x)))
        return x

# Blocks
class M_F_Block(nn.Module):
    '''
    Mamba w/ add&norm and residual + MLP w/ add&norm and residual
    '''
    def __init__(self, d_model=1024, p=0.2, d_state=128, d_conv=4, expand=2):
        super().__init__()
        self.mamba = MambaLayer(d_model=d_model, p=p, d_state=d_state, d_conv=d_conv, expand=expand)
        self.ff = FFLayer(d_model=d_model, p=p)
        
    def forward(self, x, condition_embed, condition_mask):
        # print('x',x.shape)
        B, L, C = x.shape
        x_mamba = self.mamba(x)
        x_ff = self.ff(x_mamba)
        return x_ff

class Text_Mmamba(nn.Module):
    def __init__(self, 
        layers = 24,
        vocab_size = 1024,
        codec_layer = 4,
        d_model = 1024,
        drop_p = 0.2, d_state=128, d_conv=4, expand=2, **kwargs
    ):
        super().__init__()
        self.codec_layer = codec_layer
        self.condition_norm = nn.LayerNorm(d_model)
        self.condition_linear = nn.Linear(768, d_model, bias=True)
        self.embedding = nn.ModuleList([ScaledEmbedding(vocab_size, d_model) for _ in range(self.codec_layer)])
        self.backbone = nn.ModuleList()
        
        
        # in-context SiMBA
        for i in range(layers):
            self.backbone.append(M_F_Block(d_model=d_model, p=drop_p, d_state=d_state, d_conv=d_conv, expand=expand))

        self.lm_head = nn.ModuleList([nn.Linear(d_model, vocab_size, bias=False) for _ in range(self.codec_layer)])
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, condition, condition_mask):
        '''
        condition_mask: [B, L]
        '''
        # B, K, S = x.shape
        # print(x.shape)
        # a = input('====================')
        # sum the codebooks
        x_emb = sum([self.embedding[k](x[:, k]) for k in range(self.codec_layer)])

        B_emb, T_emb, C_emb = x_emb.shape
        condition = self.condition_norm(self.condition_linear(condition))
        
        hidden_states = torch.cat([condition, x_emb], dim=1)
        
        positions = torch.arange(
                hidden_states.shape[1],
                device=hidden_states.device
            ).view(1, -1, 1)
        pos_emb = create_sin_embedding(positions, hidden_states.shape[-1])
        hidden_states = hidden_states + pos_emb
        
        
        for idx, block in enumerate(self.backbone):
            hidden_states = block(hidden_states, condition, condition_mask)
        
        lm_logits = torch.stack([self.lm_head[k](hidden_states) for k in range(self.codec_layer)], dim=1)
        
        # print(f'==============================={lm_logits.shape}=======================================')
        total_l, cond_l = hidden_states.shape[-2], condition.shape[-2]
        lm_logits = lm_logits[:, :, cond_l-1:, :]
        
        return lm_logits

def cal_torch_model_params(model):
    '''
    :param model:
    :return:
    '''
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': total_params/1000000, 'total_trainable_params': total_trainable_params/1000000}

def main():
    num = 3
    layers = 24
    self_atten_layers = []
    for i in range(layers):
        if i%(layers//num) == 0:
            self_atten_layers.append(i)
    print(self_atten_layers)
    device = 'cuda:1'
    # model = Mmamba(layers = 40,
    #     vocab_size = 2048,
    #     d_model = 1024,
    #     drop_p = 0.2, d_state=512, d_conv=4, expand=2)
    
    # pure (M-CA-FF)*N test
    # model = Text_Mmamba(
    #     layers = layers,
    #     vocab_size = 2048,
    #     d_model = 1024,
    #     drop_p = 0.2, d_state=512, d_conv=4, expand=2, num_heads=16)
    
    # inner attention test
    model = Text_Mmamba(
        layers = layers,
        vocab_size = 2048,
        d_model = 1024,
        drop_p = 0.2, d_state=512, d_conv=4, expand=2, num_heads=16, inner=True, self_atten_layers=self_atten_layers)
    print(model)
    # print(self_atten_layers)
    # return
    # outer attention test
    # model = Text_Mmamba(
    #     layers = layers,
    #     vocab_size = 2048,
    #     d_model = 1024,
    #     drop_p = 0.2, d_state=512, d_conv=4, expand=2, num_heads=16, inner=False, self_atten_layers=self_atten_layers)
    
    print(cal_torch_model_params(model))
    
    # model = model.to(device)
    # import torch
    # x = torch.randint(0, 2048, (1, 4, 1500))
    # x = x.to(device)
    # torch.cuda.set_device(x.device.index)
    # y = model(x)
    # print('OUTPUT shape:', y.shape)

if __name__ == '__main__':
    main()