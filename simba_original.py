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

def get_init_fn(method: str, input_dim: int, init_depth = None):
    """LM layer initialization.
    Inspired from xlformers: https://github.com/fairinternal/xlformers

    Args:
        method (str): Method name for init function. Valid options are:
            'gaussian', 'uniform'.
        input_dim (int): Input dimension of the initialized module.
        init_depth (int, optional): Optional init depth value used to rescale
            the standard deviation if defined.
    """
    # Compute std
    std = 1 / math.sqrt(input_dim)
    # Rescale with depth
    if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)

    if method == 'gaussian':
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    elif method == 'uniform':
        bound = math.sqrt(3) * std  # ensure the standard deviation is `std`
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)
    else:
        raise ValueError("Unsupported layer initialization method")

def init_layer(m: nn.Module,
               method: str,
               init_depth: int = None,
               zero_bias_init: bool = False):
    """Wrapper around ``get_init_fn`` for proper initialization of LM modules.

    Args:
        m (nn.Module): Module to initialize.
        method (str): Method name for the init function.
        init_depth (int, optional): Optional init depth value used to rescale
            the standard deviation if defined.
        zero_bias_init (bool): Whether to initialize the bias to 0 or not.
    """
    if isinstance(m, nn.Linear):
        init_fn = get_init_fn(method, m.in_features, init_depth=init_depth)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)
        if zero_bias_init and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init_fn = get_init_fn(method, m.embedding_dim, init_depth=None)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)



class EinFFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim #768
        self.num_blocks = 4 
        self.block_size = self.hidden_size // self.num_blocks 
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.complex_weight_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_weight_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * self.scale)
        self.complex_bias_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * self.scale)

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, N, self.num_blocks, self.block_size)

        x = torch.fft.fft2(x, dim=(1,2), norm='ortho') # FFT on N dimension

        x_real_1 = F.relu(self.multiply(x.real, self.complex_weight_1[0]) - self.multiply(x.imag, self.complex_weight_1[1]) + self.complex_bias_1[0])
        x_imag_1 = F.relu(self.multiply(x.real, self.complex_weight_1[1]) + self.multiply(x.imag, self.complex_weight_1[0]) + self.complex_bias_1[1])
        x_real_2 = self.multiply(x_real_1, self.complex_weight_2[0]) - self.multiply(x_imag_1, self.complex_weight_2[1]) + self.complex_bias_2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_2[1]) + self.multiply(x_imag_1, self.complex_weight_2[0]) + self.complex_bias_2[1]

        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        x = F.softshrink(x, lambd=self.sparsity_threshold) if self.sparsity_threshold else x
        x = torch.view_as_complex(x)

        x = torch.fft.ifft2(x, dim=(1,2), norm="ortho")
        
        # RuntimeError: "fused_dropout" not implemented for 'ComplexFloat'
        x = x.to(torch.float32)
        x = x.reshape(B, N, C)
        return x

def _generate_square_subsequent_mask(
            sz: int,
            device = torch.device('cpu'),
            dtype = torch.bool
    ):
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    mask = torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )
    return mask

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
class AttentionBase(nn.Module):
    def __init__(
        self,
        features: int = 1024,
        num_heads: int = 8,
        drop_p: float = 0.3
    ):
        super().__init__()
        head_features = features // num_heads
        
        self.scale = head_features**-0.5
        self.num_heads = num_heads
        self.drop_p = drop_p
        mid_features = features

        self.to_out = nn.Linear(
            in_features=mid_features, out_features=features
        )

        # self.use_flash = torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse('2.0.0')
        self.use_flash = True
        # self.use_flash = False
        
        if not self.use_flash:
            self.drop = nn.Dropout(self.drop_p)

        # device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        '''
        May need adjust
        open flash atten for RTX3090 or other GPU...
        and *spda does not support return attention weight*
        '''
        # if device_properties.major == 8 and device_properties.minor == 0:
        #     # Use flash attention for A100 GPUs
        #     self.sdp_kernel_config = (True, False, False)
        # else:
        #     # Don't use flash attention for other GPUs
        #     self.sdp_kernel_config = (False, True, True)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, is_causal: bool = False
    ) -> Tensor:
        # Split heads
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)

        if not self.use_flash:
            if is_causal and not mask:
                # Mask out future tokens for causal attention
                mask = causal_mask(q, k)

            # Compute similarity matrix and add eventual mask
            sim = einsum("... n d, ... m d -> ... n m", q, k) * self.scale
            sim = add_mask(sim, mask) if exists(mask) else sim

            # Get attention matrix with softmax
            # need to return for visualization
            attn = sim.softmax(dim=-1, dtype=torch.float32)
            attn = self.drop(attn)
            
            # Compute values
            out = einsum("... n m, ... m d -> ... n d", attn, v)
        else:
            # with sdp_kernel(*self.sdp_kernel_config):
            '''
            need adjust here
            mask True for take part in the attention; False for not
            '''
            # mask = repeat(context_mask, "b m -> b m d", d=v.shape[-1])
            # with torch.backends.cuda.sdp_kernel(enable_math=False):
            # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop_p, is_causal=is_causal)
            # out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop_p, is_causal=is_causal)
            attn = None

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), attn

class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        num_heads: int,
        drop_p: float,
        context_features: Optional[int] = None,
        causal: bool = False,
    ):
        super().__init__()
        # '''
        # features: 等價於 d_model
        # head_features: int,
        # num_heads: int,
        # '''
        
        self.context_features = context_features
        self.causal = causal
        self.num_heads = num_heads
        
        context_features = default(context_features, features)
        head_features = features // num_heads
        
        assert head_features * num_heads == features

        # self.norm = nn.LayerNorm(features)
        # self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(
            in_features=features, out_features=features, bias=False
        )
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=features * 2, bias=False
        )
        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            drop_p=drop_p
        )

    def forward(
        self,
        x: Tensor, # [b, n, c]
        context: Optional[Tensor] = None, # [b, m, d]
        context_mask: Optional[Tensor] = None,  # [b, m], false is masked,
        prefix_mask: Optional[Tensor] = None,  # [b, l], false is masked,
        causal: Optional[bool] = False,
    ):
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        # Use context if provided
        context = default(context, x)
        # Normalize then compute q from input and k,v from context
        # x, context = self.norm(x), self.norm_context(context)

        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))

        if exists(context_mask):
            # Mask out cross-attention for padding tokens
            mask = repeat(context_mask, "b m -> b m d", d=v.shape[-1])
            k, v = k * mask, v * mask
        
        # Compute and return attention
        if exists(prefix_mask):
            prefix_mask = repeat(prefix_mask, "b q k -> b h q k", h=self.num_heads)
            attention, weight = self.attention(q, k, v, mask=prefix_mask, is_causal=False)
        else:
            attention, weight = self.attention(q, k, v, is_causal=self.causal or causal)
        
        return attention, weight

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

class MSALayer(nn.Module):
    '''
    multi-head self-attention sub layer w/ add&norm and residual structure
    '''
    def __init__(self, d_model=1024, p=0.2, num_heads=16, norm_epsilon=1e-5):
        super().__init__()
        # self.norm = nn.LayerNorm(d_model, eps=norm_epsilon)
        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(p)
        # self.drop = DropPath(p) if p > 0. else nn.Identity()
        # self.self_atten = nn.MultiheadAttention(d_model, num_heads, dropout=p, batch_first=True)
        self.self_atten = Attention(d_model, num_heads, drop_p=p)
        
    def forward(self, x, prefix_mask=None):
        B, L, C = x.shape
        
        # positions = torch.arange(L, device=x.device).view(1, -1, 1)
        # pos_emb = create_sin_embedding(positions, C, max_period=10000, dtype=x.dtype)
        # x = x + pos_emb
        if self.training:
            is_causal = True
        else:
            is_causal = False
            is_causal = True
        
        x_norm = self.norm(x)
        assert x_norm.shape[0] == B
        # self-attention
        if exists(prefix_mask):
            attn_output, attn_output_weights = self.self_atten(x=x_norm, prefix_mask=prefix_mask, causal=False)
        else:
            attn_output, attn_output_weights = self.self_atten(x=x_norm, causal=is_causal)
        x = x + self.drop(attn_output)
        return x

class MCALayer(nn.Module):
    '''
    multi-head cross-attention sub layer w/ add&norm and residual structure
    '''
    def __init__(self, d_model=1024, p=0.2, num_heads=16, norm_epsilon=1e-5):
        super().__init__()
        # self.norm = nn.LayerNorm(d_model, eps=norm_epsilon)
        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(p)
        # self.drop = DropPath(p) if p > 0. else nn.Identity()
        # self.cross_atten = nn.MultiheadAttention(d_model, num_heads, dropout=p, batch_first=True)
        self.cross_atten = Attention(d_model, num_heads, drop_p=p)
        
    def forward(self, x, condition_embed, condition_mask):
        B, L, C = x.shape
        
        is_causal = False
        
        # positions = torch.arange(L, device=x.device).view(1, -1, 1)
        # pos_emb = create_sin_embedding(positions, C, max_period=10000, dtype=x.dtype)
        # x = x + pos_emb
        
        x_norm = self.norm(x)
        condition_embed = self.norm(condition_embed)
        assert x_norm.shape[0] == B
        attn_output, attn_output_weights = self.cross_atten(x=x_norm, context=condition_embed, context_mask=condition_mask)
        # attn_output, attn_output_weights = self.cross_atten(x_norm, condition_embed, condition_embed)
        # attn_output, attn_output_weights = self.cross_atten(x_norm, condition_embed, condition_embed, key_padding_mask=condition_mask)
        x = x + self.drop(attn_output)
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

class M_SA_F_Block(nn.Module):
    '''
    for text_v9, text_v10 use
    for text condition generation
    
    (M)     Mamba w/ add&norm and residual + MLP w/ add&norm and residual
    (SA)    Cross-attention w/ add&norm and residual + MLP w/ add&norm and residual
    (F)     MLP w/ add&norm and residual + MLP w/ add&norm and residual
    '''
    def __init__(self, d_model=1024, p=0.2, d_state=128, d_conv=4, expand=2, num_heads=16):
        super().__init__()
        
        ## determine the self-attention's staking way (sequential or parallel)
        self.is_sequential = False
        
        self.mamba = MambaLayer(d_model=d_model, p=p, d_state=d_state, d_conv=d_conv, expand=expand)
        self.self_atten = MSALayer(d_model=d_model, p=p, num_heads=num_heads)
        self.ff = FFLayer(d_model=d_model, p=p)
        
        self.fusion_gate = nn.Linear(d_model * 2, d_model)
        # self.alpha = nn.Parameter(torch.zeros(d_model))
        nn.init.constant_(self.fusion_gate.bias, 5.0)
        nn.init.zeros_(self.fusion_gate.weight)
    
    def forward(self, x, condition_embed, condition_mask):
        B, L, C = x.shape
        
        if self.is_sequential:
            # mamba => self-attn => feed-forward
            x_mamba = self.mamba(x)
            x_self = self.self_atten(x_mamba)
            gate = torch.sigmoid(self.fusion_gate(torch.cat([x_mamba, x_self], dim=-1)))
            fused = gate * x_mamba + (1 - gate) * x_self
            # fused = x_mamba + self.alpha * x_self
            x_ff = self.ff(fused)
        else:
            # mamba + self-attn => feed-forward
            x_mamba = self.mamba(x)
            x_self = self.self_atten(x)
            gate = torch.sigmoid(self.fusion_gate(torch.cat([x_mamba, x_self], dim=-1)))
            fused = gate * x_mamba + (1 - gate) * x_self
            x_ff = self.ff(fused)
        
        return x_ff

class M_CA_F_Block(nn.Module):
    '''
    for text_v9, text_v10 use
    for text condition generation
    
    (M)     Mamba w/ add&norm and residual + MLP w/ add&norm and residual
    (CA)    Cross-attention w/ add&norm and residual + MLP w/ add&norm and residual
    (F)     MLP w/ add&norm and residual + MLP w/ add&norm and residual
    '''
    def __init__(self, d_model=1024, p=0.2, d_state=128, d_conv=4, expand=2, num_heads=16):
        super().__init__()
        
        self.mamba = MambaLayer(d_model=d_model, p=p, d_state=d_state, d_conv=d_conv, expand=expand)
        self.cross_atten = MCALayer(d_model=d_model, p=p, num_heads=num_heads)
        self.ff = FFLayer(d_model=d_model, p=p)
        
    def forward(self, x, condition_embed, condition_mask):
        B, L, C = x.shape
        
        x_mamba = self.mamba(x)
        x_cross = self.cross_atten(x_mamba, condition_embed, condition_mask)
        x_ff = self.ff(x_cross)
        return x_ff

# class M_CA_F_Block(nn.Module):
#     '''
#     for text_v11 use w/ swiGLU
#     for text condition generation

#     (M)     Mamba w/ add&norm and residual + MLP w/ add&norm and residual
#     (CA)    Cross-attention w/ add&norm and residual + MLP w/ add&norm and residual
#     (F)     MLP w/ add&norm and residual + MLP w/ add&norm and residual
#     '''
#     def __init__(self, d_model=1024, p=0.2, d_state=128, d_conv=4, expand=2, num_heads=16):
#         super().__init__()
#         # text_v9
#         self.mamba1 = MambaLayer(d_model=d_model, p=p, d_state=d_state, d_conv=d_conv, expand=expand)
#         # self.fusion_gate = nn.Linear(d_model * 2, d_model)
#         self.cross_atten = MCALayer(d_model=d_model, p=p, num_heads=num_heads)
#         self.ff = FFLayer(d_model=d_model, p=p)
        
#     def forward(self, x, condition_embed, condition_mask):
#         # print('x',x.shape)
#         B, L, C = x.shape
        
#         x_mamba1 = self.mamba1(x)
#         x_cross = self.cross_atten(x_mamba1, condition_embed, condition_mask)
#         # gate = torch.sigmoid(self.fusion_gate(torch.cat([x_mamba1, x_cross], dim=-1)))
#         # fused = gate * x_mamba1 + (1 - gate) * x_cross
        
#         x_ff = self.ff(x_cross)
#         return x_ff

class SA_CA_F_Block(nn.Module):
    '''
    Inner attention structure in Mamba based transformer for text condition generation
    '''
    def __init__(self, d_model=1024, p=0.2, num_heads=16):
        super().__init__()
        self.self_atten = MSALayer(d_model=d_model, p=p, num_heads=num_heads)
        self.cross_atten = MCALayer(d_model=d_model, p=p, num_heads=num_heads)
        self.ff = FFLayer(d_model=d_model, p=p)
        
    def forward(self, x, condition_embed, condition_mask):
        B, L, C = x.shape
        x_self = self.self_atten(x)
        x_cross = self.cross_atten(x_self, condition_embed, condition_mask)
        x_ff = self.ff(x_cross)
        return x_ff

class SA_F_Block(nn.Module):
    '''
    Only apply in in-context condition transformer
    '''
    def __init__(self, d_model=1024, p=0.2, num_heads=16):
        super().__init__()
        self.self_atten = MSALayer(d_model=d_model, p=p, num_heads=num_heads)
        self.ff = FFLayer(d_model=d_model, p=p)
        
    def forward(self, x, condition_embed, condition_mask):
        B, L, C = x.shape
        prefix_mask = get_prefix_mask(x, condition_embed)
        
        x_self = self.self_atten(x, prefix_mask)
        x_ff = self.ff(x_self)
        return x_ff


# Models
class Mmamba(nn.Module):
    def __init__(self, 
        layers = 48,
        vocab_size = 2048,
        d_model = 1024,
        drop_p = 0.2, d_state=128, d_conv=4, expand=2
    ):
        super().__init__()
        
        self.embedding = nn.ModuleList([ScaledEmbedding(vocab_size, d_model) for _ in range(4)])
        self.backbone = nn.ModuleList([M_F_Block(d_model=d_model, p=drop_p, d_state=d_state, d_conv=d_conv, expand=expand) for j in range(layers)])
        self.lm_head = nn.ModuleList([nn.Linear(d_model, vocab_size, bias=False) for _ in range(4)])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, K, S = x.shape
        x_emb = sum([self.embedding[k](x[:, k]) for k in range(K)])
        hidden_states = x_emb
        for block in self.backbone:
            hidden_states = block(hidden_states)
            # print(hidden_states.shape)
        lm_logits = torch.stack([self.lm_head[k](hidden_states) for k in range(K)], dim=1)
        return lm_logits

class Text_Mmamba(nn.Module):
    def __init__(self, 
        layers = 24,
        vocab_size = 1024,
        codec_layer = 4,
        d_model = 1024,
        is_incontext = False,
        is_pure_mamba = False,
        drop_p = 0.2, d_state=128, d_conv=4, expand=2, num_heads=8, self_atten_layers=[], **kwargs
    ):
        super().__init__()
        self.codec_layer = codec_layer
        self.condition_norm = nn.LayerNorm(d_model)
        self.condition_linear = nn.Linear(768, d_model, bias=True)
        self.embedding = nn.ModuleList([ScaledEmbedding(vocab_size, d_model) for _ in range(self.codec_layer)])
        self.backbone = nn.ModuleList()
        self.is_incontext = is_incontext
        self.is_pure_mamba = is_pure_mamba
        
        if not is_incontext:
            # cross-attention
            for i in range(layers):
                if i not in self_atten_layers:
                    self.backbone.append(M_CA_F_Block(d_model=d_model, p=drop_p, d_state=d_state, d_conv=d_conv, expand=expand, num_heads=num_heads))
                else:
                    self.backbone.append(SA_CA_F_Block(d_model=d_model, p=drop_p, num_heads=num_heads))
        else:
            # in-context
            for i in range(layers):
                if is_pure_mamba:
                    self.backbone.append(MambaLayer(d_model=d_model, p=drop_p, d_state=d_state, d_conv=d_conv, expand=expand))
                else:
                    if i not in self_atten_layers:
                        self.backbone.append(M_F_Block(d_model=d_model, p=drop_p, d_state=d_state, d_conv=d_conv, expand=expand))
                    else:
                        # 原本是把mamba換成self-attn，但效果不好
                        # self.backbone.append(SA_F_Block(d_model=d_model, p=drop_p, num_heads=num_heads))
                        
                        # 這邊是把 self-attn 加入特定的層數，與 mamba 做hybrid。
                        # 但有sequential & parallel的差別，所以要再注意一下
                        self.backbone.append(M_SA_F_Block(d_model=d_model, p=drop_p, d_state=d_state, d_conv=d_conv, expand=expand, num_heads=num_heads))
        
        self.lm_head = nn.ModuleList([nn.Linear(d_model, vocab_size, bias=False) for _ in range(self.codec_layer)])
        # self.apply(self._init_weights)
    
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
        
        if self.is_incontext:
            B_emb, T_emb, C_emb = x_emb.shape
            condition = self.condition_norm(self.condition_linear(condition))
            
            if self.is_pure_mamba:
                condition = condition * condition_mask.unsqueeze(-1)
            
            hidden_states = torch.cat([condition, x_emb], dim=1)
            # print(hidden_states.shape)
            
            positions = torch.arange(
                    hidden_states.shape[1],
                    device=hidden_states.device
                ).view(1, -1, 1)
            pos_emb = create_sin_embedding(positions, hidden_states.shape[-1])
            hidden_states = hidden_states + pos_emb
        else:
            # add positional encoding
            B_emb, T_emb, C_emb = x_emb.shape
            positions = torch.arange(T_emb, device=x.device).view(1, -1, 1)
            pos_emb = create_sin_embedding(positions, C_emb, max_period=10000, dtype=x.dtype)
            x_emb = x_emb + pos_emb
            
            # condition pre-norm
            # all layers share weight
            condition = self.condition_norm(self.condition_linear(condition))
            # condition_mask = (condition_mask == 0)
            positions = torch.arange(
                    condition.shape[1],
                    device=condition.device
                ).view(1, -1, 1)
            pos_emb = create_sin_embedding(positions, condition.shape[-1])
            condition = condition + pos_emb
            
            hidden_states = x_emb
        
        
        for idx, block in enumerate(self.backbone):
            if self.is_pure_mamba:
                hidden_states = block(hidden_states)
            else:
                hidden_states = block(hidden_states, condition, condition_mask)
        
        lm_logits = torch.stack([self.lm_head[k](hidden_states) for k in range(self.codec_layer)], dim=1)
        
        # print(f'==============================={lm_logits.shape}=======================================')
        if self.is_incontext:
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