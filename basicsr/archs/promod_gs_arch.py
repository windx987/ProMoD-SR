'''
ProMoD v1.1 (PMDGSModel): PFT + Mixture-of-Depths with REAL gather/scatter
execution, instead of v1.0's mask-multiply.

v1.0 (promod_arch.py, PMDModel) already implements MoD routing exactly as
Raposo et al. 2024 (arXiv:2404.02258) describe it — per-window top-k select,
weights = mask * sigmoid(score) multiplied into the attention/FFN output
before the residual add, giving the router a real gradient path. Its one
documented gap: attention and FFN still run densely on every token; only the
*output* gets zeroed for inactive tokens (see benchmark.py's header comment).

v1.1 keeps the routing math byte-for-byte identical and changes only the
compute path: attention's query side and the FFN's fc2 are gathered to
active tokens only, run at reduced size, and scattered back — real FLOPs
and latency savings, not just an analytical estimate.

Two things this file deliberately does NOT do, both informed by bugs found
in the ProSAT (basicsr/archs/prosat_arch.py) investigation this session —
see PROGRESS.md and this experiment's plan for the full reasoning:

1. It never gathers *before* a depthwise (spatial) convolution. ConvFFN's
   dwconv and v_LePE both mix spatial neighbors over the full H*W grid;
   scattering skipped tokens into a zero-filled buffer before such a conv
   would corrupt neighboring active tokens' conv output (ProSAT's GDFN bug).
   fc1/act/dwconv and v_LePE stay fully dense here; only fc2 (pointwise) and
   attention's query-side matmuls are ever gathered.
2. It never threads PFA's (Progressive Focusing Attention, PFT's own
   cross-layer sparse-attention cascade) index chain through a routed layer.
   PFA assumes every layer's query-row count equals the full window size —
   a real query gather breaks that invariant for every later layer. Routed
   layers (capacity_ratio < 1.0) bypass PFA bookkeeping entirely and do
   inline attention using attn_win's own parameters; only capacity_ratio ==
   1.0 (warmup) layers call attn_win.forward() and participate in the PFA
   chain, identically to v1.0.
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs.pft_arch import (
    dwconv,
    window_partition, window_reverse,
    WindowAttention,
    PatchEmbed, PatchUnEmbed,
    Upsample, UpsampleOneStep,
)
from basicsr.archs.promod_arch import build_capacity_schedule


class RoutedConvFFN(nn.Module):
    """ConvFFN with fc1/act/dwconv always dense (dwconv mixes spatial
    neighbors over the full grid — gathering before it would reproduce the
    zero-fill corruption diagnosed in ProSAT's GDFN); only fc2 (pointwise,
    safe to gather) is applied to active tokens only.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.out_features = out_features

    def forward(self, x, x_size, active_idx=None):
        # x: (b, n, in_features). active_idx: (b, k) full-image-space indices,
        # or None for the dense (capacity_ratio == 1.0) path.
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)  # always dense — needs the full grid

        if active_idx is None:
            return self.fc2(x)

        b, n, hidden = x.shape
        idx_h = active_idx.unsqueeze(-1).expand(-1, -1, hidden)
        x_active = torch.gather(x, 1, idx_h)            # (b, k, hidden) — real savings
        out_active = self.fc2(x_active)                  # (b, k, out_features)

        out = out_active.new_zeros(b, n, self.out_features)
        idx_out = active_idx.unsqueeze(-1).expand(-1, -1, self.out_features)
        out.scatter_(1, idx_out, out_active)
        return out


class PMDTLv1_1(nn.Module):
    """PFT Transformer Layer with real gather/scatter MoD execution.

    Routing math (per-window top-k + sigmoid gate) is identical to PMDTL's
    _compute_routing_weights. The only change is WHERE compute happens:
    capacity_ratio == 1.0 layers call attn_win.forward() unchanged (dense,
    PFA chain intact, bit-identical to v1.0). capacity_ratio < 1.0 layers
    gather Q (and v_LePE's active rows) to the active token set, compute
    attention manually against full dense K/V using attn_win's own
    relative_position_bias_table/scale/proj, and scatter back — bypassing
    PFA's pfa_values/pfa_indices bookkeeping entirely (see module docstring
    for why: a real query-gather breaks PFA's cross-layer index chain).
    """

    def __init__(self,
                 dim,
                 block_id,
                 layer_id,
                 input_resolution,
                 num_heads,
                 num_topk,
                 window_size,
                 shift_size,
                 convffn_kernel_size,
                 mlp_ratio,
                 capacity_ratio=1.0,
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.layer_id = layer_id
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.capacity_ratio = capacity_ratio

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # Router: scores tokens from the current layer's normed input.
        # Only instantiated for routing layers (capacity_ratio < 1.0).
        if capacity_ratio < 1.0:
            self.router = nn.Linear(dim, 1, bias=False)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.convlepe_kernel_size = convffn_kernel_size
        self.v_LePE = dwconv(hidden_features=dim, kernel_size=self.convlepe_kernel_size)

        # Always instantiated: holds relative_position_bias_table/scale/proj
        # used directly by the routed path, and its full forward() is called
        # unchanged by the dense (capacity_ratio == 1.0) path.
        self.attn_win = WindowAttention(
            self.dim,
            layer_id=layer_id,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            num_topk=num_topk,
            qkv_bias=qkv_bias,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = RoutedConvFFN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            kernel_size=convffn_kernel_size,
            act_layer=act_layer,
        )

    def _compute_routing_weights(self, x, h, w, b, n):
        """Identical math to PMDTL._compute_routing_weights (promod_arch.py),
        additionally returning the per-window active_idx (discarded there
        after building the mask) and k, since v1.1 needs both to gather rows
        for real instead of just zeroing an output.
        """
        scores = self.router(x)  # [B, N, 1], x is norm1'd

        scores = scores.view(b, h, w, 1)
        if self.shift_size > 0:
            scores = torch.roll(scores, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        score_windows = window_partition(scores, self.window_size)  # [B*nW, ws, ws, 1]
        win_n = self.window_size * self.window_size
        score_windows = score_windows.view(-1, win_n)  # [B*nW, win_n]

        k = max(1, int(math.ceil(self.capacity_ratio * win_n)))
        _, active_idx = torch.topk(score_windows, k, dim=-1)  # [B*nW, k], window-local coords
        active_idx, _ = torch.sort(active_idx, dim=-1)  # restore left-to-right order within window
        mask = torch.zeros_like(score_windows)
        mask.scatter_(1, active_idx, 1.0)

        weights = mask * torch.sigmoid(score_windows)  # [B*nW, win_n]

        weights = weights.view(-1, self.window_size, self.window_size, 1)
        weights = window_reverse(weights, self.window_size, h, w)  # [B, H, W, 1]
        if self.shift_size > 0:
            weights = torch.roll(weights, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        weights = weights.reshape(b, n, 1)
        return weights, active_idx, k

    def forward(self, x, pfa_list, x_size, params):
        pfa_values, pfa_indices = pfa_list[0], pfa_list[1]
        h, w = x_size
        b, n, c = x.shape
        c4 = 4 * c
        win_n = self.window_size * self.window_size

        shortcut = x
        x = self.norm1(x)
        x_qkv = self.wqkv(x)

        v_lepe = self.v_LePE(torch.split(x_qkv, c, dim=-1)[-1], x_size)  # dense — full grid, unchanged
        x_qkvp = torch.cat([x_qkv, v_lepe], dim=-1)

        if self.shift_size > 0:
            shift = 1
            shifted_x = torch.roll(
                x_qkvp.reshape(b, h, w, c4), shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shift = 0
            shifted_x = x_qkvp.reshape(b, h, w, c4)

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, win_n, c4)  # [b_, win_n, c4], b_ = B*nW

        if self.capacity_ratio >= 1.0:
            # --- Dense path: bit-identical to PMDTL, PFA chain intact ---
            attn_windows, pfa_values, pfa_indices = self.attn_win(
                x_windows, pfa_values=pfa_values, pfa_indices=pfa_indices,
                rpi=params['rpi_sa'], mask=params['attn_mask'], shift=shift,
            )
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
            shifted_out = window_reverse(attn_windows, self.window_size, h, w)
            if self.shift_size > 0:
                attn_x = torch.roll(shifted_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                attn_x = shifted_out

            x_attn = attn_x.view(b, n, c)
            x = shortcut + x_attn

            x_ffn = self.convffn(self.norm2(x), x_size)  # dense, active_idx=None
            x = x + x_ffn

            pfa_list = [pfa_values, pfa_indices]
            return x, pfa_list

        # --- Routed path: real gather/scatter, bypasses PFA entirely ---
        weights, active_idx, k = self._compute_routing_weights(x, h, w, b, n)

        num_heads = self.num_heads
        head_dim = c // num_heads
        b_ = x_windows.shape[0]

        qkvp = x_windows.reshape(b_, win_n, 4, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, kk, vv, v_lepe_w = qkvp[0], qkvp[1], qkvp[2], qkvp[3]  # each [b_, heads, win_n, head_dim]
        q = q * self.attn_win.scale

        idx_h = active_idx.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)  # [b_,heads,k,hd]
        q_active = torch.gather(q, 2, idx_h)                # [b_, heads, k, head_dim] — real savings
        v_lepe_active = torch.gather(v_lepe_w, 2, idx_h)     # [b_, heads, k, head_dim]

        attn = q_active @ kk.transpose(-2, -1)               # [b_, heads, k, win_n] — K/V stay full

        rpi = params['rpi_sa']
        rpb = self.attn_win.relative_position_bias_table[rpi.view(-1)].view(win_n, win_n, -1)
        # last dim is num_heads for classical SR (dim>100) but a single shared
        # value for lightweight SR (dim<=100, broadcast across heads) — explicit
        # expand to num_heads (not -1) handles both, matching WindowAttention's
        # own sparse-branch broadcast (pft_arch.py WindowAttention.forward).
        rpb = rpb.permute(2, 0, 1).contiguous()              # [heads_or_1, win_n, win_n]
        rpb = rpb.unsqueeze(0).expand(b_, num_heads, -1, -1)  # [b_, heads, win_n, win_n]
        idx_rpb = active_idx.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, win_n)
        rpb_active = torch.gather(rpb, 2, idx_rpb)           # [b_, heads, k, win_n]
        attn = attn + rpb_active

        if shift:
            mask_t = params['attn_mask']                      # [nW, win_n, win_n]
            nw = mask_t.shape[0]
            mask_exp = mask_t.unsqueeze(0).expand(b_ // nw, -1, -1, -1).reshape(b_, win_n, win_n)
            idx_mask = active_idx.unsqueeze(-1).expand(-1, -1, win_n)  # [b_, k, win_n]
            mask_active = torch.gather(mask_exp, 1, idx_mask)          # [b_, k, win_n]
            attn = attn + mask_active.unsqueeze(1)                     # broadcast over heads

        attn = torch.softmax(attn, dim=-1)
        out = (attn @ vv) + v_lepe_active                    # [b_, heads, k, head_dim]
        out = out.transpose(1, 2).reshape(b_, k, c)
        out = self.attn_win.proj(out)                        # [b_, k, c] — real savings

        attn_windows = out.new_zeros(b_, win_n, c)
        idx_scatter = active_idx.unsqueeze(-1).expand(-1, -1, c)
        attn_windows.scatter_(1, idx_scatter, out)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_out = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_out

        x_attn = attn_x.view(b, n, c) * weights
        x = shortcut + x_attn

        # Recover full-image-space active indices for the FFN gather. Every
        # window has the identical count k (single scalar, same topk call
        # for the whole batch), so num_windows*k values of `weights` are
        # strictly positive (sigmoid output) and the rest are exactly 0 —
        # a plain topk-by-weight over the flattened image reproduces the
        # same active set without re-deriving the window<->image mapping.
        num_windows = (h // self.window_size) * (w // self.window_size)
        k_total = k * num_windows
        _, active_idx_full = torch.topk(weights.reshape(b, n), k_total, dim=-1)
        active_idx_full, _ = torch.sort(active_idx_full, dim=-1)

        x_ffn = self.convffn(self.norm2(x), x_size, active_idx=active_idx_full) * weights
        x = x + x_ffn

        pfa_list = [pfa_values, pfa_indices]  # untouched — routed layers never read/write PFA state
        return x, pfa_list

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.input_resolution if input_resolution is None else input_resolution
        r = self.capacity_ratio

        flops += self.dim * 3 * self.dim * h * w  # wqkv, dense (unchanged from v1.0)

        nw = h * w / self.window_size / self.window_size
        # attention: gathered Q -> honest r scaling (unchanged from v1.0; Q@K^T,
        # Attn@V, and proj are all linear in query-row count)
        flops += nw * self.attn_win.flops(self.window_size * self.window_size) * r

        # FFN: fc1 (dense, factor 1) + fc2 (routed, factor r). v1.0's flops()
        # scaled BOTH by r (`2 * ... * r`) — optimistic, since fc1 must stay
        # dense to give dwconv a correct (non-zero-filled) full-grid input.
        flops += h * w * self.dim * self.dim * self.mlp_ratio * (1 + r)
        # dwconv: always dense here (can't be gathered without reproducing
        # the zero-fill bug diagnosed in ProSAT's GDFN) — v1.0's flops()
        # scaled this by r too; that was never actually realizable.
        flops += h * w * self.dim * (self.convffn_kernel_size ** 2) * self.mlp_ratio
        flops += h * w * self.dim * (self.convlepe_kernel_size ** 2)  # LePE, dense (unchanged)

        return flops


class PMDBBv1_1(nn.Module):
    """Container of ProMoD v1.1 transformer layers (PMDBB equivalent)."""

    def __init__(self,
                 dim,
                 input_resolution,
                 idx,
                 layer_id,
                 depth,
                 num_heads,
                 num_topk,
                 window_size,
                 convffn_kernel_size,
                 capacity_schedule,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.layers = nn.ModuleList()
        for i in range(depth):
            global_layer_id = layer_id + i
            r = capacity_schedule[global_layer_id] if global_layer_id < len(capacity_schedule) else 0.5
            self.layers.append(
                PMDTLv1_1(
                    dim=dim,
                    block_id=idx,
                    layer_id=global_layer_id,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    num_topk=num_topk,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    capacity_ratio=r,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
            )

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, pfa_list, x_size, params):
        for layer in self.layers:
            x, pfa_list = layer(x, pfa_list, x_size, params)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, pfa_list

    def flops(self, input_resolution=None):
        flops = 0
        for layer in self.layers:
            flops += layer.flops(input_resolution)
        if self.downsample is not None:
            flops += self.downsample.flops(input_resolution)
        return flops


class PMDBv1_1(nn.Module):
    """ProMoD v1.1 group block (PMDB equivalent)."""

    def __init__(self,
                 dim,
                 idx,
                 layer_id,
                 input_resolution,
                 depth,
                 num_heads,
                 num_topk,
                 window_size,
                 convffn_kernel_size,
                 mlp_ratio,
                 capacity_schedule,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv'):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.residual_group = PMDBBv1_1(
            dim=dim,
            input_resolution=input_resolution,
            idx=idx,
            layer_id=layer_id,
            depth=depth,
            num_heads=num_heads,
            num_topk=num_topk,
            window_size=window_size,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            capacity_schedule=capacity_schedule,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, pfa_list, x_size, params):
        x_block, pfa_list = self.residual_group(x, pfa_list, x_size, params)
        return self.patch_embed(self.conv(self.patch_unembed(x_block, x_size))) + x, pfa_list

    def flops(self, input_resolution=None):
        flops = 0
        flops += self.residual_group.flops(input_resolution)
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops(input_resolution)
        flops += self.patch_unembed.flops(input_resolution)
        return flops


@ARCH_REGISTRY.register()
class PMDGSModel(nn.Module):
    """ProMoD v1.1: PFT + Mixture-of-Depths with real gather/scatter
    execution. Same routing math as PMDModel (promod_arch.py); see module
    docstring at the top of this file for what changed and why.
    """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=90,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 num_topk=None,
                 window_size=8,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 mod_warmup_layers=2,
                 mod_disable=False,
                 mod_capacity=None,
                 **kwargs):
        super().__init__()

        if num_topk is None:
            num_topk = [256] * sum(depths)

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # --- 1. Shallow feature extraction ---
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # --- 2. Deep feature extraction ---
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        total_layers = sum(depths)
        self.capacity_schedule = build_capacity_schedule(
            total_layers, mod_warmup_layers, disable=mod_disable, capacity=mod_capacity)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim,
            embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        self.layers = nn.ModuleList()
        layer_id = 0
        for i_layer in range(self.num_layers):
            layer = PMDBv1_1(
                dim=embed_dim,
                idx=i_layer,
                layer_id=layer_id,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads,
                num_topk=num_topk,
                window_size=window_size,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                capacity_schedule=self.capacity_schedule,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)
            layer_id += depths[i_layer]

        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # --- 3. Reconstruction ---
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])

        pfa_values = [None, None]
        pfa_indices = [None, None]
        pfa_list = [pfa_values, pfa_indices]

        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed

        for layer in self.layers:
            x, pfa_list = layer(x, pfa_list, x_size, params)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, params)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]

        return x

    def flops(self, input_resolution=None):
        flops = 0
        resolution = self.patches_resolution if input_resolution is None else input_resolution
        h, w = resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops(resolution)
        for layer in self.layers:
            flops += layer.flops(resolution)
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops(resolution)
        return flops


if __name__ == '__main__':
    model = PMDGSModel(
        upscale=2,
        img_size=64,
        embed_dim=52,
        depths=[2, 4, 6, 6, 6],
        num_heads=4,
        num_topk=[1024, 1024,
                  256, 256, 256, 256,
                  128, 128, 128, 128, 128, 128,
                  64, 64, 64, 64, 64, 64,
                  32, 32, 32, 32, 32, 32],
        window_size=32,
        convffn_kernel_size=7,
        img_range=1.,
        mlp_ratio=1,
        upsampler='pixelshuffledirect',
        mod_warmup_layers=2,
    )

    total = sum([param.nelement() for param in model.parameters()])
    print(f"Number of parameters: {total / 1e6:.3f}M")
    print(f"Capacity schedule: {model.capacity_schedule}")
    print(f"FLOPs (640x360): {model.flops([640, 360]) / 1e9:.2f}G")
    print(f"FLOPs (320x180): {model.flops([320, 180]) / 1e9:.2f}G")
