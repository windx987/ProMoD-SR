'''
ProMoD-CLF: Cross-Layer Feature Fusion for Image Super-Resolution.

Based on PFT (Progressive Focused Transformer). Unlike ProMoD's Mixture-of-
Depths variants (v1.0/v1.1/MoE), this file adds NO token routing, NO
capacity schedule, and NO router of any kind -- it is deliberately MoD-free.

PFT's own cross-layer mechanism (PFA) only narrows attention *indices*
across layers; no layer can ever pull out and independently weight one
specific earlier layer's raw hidden-state output -- the residual stream
only ever carries an undifferentiated running sum forward. CrossLayerFusion
adds exactly that: a small, gated, per-layer hook into the last K layers'
undiluted feature snapshots, fused into the *input to attention* (not the
residual `shortcut`) so an over-eager gate can only perturb what attention
attends to, never inject an unbounded term into the residual highway.

Two things make this stable from step 0, not just "hopefully fine":
  - `proj` weights are zero-initialized -> the module is the exact identity
    function at init, regardless of gate value.
  - the gate is a raw, unconstrained scalar (or per-channel vector), never
    passed through sigmoid -- there is no saturation region to get stuck in.
All ops are pointwise (nn.Linear + elementwise multiply); no spatial/conv op
ever touches the history tensors, so there is no "zero-filled buffer next to
a depthwise conv" failure mode to worry about.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs.pft_arch import (
    dwconv, ConvFFN,
    window_partition, window_reverse,
    WindowAttention,
    PatchEmbed, PatchUnEmbed,
    Upsample, UpsampleOneStep,
)


class CrossLayerFusion(nn.Module):
    """Gated blend of the last K layers' raw output hidden states into the
    current layer's pre-attention input.

    forward() handles a partially-filled history gracefully: at layer l,
    only min(l, history_window) slots are available/used, so no special
    casing is needed for the first few layers of the stack.
    """

    def __init__(self, dim, history_window, gate_type='scalar', fusion_proj=True):
        super().__init__()
        self.k = history_window
        self.gate_type = gate_type

        if fusion_proj:
            self.proj = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(history_window)])
            for lin in self.proj:
                nn.init.zeros_(lin.weight)
        else:
            self.proj = nn.ModuleList([nn.Identity() for _ in range(history_window)])

        if gate_type == 'scalar':
            self.gate = nn.Parameter(torch.zeros(history_window))
        elif gate_type == 'channel':
            self.gate = nn.Parameter(torch.zeros(history_window, dim))
        else:
            raise ValueError(f'Unknown gate_type: {gate_type}')

    def forward(self, x, feat_history):
        recent = feat_history[-self.k:]  # oldest-first, most-recent-last
        fused = 0.
        for j, h in enumerate(reversed(recent)):  # j=0 -> lag1 (most recent), j=1 -> lag2, ...
            h_proj = self.proj[j](h)
            if self.gate_type == 'scalar':
                g = self.gate[j]
            else:  # 'channel'
                g = self.gate[j].view(1, 1, -1)
            fused = fused + g * h_proj
        return x + fused


class PMDCLFTL(nn.Module):
    """PFT Transformer Layer with Cross-Layer Feature Fusion. No MoD routing."""

    def __init__(self,
                 dim,
                 layer_id,
                 input_resolution,
                 num_heads,
                 num_topk,
                 window_size,
                 shift_size,
                 convffn_kernel_size,
                 mlp_ratio,
                 history_window=3,
                 gate_type='scalar',
                 fusion_proj=True,
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
        self.history_window = history_window

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.clf = CrossLayerFusion(dim, history_window, gate_type=gate_type,
                                     fusion_proj=fusion_proj) if history_window > 0 else None

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.convlepe_kernel_size = convffn_kernel_size
        self.v_LePE = dwconv(hidden_features=dim, kernel_size=self.convlepe_kernel_size)

        self.attn_win = WindowAttention(
            self.dim,
            layer_id=layer_id,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            num_topk=num_topk,
            qkv_bias=qkv_bias,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            kernel_size=convffn_kernel_size,
            act_layer=act_layer,
        )

    def forward(self, x, pfa_list, feat_history, x_size, params):
        pfa_values, pfa_indices = pfa_list[0], pfa_list[1]
        h, w = x_size
        b, n, c = x.shape
        c4 = 4 * c

        # --- Cross-Layer Feature Fusion ---
        # Fuses into what attention SEES, not into `shortcut` -- the residual
        # highway stays byte-identical to stock PFT.
        if self.clf is not None and len(feat_history) > 0:
            x_in = self.clf(x, feat_history)
        else:
            x_in = x

        shortcut = x                # unchanged -- original, un-fused input
        x = self.norm1(x_in)        # CHANGED vs stock PFT: was self.norm1(x)
        x_qkv = self.wqkv(x)

        v_lepe = self.v_LePE(torch.split(x_qkv, c, dim=-1)[-1], x_size)
        x_qkvp = torch.cat([x_qkv, v_lepe], dim=-1)

        # SW-MSA
        if self.shift_size > 0:
            shift = 1
            shifted_x = torch.roll(
                x_qkvp.reshape(b, h, w, c4),
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shift = 0
            shifted_x = x_qkvp.reshape(b, h, w, c4)

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c4)

        attn_windows, pfa_values, pfa_indices = self.attn_win(
            x_windows,
            pfa_values=pfa_values,
            pfa_indices=pfa_indices,
            rpi=params['rpi_sa'],
            mask=params['attn_mask'],
            shift=shift,
        )

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x

        x = shortcut + attn_x.view(b, n, c)
        x = x + self.convffn(self.norm2(x), x_size)

        feat_history = (feat_history + [x])[-self.history_window:] if self.history_window > 0 else feat_history

        pfa_list = [pfa_values, pfa_indices]
        return x, pfa_list, feat_history

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.input_resolution if input_resolution is None else input_resolution

        flops += self.dim * 3 * self.dim * h * w

        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn_win.flops(self.window_size * self.window_size)

        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
        flops += h * w * self.dim * (self.convffn_kernel_size ** 2) * self.mlp_ratio
        flops += h * w * self.dim * (self.convlepe_kernel_size ** 2)

        # CrossLayerFusion: exact (not worst-case) active-slot count for this
        # layer's position in the stack -- layer_id counts prior layers, so
        # only min(layer_id, history_window) slots are ever populated.
        if self.clf is not None:
            active_slots = min(self.layer_id, self.history_window)
            if isinstance(self.clf.proj[0], nn.Linear):
                flops += active_slots * h * w * self.dim * self.dim   # proj_j
            flops += active_slots * h * w * self.dim                  # gate multiply + add

        return flops


class PMDCLFBB(nn.Module):
    """Container of CLF transformer layers. Carries PFA state and feature
    history across layers, exactly as PMDBB threads pfa_list today."""

    def __init__(self,
                 dim,
                 input_resolution,
                 layer_id,
                 depth,
                 num_heads,
                 num_topk,
                 window_size,
                 convffn_kernel_size,
                 mlp_ratio=4.,
                 history_window=3,
                 gate_type='scalar',
                 fusion_proj=True,
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
            self.layers.append(
                PMDCLFTL(
                    dim=dim,
                    layer_id=global_layer_id,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    num_topk=num_topk,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    history_window=history_window,
                    gate_type=gate_type,
                    fusion_proj=fusion_proj,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
            )

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, pfa_list, feat_history, x_size, params):
        for layer in self.layers:
            x, pfa_list, feat_history = layer(x, pfa_list, feat_history, x_size, params)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, pfa_list, feat_history

    def flops(self, input_resolution=None):
        flops = 0
        for layer in self.layers:
            flops += layer.flops(input_resolution)
        if self.downsample is not None:
            flops += self.downsample.flops(input_resolution)
        return flops


class PMDCLFB(nn.Module):
    """CLF group block (PFTB equivalent, no MoD)."""

    def __init__(self,
                 dim,
                 layer_id,
                 input_resolution,
                 depth,
                 num_heads,
                 num_topk,
                 window_size,
                 convffn_kernel_size,
                 mlp_ratio,
                 history_window=3,
                 gate_type='scalar',
                 fusion_proj=True,
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

        self.residual_group = PMDCLFBB(
            dim=dim,
            input_resolution=input_resolution,
            layer_id=layer_id,
            depth=depth,
            num_heads=num_heads,
            num_topk=num_topk,
            window_size=window_size,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            history_window=history_window,
            gate_type=gate_type,
            fusion_proj=fusion_proj,
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

    def forward(self, x, pfa_list, feat_history, x_size, params):
        x_block, pfa_list, feat_history = self.residual_group(x, pfa_list, feat_history, x_size, params)
        return self.patch_embed(self.conv(self.patch_unembed(x_block, x_size))) + x, pfa_list, feat_history

    def flops(self, input_resolution=None):
        flops = 0
        flops += self.residual_group.flops(input_resolution)
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops(input_resolution)
        flops += self.patch_unembed.flops(input_resolution)
        return flops


@ARCH_REGISTRY.register()
class PMDCLFModel(nn.Module):
    """ProMoD-CLF: PFT + Cross-Layer Feature Fusion for Image Super-Resolution.

    Quality-focused, MoD-free. Every layer gets a gated, learned hook into
    the last `history_window` layers' raw output features -- a genuinely
    new information pathway PFT's own PFA cascade never provides, since PFA
    only narrows attention indices, never carries hidden-state content
    forward beyond the standard residual stream.
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
                 history_window=3,
                 fusion_gate_type='scalar',
                 fusion_proj=True,
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
        self.history_window = history_window

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

        # Build CLF groups
        self.layers = nn.ModuleList()
        layer_id = 0
        for i_layer in range(self.num_layers):
            layer = PMDCLFB(
                dim=embed_dim,
                layer_id=layer_id,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads,
                num_topk=num_topk,
                window_size=window_size,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                history_window=history_window,
                gate_type=fusion_gate_type,
                fusion_proj=fusion_proj,
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

    def get_gate_stats(self):
        """min/mean/max of every layer's CLF gate tensor, keyed by global
        layer_id -- used by the staged smoke-test to confirm gates are
        moving from their zero-init (not dead) and not blowing up
        (not unstable)."""
        stats = {}
        for group in self.layers:
            for layer in group.residual_group.layers:
                if layer.clf is not None:
                    g = layer.clf.gate.detach()
                    stats[layer.layer_id] = (g.min().item(), g.mean().item(), g.max().item())
        return stats

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])

        pfa_values = [None, None]
        pfa_indices = [None, None]
        pfa_list = [pfa_values, pfa_indices]
        feat_history = []

        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed

        for layer in self.layers:
            x, pfa_list, feat_history = layer(x, pfa_list, feat_history, x_size, params)

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
    # NOTE: deliberately does NOT import basicsr.archs.promod_arch here --
    # doing so forces `import basicsr`, whose __init__.py auto-discovers and
    # re-imports every *_arch.py file (including this one, already running
    # as __main__), causing a duplicate @ARCH_REGISTRY.register() error.
    # For a side-by-side param/FLOPs diff against PMDModel, run a separate
    # `python3 -c "..."` snippet instead (see PROGRESS.md), which only
    # imports this file once via the normal package path.
    SHARED_CFG = dict(
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
    )

    clf_model = PMDCLFModel(**SHARED_CFG, history_window=3, fusion_gate_type='scalar', fusion_proj=True)
    clf_params = sum(p.nelement() for p in clf_model.parameters())

    x = torch.randn(1, 3, 64, 64)
    out = clf_model(x)

    print(f"Output shape: {out.shape}")
    print(f"PMDCLFModel params: {clf_params / 1e6:.3f}M")
    print(f"FLOPs (640x360): {clf_model.flops([640, 360]) / 1e9:.2f}G")
    print(f"FLOPs (320x180): {clf_model.flops([320, 180]) / 1e9:.2f}G")
    print(f"FLOPs (64x64, train patch): {clf_model.flops([64, 64]) / 1e9:.2f}G")

    gate_stats = clf_model.get_gate_stats()
    print(f"Gate stats (layer_id: min/mean/max), all should be exactly 0.0 at init:")
    for lid in sorted(gate_stats):
        mn, mean, mx = gate_stats[lid]
        print(f"  layer {lid:2d}: {mn:+.4f} / {mean:+.4f} / {mx:+.4f}")
