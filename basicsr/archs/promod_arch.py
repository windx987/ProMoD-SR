'''
ProMoD: Progressive Focused Attention with Mixture of Depths for Image Super-Resolution.

Based on PFT (Progressive Focused Transformer), adding depth-adaptive token routing
using a lightweight learned router (nn.Linear(dim, 1)) per routing layer.
Routing scores are computed from the current layer's normed input — gradients flow
through the router every step, preventing the routing collapse seen with
attention-score-based routing.
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs.pft_arch import (
    SMM_QmK, SMM_AmV,
    dwconv, ConvFFN,
    window_partition, window_reverse,
    WindowAttention,
    PatchEmbed, PatchUnEmbed,
    Upsample, UpsampleOneStep,
)


def build_capacity_schedule(total_layers, warmup_layers=2, disable=False):
    """Build progressive MoD capacity schedule.

    Returns a list of capacity ratios (r) per layer index.
    With disable=True all layers get r=1.0 (identical to stock PFT).
    """
    if disable:
        return [1.0] * total_layers
    schedule = []
    for l in range(total_layers):
        if l < warmup_layers:
            r = 1.0
        elif l < 8:
            r = 0.875
        elif l < 16:
            r = 0.75
        else:
            r = 0.625
        schedule.append(r)
    return schedule


class PMDTL(nn.Module):
    """PFT Transformer Layer with Mixture-of-Depths routing.

    Adds MoD token skipping on top of PFT's progressive focused attention.
    Skipped tokens bypass attention and FFN via residual connection.
    All tokens remain in K/V for context preservation.
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

        # Learned router: scores tokens from current layer's normed input.
        # Only instantiated for routing layers (capacity_ratio < 1.0).
        if capacity_ratio < 1.0:
            self.router = nn.Linear(dim, 1, bias=False)

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

    def _compute_mod_mask(self, importance, b, n, device):
        """Compute binary A-MoD active mask via top-k selection."""
        if self.capacity_ratio >= 1.0:
            return torch.ones(b, n, 1, device=device)

        k = max(1, int(math.ceil(self.capacity_ratio * n)))
        _, active_idx = torch.topk(importance, k, dim=-1)
        mask = torch.zeros(b, n, 1, device=device)
        mask.scatter_(1, active_idx.unsqueeze(-1), 1.0)
        return mask

    def forward(self, x, pfa_list, x_size, params):
        pfa_values, pfa_indices = pfa_list[0], pfa_list[1]
        h, w = x_size
        b, n, c = x.shape
        c4 = 4 * c

        shortcut = x
        x = self.norm1(x)
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

        # --- LR-MoD Routing: learned router scores from current layer's normed input ---
        if self.capacity_ratio < 1.0:
            scores = self.router(x).squeeze(-1)  # x is already norm1'd, [B, N]
            active_mask = self._compute_mod_mask(scores, b, n, x.device)
        else:
            active_mask = torch.ones(b, n, 1, device=x.device)

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c4)

        # W-MSA/SW-MSA
        attn_windows, pfa_values, pfa_indices = self.attn_win(
            x_windows,
            pfa_values=pfa_values,
            pfa_indices=pfa_indices,
            rpi=params['rpi_sa'],
            mask=params['attn_mask'],
            shift=shift,
        )

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x

        # --- Apply MoD mask to attention output ---
        x_attn = attn_x.view(b, n, c) * active_mask
        x = shortcut + x_attn

        # --- FFN with MoD mask ---
        x_ffn = self.convffn(self.norm2(x), x_size) * active_mask
        x = x + x_ffn

        pfa_list = [pfa_values, pfa_indices]
        return x, pfa_list

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.input_resolution if input_resolution is None else input_resolution
        r = self.capacity_ratio

        flops += self.dim * 3 * self.dim * h * w

        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn_win.flops(self.window_size * self.window_size) * r

        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio * r
        flops += h * w * self.dim * (self.convffn_kernel_size ** 2) * self.mlp_ratio * r
        flops += h * w * self.dim * (self.convlepe_kernel_size ** 2)

        return flops


class PMDBB(nn.Module):
    """Container of ProMoD transformer layers. Carries importance across layers."""

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
                PMDTL(
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


class PMDB(nn.Module):
    """ProMoD group block (PFTB equivalent with MoD)."""

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

        self.residual_group = PMDBB(
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
class PMDModel(nn.Module):
    """ProMoD: Progressive Focused Attention with Mixture of Depths for Image Super-Resolution.

    PFT + MoD with PFT's progressive attention cascade as a free routing signal.
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
        self.capacity_schedule = build_capacity_schedule(total_layers, mod_warmup_layers, disable=mod_disable)

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

        # Build ProMoD groups
        self.layers = nn.ModuleList()
        layer_id = 0
        for i_layer in range(self.num_layers):
            layer = PMDB(
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
        if self.upsampler == 'pixelshuffle':
            flops += self.upsample.flops(resolution)
        else:
            flops += self.upsample.flops(resolution)
        return flops


if __name__ == '__main__':
    model = PMDModel(
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
