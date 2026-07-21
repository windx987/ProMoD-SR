'''
ProMoD-MoE: PFT + Mixture-of-Depths + a soft, fully-dense multi-expert FFN.

Adds width-sparsity (MoE) as a capacity/quality axis, orthogonal to MoD's
depth-sparsity. Unlike MoD, this is NOT a FLOPs-reduction technique: every
expert is computed for every token (soft combination, no top-k, no
gather/scatter, no auxiliary load-balancing loss) — the same "dense compute,
mask/weight the output" philosophy that made MoD's v1.0 mask-multiply faster
than v1.1's real gather/scatter on GPU. Because the expert combination lives
entirely inside ConvFFN's dense computation, MoD's active_mask (still applied
externally to the FFN's output, unchanged) never interacts with it.
'''

import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs.pft_arch import (
    dwconv,
    window_partition, window_reverse,
    WindowAttention,
    PatchEmbed, PatchUnEmbed,
    Upsample, UpsampleOneStep,
)
from basicsr.archs.promod_arch import build_capacity_schedule


class SoftMoEConvFFN(nn.Module):
    """ConvFFN with a soft, fully-dense multi-expert fc2.

    fc1/act/dwconv stay single-copy and fully shared (no param growth,
    matches the "keep spatial-mixing dense" lesson learned from a related
    zero-fill bug elsewhere in this project). Only fc2 is replicated across
    num_experts independent branches, combined per-token by a softmax gate
    computed from the FFN's input (the same signal MoD's own router uses).
    No top-k, no auxiliary load-balancing loss: soft weighted combination
    is inherently balanced since every expert gets gradient every step.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 kernel_size=5, act_layer=nn.GELU, num_experts=2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.num_experts = num_experts

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.ModuleList([
            nn.Linear(hidden_features, out_features) for _ in range(num_experts)
        ])
        self.router = nn.Linear(in_features, num_experts, bias=False)

    def forward(self, x, x_size):
        gate = torch.softmax(self.router(x), dim=-1)  # [B, N, E]

        h = self.fc1(x)
        h = self.act(h)
        h = h + self.dwconv(h, x_size)

        out = gate[..., 0:1] * self.fc2[0](h)
        for e in range(1, self.num_experts):
            out = out + gate[..., e:e + 1] * self.fc2[e](h)
        return out


class PMDTLMoE(nn.Module):
    """PFT Transformer Layer with MoD routing + soft-MoE FFN.

    Identical to PMDTL (promod_arch.py) except convffn is SoftMoEConvFFN
    instead of ConvFFN. MoD's active_mask is still applied to the FFN's
    final output exactly as in PMDTL — the two mechanisms don't interact.
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
                 num_experts=2,
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
        self.num_experts = num_experts

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        if capacity_ratio < 1.0:
            self.router = nn.Linear(dim, 1, bias=False)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.convlepe_kernel_size = convffn_kernel_size
        self.v_LePE = dwconv(hidden_features=dim, kernel_size=self.convlepe_kernel_size)

        self.attn_win = WindowAttention(
            self.dim,
            layer_id=layer_id,
            window_size=(window_size, window_size),
            num_heads=num_heads,
            num_topk=num_topk,
            qkv_bias=qkv_bias,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = SoftMoEConvFFN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            kernel_size=convffn_kernel_size,
            act_layer=act_layer,
            num_experts=num_experts,
        )

    def _compute_routing_weights(self, x, h, w, b, n):
        import math
        scores = self.router(x)  # [B, N, 1]

        scores = scores.view(b, h, w, 1)
        if self.shift_size > 0:
            scores = torch.roll(scores, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        score_windows = window_partition(scores, self.window_size)
        win_n = self.window_size * self.window_size
        score_windows = score_windows.view(-1, win_n)

        k = max(1, int(math.ceil(self.capacity_ratio * win_n)))
        _, active_idx = torch.topk(score_windows, k, dim=-1)
        mask = torch.zeros_like(score_windows)
        mask.scatter_(1, active_idx, 1.0)

        weights = mask * torch.sigmoid(score_windows)

        weights = weights.view(-1, self.window_size, self.window_size, 1)
        weights = window_reverse(weights, self.window_size, h, w)
        if self.shift_size > 0:
            weights = torch.roll(weights, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return weights.reshape(b, n, 1)

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

        if self.capacity_ratio < 1.0:
            active_mask = self._compute_routing_weights(x, h, w, b, n)
        else:
            active_mask = None

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

        x_attn = attn_x.view(b, n, c)
        if active_mask is not None:
            x_attn = x_attn * active_mask
        x = shortcut + x_attn

        x_ffn = self.convffn(self.norm2(x), x_size)
        if active_mask is not None:
            x_ffn = x_ffn * active_mask
        x = x + x_ffn

        pfa_list = [pfa_values, pfa_indices]
        return x, pfa_list

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.input_resolution if input_resolution is None else input_resolution
        r = self.capacity_ratio
        e = self.num_experts

        flops += self.dim * 3 * self.dim * h * w

        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn_win.flops(self.window_size * self.window_size) * r

        # fc1 (dense) + dwconv (dense) + e dense fc2 branches (no r scaling: MoE
        # runs on every token regardless of MoD's mask) + router (negligible)
        flops += h * w * self.dim * self.dim * self.mlp_ratio  # fc1
        flops += h * w * self.dim * (self.convffn_kernel_size ** 2) * self.mlp_ratio  # dwconv
        flops += e * h * w * self.dim * self.dim * self.mlp_ratio  # e x fc2
        flops += h * w * self.dim * e  # router
        flops += h * w * self.dim * (self.convlepe_kernel_size ** 2)  # LePE

        return flops


class PMDBBMoE(nn.Module):
    """Container of ProMoD-MoE transformer layers. Carries importance across layers."""

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
                 num_experts=2,
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
                PMDTLMoE(
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
                    num_experts=num_experts,
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


class PMDBMoE(nn.Module):
    """ProMoD-MoE group block (PFTB equivalent with MoD + soft-MoE FFN)."""

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
                 num_experts=2,
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

        self.residual_group = PMDBBMoE(
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
            num_experts=num_experts,
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
class PMDMoEModel(nn.Module):
    """ProMoD-MoE: PFT + Mixture-of-Depths + soft dense multi-expert FFN."""

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
                 num_experts=2,
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

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

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
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)

        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        self.layers = nn.ModuleList()
        layer_id = 0
        for i_layer in range(self.num_layers):
            layer = PMDBMoE(
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
                num_experts=num_experts,
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
            nn.init.trunc_normal_(m.weight, std=.02)
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
            x = self.lrelu(self.conv_up1(nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
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
    model = PMDMoEModel(
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
        num_experts=2,
    )

    total = sum([param.nelement() for param in model.parameters()])
    print(f"Number of parameters: {total / 1e6:.3f}M")
    print(f"Capacity schedule: {model.capacity_schedule}")
    print(f"FLOPs (640x360): {model.flops([640, 360]) / 1e9:.2f}G")
    print(f"FLOPs (320x180): {model.flops([320, 180]) / 1e9:.2f}G")
