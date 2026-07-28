'''
ProSAT-v2: ProSAT's DTA-merged global attention, with ProMoD v1.1's actual
(trainable) Mixture-of-Depths mechanism in place of ProSAT's own
parameter-free routing.

Two things changed vs `prosat_arch.py`, both necessary together (see this
experiment's plan for the full investigation):

1. ProSAT's `importance` heuristic (a parameter-free running product,
   `score = attn_map.detach()...`) is replaced by an independent learned
   `nn.Linear(dim, 1, bias=False)` per routed block, scoring fresh from
   that block's own `norm1(x)` -- exactly v1.0/v1.1's convention (no
   cross-block routing state at all, unlike ProSAT's `importance`
   threading). Critically, this also required adding the multiplicative
   gate v1.0/v1.1 use (`weights = mask * sigmoid(score)`, multiplying the
   block's output before the residual add) -- ProSAT's `importance` never
   had one (`x = x.scatter_add(1, idx_c, attn_out)`, no `* weights`
   anywhere), which is *why* it had to be parameter-free: `topk` itself is
   non-differentiable, and without a gate-multiply step downstream of it,
   a learned router would receive literally zero gradient. Adding the
   router without the gate would have just been uselessly parameterized,
   not actually trainable.
2. GDFN's gate side-channel scattered inactive positions as literal zero
   before its depthwise conv (`prosat_arch.py` `GDFN.forward`), corrupting
   the conv's output for active tokens next to a skipped one -- documented
   as the actual root cause of ProSAT's iter-50K flat-loss stall. Fixed
   here with partial (mask-renormalized) convolution: convolve the binary
   active-mask with an all-ones kernel to get each position's actual
   active-neighbor count, and rescale the conv output by
   `kernel_area / neighbor_count` instead of silently accepting whatever a
   zero-padded neighborhood produces. This is a correctness fix for a known
   bug class (not a new architectural mechanism), unlike the router change
   above, which is this project's own established convention.

`cluster_and_merge` (DTA), `PSAA` (the global attention itself), and
`Upsample` are imported UNCHANGED from `prosat_arch.py` -- validated,
untouched by this change.
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs.prosat_arch import (
    build_prosat_schedule,
    cluster_and_merge,
    PSAA,
    Upsample,
)


class GateV2(nn.Module):
    """Depthwise-conv gate branch with partial-convolution renormalization,
    so scattering inactive positions as zero before the conv doesn't bias
    the output for active tokens sitting next to a skipped one.
    """

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # bias=False: the standard Partial Convolution formula (Liu et al.)
        # rescales only the weight-sum contribution by kernel_area/valid_count;
        # the bias is a constant per-channel offset added AFTER rescaling, not
        # itself rescaled. Folding bias into nn.Conv2d and rescaling the whole
        # output (weight-sum + bias together) was a real bug caught by this
        # experiment's unit test -- see plan's staged-verification step 1.
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, groups=dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer('ones_kernel', torch.ones(1, 1, kernel_size, kernel_size))
        self.kernel_area = float(kernel_size * kernel_size)
        self.padding = kernel_size // 2

    def forward(self, x2_full, mask, H, W):
        # x2_full: (B, N, C) hidden map, zero-filled at inactive positions
        # (or genuinely dense if mask is None, i.e. this block isn't routing).
        # mask: (B, N, 1) binary active mask, or None.
        B, N, C = x2_full.shape
        xn = self.norm(x2_full).transpose(1, 2).contiguous().view(B, C, H, W)
        conv_out = self.conv(xn)  # weight-sum only, no bias yet
        if mask is not None:
            mask_img = mask.transpose(1, 2).contiguous().view(B, 1, H, W)
            neighbor_count = F.conv2d(mask_img, self.ones_kernel, padding=self.padding)
            conv_out = conv_out * (self.kernel_area / (neighbor_count + 1e-6))
        conv_out = conv_out + self.bias.view(1, C, 1, 1)
        return conv_out.flatten(2).transpose(-1, -2).contiguous()


class GDFNv2(nn.Module):
    """Same split as ProSAT's GDFN (fc1 on active tokens only, narrow gate
    side-channel needs the full spatial map) -- gate side-channel now uses
    GateV2's partial convolution instead of a naive zero-fill.
    """

    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.sg = GateV2(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, in_features)

    def forward(self, x_active, active_idx, x_size, n_tokens):
        H, W = x_size
        h = self.act(self.fc1(x_active))            # (B, k, hidden)
        x1, x2 = h.chunk(2, dim=-1)                  # (B, k, hidden/2) each
        half = self.hidden_features // 2
        if active_idx is None:
            x2_conv = self.sg(x2, None, H, W)
        else:
            B = x_active.shape[0]
            x2_full = x2.new_zeros(B, n_tokens, half)
            x2_full.scatter_(1, active_idx[..., None].expand(-1, -1, half), x2)
            mask = x2.new_zeros(B, n_tokens, 1)
            mask.scatter_(1, active_idx[..., None], 1.0)
            x2_conv = self.sg(x2_full, mask, H, W)
            x2_conv = torch.gather(x2_conv, 1, active_idx[..., None].expand(-1, -1, half))
        return self.fc2(x1 * x2_conv)                # (B, k, C)


class PSABv2(nn.Module):
    """ProSAT block with v1.1-style trainable routing: independent learned
    router per block (no cross-block importance state), sigmoid gate
    multiplying the output before the residual add so the router actually
    receives a gradient.
    """

    def __init__(self, dim, num_heads, ffn_ratio=2.0, capacity_ratio=1.0,
                 qkv_bias=True, m_ratio=0.03, min_clusters=16,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.capacity_ratio = capacity_ratio
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = PSAA(dim, num_heads, qkv_bias=qkv_bias,
                         m_ratio=m_ratio, min_clusters=min_clusters)
        self.ffn = GDFNv2(dim, int(dim * ffn_ratio))

        if capacity_ratio < 1.0:
            self.router = nn.Linear(dim, 1, bias=False)

    def forward(self, x, x_size):
        B, N, C = x.shape
        r = self.capacity_ratio
        k = min(N, max(1, math.ceil(r * N)))

        xn = self.norm1(x)

        if r >= 1.0:
            active_idx, weights = None, None
        else:
            score = self.router(xn).squeeze(-1)                      # [B, N], real gradient
            _, active_idx = torch.topk(score, k, dim=1)
            mask = torch.zeros_like(score)
            mask.scatter_(1, active_idx, 1.0)
            weights = mask * torch.sigmoid(score)                    # [B, N]

        idx_c = None if active_idx is None else active_idx[..., None].expand(-1, -1, C)

        # --- attention on active tokens; DTA-merged K/V context from ALL tokens ---
        xa = xn if idx_c is None else torch.gather(xn, 1, idx_c)
        attn_out, _ = self.attn(xa, xn)
        if idx_c is None:
            x = x + attn_out
        else:
            w = torch.gather(weights, 1, active_idx).unsqueeze(-1)    # [B, k, 1]
            x = x.scatter_add(1, idx_c, attn_out * w)

        # --- FFN on active tokens ---
        xn2 = self.norm2(x)
        xa2 = xn2 if idx_c is None else torch.gather(xn2, 1, idx_c)
        ffn_out = self.ffn(xa2, active_idx, x_size, N)
        if idx_c is None:
            x = x + ffn_out
        else:
            x = x.scatter_add(1, idx_c, ffn_out * w)

        return x

    def flops(self, n_tokens, m):
        r = self.capacity_ratio
        C = self.norm1.normalized_shape[0]
        hidden = self.ffn.hidden_features
        fl = 0
        S = min(n_tokens, 4 * m)
        fl += S * S * C + 2 * n_tokens * m * C          # DTA
        fl += r * n_tokens * C * C                      # w_q
        fl += 2 * m * C * C                             # w_k, w_v
        fl += r * n_tokens * C * C                      # attn.proj
        fl += 2 * r * n_tokens * m * C                  # attention matmuls
        fl += r * n_tokens * C * hidden                 # fc1 (routed -- real savings)
        fl += n_tokens * (hidden // 2) * 9              # gate conv, dense (always full grid)
        fl += r * n_tokens * (hidden // 2) * C          # fc2 (routed)
        if r < 1.0:
            fl += n_tokens * C                          # router linear, negligible
        return fl


class ProSATGroupv2(nn.Module):
    """B blocks + 3x3 conv + group residual. No `importance` threading."""

    def __init__(self, dim, depth, num_heads, ffn_ratio, capacity_ratios,
                 qkv_bias=True, m_ratio=0.03, min_clusters=16,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            PSABv2(dim, num_heads, ffn_ratio=ffn_ratio,
                   capacity_ratio=capacity_ratios[i], qkv_bias=qkv_bias,
                   m_ratio=m_ratio, min_clusters=min_clusters,
                   norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        H, W = x_size
        B, N, C = x.shape
        res = x
        for blk in self.blocks:
            x = blk(x, x_size)
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return res + x

    def flops(self, n_tokens, m):
        C = self.conv.in_channels
        fl = sum(blk.flops(n_tokens, m) for blk in self.blocks)
        fl += n_tokens * C * C * 9
        return fl


@ARCH_REGISTRY.register()
class ProSATv2(nn.Module):
    """ProSAT-v2-Light: same defaults as ProSAT (`prosat_arch.py`) -- C=60,
    4 groups x 4 blocks, 6 heads, GDFN ratio 2.0, DTA 3% / min 16 clusters,
    same MoD schedule -- with the trainable-router + partial-conv fixes
    described in this file's module docstring.
    """

    def __init__(self,
                 in_chans=3,
                 embed_dim=60,
                 depths=(4, 4, 4, 4),
                 num_heads=6,
                 ffn_ratio=2.0,
                 qkv_bias=True,
                 dta_m_ratio=0.03,
                 dta_min_clusters=16,
                 mod_warmup_layers=2,
                 mod_disable=False,
                 mod_schedule=None,
                 norm_layer=nn.LayerNorm,
                 upscale=2,
                 img_range=1.,
                 **kwargs):
        super().__init__()

        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.embed_dim = embed_dim
        self.dta_m_ratio = dta_m_ratio
        self.dta_min_clusters = dta_min_clusters

        total_layers = sum(depths)
        self.capacity_schedule = build_prosat_schedule(
            total_layers, mod_warmup_layers, disable=mod_disable, schedule=mod_schedule)

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        self.norm_first = norm_layer(embed_dim)
        self.groups = nn.ModuleList()
        offset = 0
        for depth in depths:
            self.groups.append(ProSATGroupv2(
                embed_dim, depth, num_heads, ffn_ratio,
                capacity_ratios=self.capacity_schedule[offset:offset + depth],
                qkv_bias=qkv_bias, m_ratio=dta_m_ratio,
                min_clusters=dta_min_clusters, norm_layer=norm_layer))
            offset += depth
        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(upscale, num_feat)
        self.conv_last = nn.Conv2d(num_feat, in_chans, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def get_router_stats(self):
        """min/mean/max of each routed block's router weight, for smoke-test telemetry."""
        stats = {}
        for gi, group in enumerate(self.groups):
            for bi, blk in enumerate(group.blocks):
                if hasattr(blk, 'router'):
                    w = blk.router.weight.detach()
                    stats[f'g{gi}b{bi}'] = (w.min().item(), w.mean().item(), w.max().item())
        return stats

    def forward_features(self, x):
        B, C, H, W = x.shape
        x_size = (H, W)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm_first(x)
        for group in self.groups:
            x = group(x, x_size)
        x = self.norm(x)
        x = x.transpose(1, 2).contiguous().view(B, self.embed_dim, H, W)
        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean
        return x

    def flops(self, input_resolution):
        h, w = input_resolution
        n = h * w
        m = min(n, max(int(self.dta_m_ratio * n), self.dta_min_clusters))
        fl = 0
        fl += n * 3 * self.embed_dim * 9
        for group in self.groups:
            fl += group.flops(n, m)
        fl += n * self.embed_dim * self.embed_dim * 9
        fl += n * self.embed_dim * 64 * 9
        scale = self.upscale
        if (scale & (scale - 1)) == 0:
            r = 1
            for _ in range(int(math.log(scale, 2))):
                fl += (n * r * r) * 64 * 64 * 4 * 9
                r *= 2
        elif scale == 3:
            fl += n * 64 * 64 * 9 * 9
        fl += (n * scale * scale) * 64 * 3 * 9
        return fl


if __name__ == '__main__':
    model = ProSATv2(upscale=2)
    params = sum(p.numel() for p in model.parameters())
    print(f'Params: {params / 1e6:.4f}M')
    print(f'Capacity schedule: {model.capacity_schedule}')
    print(f'FLOPs (640x360): {model.flops([640, 360]) / 1e9:.2f}G')
    x = torch.randn(1, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f'Forward: {tuple(x.shape)} -> {tuple(y.shape)}')
    print(f'Router stats (should be nonzero-init, non-degenerate): {model.get_router_stats()}')
