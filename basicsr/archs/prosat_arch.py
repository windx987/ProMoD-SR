'''
ProSAT: Progressive Selective Aggregation Transformer with Depth-Adaptive Routing.

SAT's density-driven K/V token aggregation (cluster_and_merge, ported from the
upstream SAT repo) combined with parameter-free Mixture-of-Depths routing driven
by progressive attention-importance scores (see ProSAT.md).

Routing is real gather/scatter (skipped tokens bypass attention and FFN via the
residual and are NOT computed), unlike ProMoD's mask-multiply. Routing top-k is
proportional (k = ceil(r*N)) so behavior is input-size invariant.

The GDFN gate's depthwise conv needs the full spatial map, so its hidden
features are scattered into a zero-filled full-size buffer for the conv and
gathered back; the FFN linears (the dominant cost) run on active tokens only.
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY


def build_prosat_schedule(total_layers, warmup_layers=2, disable=False, schedule=None):
    """Per-layer MoD capacity ratios (ProSAT.md §5).

    Light rule (16 layers): 2x1.0 warmup, 2x0.8 ramp, 4x0.6, rest 0.5.
    Standard rule extends with 0.4 from layer 17 on.
    An explicit `schedule` list overrides everything; disable=True gives all-1.0.
    """
    if disable:
        return [1.0] * total_layers
    if schedule is not None:
        assert len(schedule) == total_layers, \
            f'mod_schedule length {len(schedule)} != total layers {total_layers}'
        return list(schedule)
    out = []
    for l in range(total_layers):
        if l < warmup_layers:
            r = 1.0
        elif l < warmup_layers + 2:
            r = 0.8
        elif l < warmup_layers + 6:
            r = 0.6
        elif l < 16:
            r = 0.5
        else:
            r = 0.4
        out.append(r)
    return out


def cluster_and_merge(x, cluster_num, subsample_factor=4):
    # Ported unmodified from upstream SAT (sat_arch.py): density-driven token
    # aggregation. Compresses (B, N, C) -> (B, K, C) by selecting high
    # density*isolation cluster centers on a subsample, assigning every token
    # to its nearest center (cosine), and averaging within clusters.
    B, N, C = x.shape
    device = x.device
    K = cluster_num

    x_proj = x

    x_norm = F.normalize(x_proj, dim=-1)

    S = min(N, max(2 * K, subsample_factor * K))

    samples_per_region = S // K
    sub_idx = []
    for i in range(K):
        start_idx = i * (N // K)
        end_idx = (i + 1) * (N // K) if i < K - 1 else N
        region_size = end_idx - start_idx
        n_samples = min(samples_per_region, region_size)

        if region_size > 0:
            region_perm = torch.randperm(region_size, device=device)[:n_samples]
            sub_idx.append(start_idx + region_perm)

    sub_idx = torch.cat(sub_idx)
    if len(sub_idx) < S:
        remaining = S - len(sub_idx)
        all_idx = torch.arange(N, device=device)
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[sub_idx] = False
        additional = all_idx[mask][torch.randperm((~mask).sum(), device=device)[:remaining]]
        sub_idx = torch.cat([sub_idx, additional])

    x_norm_sub = x_norm[:, sub_idx]

    sim_sub = x_norm_sub @ x_norm_sub.transpose(1, 2)
    torch.diagonal(sim_sub, dim1=1, dim2=2).fill_(-1)

    k = min(K, S - 1)
    sim_topk_sub, _ = torch.topk(sim_sub, k=k, dim=-1)
    density_sub = sim_topk_sub.mean(dim=-1)
    density_sub = density_sub + torch.rand_like(density_sub) * 1e-6

    mask_higher_density = (density_sub[:, None, :] > density_sub[:, :, None]).float()

    masked_sim_sub = sim_sub * mask_higher_density - 1e9 * (1.0 - mask_higher_density)

    max_sim_to_higher, _ = masked_sim_sub.max(dim=-1)

    delta_sub = 1.0 - max_sim_to_higher

    max_density_mask_sub = (mask_higher_density.sum(dim=-1) == 0)

    min_sim_global = sim_sub.min(dim=-1)[0]
    max_dist_global = 1.0 - min_sim_global
    delta_sub[max_density_mask_sub] = max_dist_global[max_density_mask_sub]

    delta_sub = torch.clamp(delta_sub, min=0.0)

    score_sub = density_sub * delta_sub

    _, center_idx_in_sub = torch.topk(score_sub, k=K, dim=-1)

    center_idx = sub_idx[center_idx_in_sub]

    centers_norm = torch.gather(
        x_norm,
        1,
        center_idx[..., None].expand(B, K, x_norm.shape[-1])
    )

    sim_token_center = x_norm @ centers_norm.transpose(1, 2)

    assign_idx = sim_token_center.argmax(dim=-1)

    one_hot = F.one_hot(assign_idx, num_classes=K).type_as(x)

    cluster_counts = one_hot.sum(dim=1, keepdim=True).clamp(min=1e-6)

    out = torch.einsum("bnc,bnk->bkc", x, one_hot) / cluster_counts.transpose(1, 2)

    return out


class Gate(nn.Module):
    # Ported from upstream SAT. Operates on scattered-to-full spatial maps in
    # ProSAT so the depthwise conv always sees the complete H x W layout.
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x1, x2_full, H, W):
        # x1: (B, k, C) active gate input; x2_full: (B, N, C) hidden map with
        # zeros at skipped positions (k == N when routing is off).
        B, N, C = x2_full.shape
        x2 = self.conv(self.norm(x2_full).transpose(1, 2).contiguous().view(B, C, H, W))
        x2 = x2.flatten(2).transpose(-1, -2).contiguous()
        return x1, x2


class GDFN(nn.Module):
    """SAT's gated MLP (fc1 -> GELU -> split gate w/ DWConv -> fc2), adapted so
    the linears run on active tokens only while the gate conv runs full-size."""

    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.sg = Gate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, in_features)

    def forward(self, x_active, active_idx, x_size, n_tokens):
        # x_active: (B, k, C); active_idx: (B, k) or None (dense)
        H, W = x_size
        h = self.act(self.fc1(x_active))            # (B, k, hidden)
        x1, x2 = h.chunk(2, dim=-1)                 # (B, k, hidden/2) each
        half = self.hidden_features // 2
        if active_idx is None:
            x2_full = x2
        else:
            B = x_active.shape[0]
            x2_full = x2.new_zeros(B, n_tokens, half)
            x2_full.scatter_(1, active_idx[..., None].expand(-1, -1, half), x2)
        _, x2_conv = self.sg(x1, x2_full, H, W)     # (B, N, hidden/2)
        if active_idx is not None:
            x2_conv = torch.gather(x2_conv, 1, active_idx[..., None].expand(-1, -1, half))
        return self.fc2(x1 * x2_conv)               # (B, k, C)


class PSAA(nn.Module):
    """Progressive Selective Aggregation Attention (ProSAT.md §4).

    Q from active tokens only; K/V projected from DTA-aggregated tokens built
    over ALL tokens (skipped tokens still contribute context). Returns the
    head-averaged attention map for the importance update.
    """

    def __init__(self, dim, num_heads, qkv_bias=True, m_ratio=0.03, min_clusters=16):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.m_ratio = m_ratio
        self.min_clusters = min_clusters
        self.scale = (dim // num_heads) ** -0.5

        self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def n_clusters(self, n_tokens):
        return min(n_tokens, max(int(self.m_ratio * n_tokens), self.min_clusters))

    def forward(self, x_active, x_all):
        B, k, C = x_active.shape
        N = x_all.shape[1]
        h = self.num_heads
        d = C // h

        m = self.n_clusters(N)
        kv = cluster_and_merge(x_all, m)                       # (B, m, C)
        # Norm preservation (upstream SAT): rescale aggregated tokens to the
        # max token norm so averaging doesn't shrink K/V magnitudes.
        max_norm = torch.norm(x_all, dim=-1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        agg_norm = torch.norm(kv, dim=-1, keepdim=True)
        eps = 1e-6
        kv = torch.where(agg_norm > eps, (kv / (agg_norm + eps)) * max_norm, kv)

        q = self.w_q(x_active).reshape(B, k, h, d).permute(0, 2, 1, 3)
        key = self.w_k(kv).reshape(B, m, h, d).permute(0, 2, 1, 3)
        v = self.w_v(kv).reshape(B, m, h, d).permute(0, 2, 1, 3)

        attn = (q @ key.transpose(-2, -1)) * self.scale        # (B, h, k, m)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, k, C)
        out = self.proj(out)

        attn_map = attn.mean(dim=1)                            # (B, k, m)
        return out, attn_map


class PSAB(nn.Module):
    """ProSAT Block: importance-routed PSAA + GDFN with real token skipping."""

    def __init__(self, dim, num_heads, ffn_ratio=2.0, capacity_ratio=1.0,
                 qkv_bias=True, m_ratio=0.03, min_clusters=16,
                 active_gain=0.5, skip_decay=0.9, norm_layer=nn.LayerNorm):
        super().__init__()
        self.capacity_ratio = capacity_ratio
        self.active_gain = active_gain
        self.skip_decay = skip_decay
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = PSAA(dim, num_heads, qkv_bias=qkv_bias,
                         m_ratio=m_ratio, min_clusters=min_clusters)
        self.ffn = GDFN(dim, int(dim * ffn_ratio))

    def forward(self, x, x_size, importance):
        B, N, C = x.shape
        r = self.capacity_ratio
        k = min(N, max(1, math.ceil(r * N)))

        if k >= N:
            active_idx = None
        else:
            _, active_idx = torch.topk(importance, k, dim=1)   # (B, k)
        idx_c = None if active_idx is None else active_idx[..., None].expand(-1, -1, C)

        # --- attention on active tokens; K/V context from all tokens ---
        xn = self.norm1(x)
        xa = xn if idx_c is None else torch.gather(xn, 1, idx_c)
        attn_out, attn_map = self.attn(xa, xn)
        if idx_c is None:
            x = x + attn_out
        else:
            x = x.scatter_add(1, idx_c, attn_out)

        # --- FFN on active tokens ---
        xn2 = self.norm2(x)
        xa2 = xn2 if idx_c is None else torch.gather(xn2, 1, idx_c)
        ffn_out = self.ffn(xa2, active_idx, x_size, N)
        if idx_c is None:
            x = x + ffn_out
        else:
            x = x.scatter_add(1, idx_c, ffn_out)

        # --- importance update (detached: drives routing only, not the loss) ---
        score = attn_map.detach().max(dim=-1).values           # (B, k)
        if active_idx is None:
            importance = importance * (self.active_gain + self.active_gain * score)
        else:
            new_imp = importance * self.skip_decay
            active_imp = torch.gather(importance, 1, active_idx) \
                * (self.active_gain + self.active_gain * score)
            importance = new_imp.scatter(1, active_idx, active_imp)

        return x, importance

    def flops(self, n_tokens, m):
        # Same MAC-count convention as promod_arch. Attention/FFN linears scale
        # by the target capacity r; the gate's DWConv and DTA run full-size.
        r = self.capacity_ratio
        C = self.norm1.normalized_shape[0]
        hidden = self.ffn.hidden_features
        fl = 0
        # DTA: subsample sim (S=4m), token-center sim + merge
        S = min(n_tokens, 4 * m)
        fl += S * S * C + 2 * n_tokens * m * C
        # projections: Q on active, K/V + out on m / active
        fl += r * n_tokens * C * C          # w_q
        fl += 2 * m * C * C                 # w_k, w_v
        fl += r * n_tokens * C * C          # proj
        # attention
        fl += 2 * r * n_tokens * m * C
        # GDFN: fc1, gate conv (full), fc2
        fl += r * n_tokens * C * hidden
        fl += n_tokens * (hidden // 2) * 9
        fl += r * n_tokens * (hidden // 2) * C
        return fl


class ProSATGroup(nn.Module):
    """B blocks + 3x3 conv + group residual (ProSAT.md §2)."""

    def __init__(self, dim, depth, num_heads, ffn_ratio, capacity_ratios,
                 qkv_bias=True, m_ratio=0.03, min_clusters=16,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            PSAB(dim, num_heads, ffn_ratio=ffn_ratio,
                 capacity_ratio=capacity_ratios[i], qkv_bias=qkv_bias,
                 m_ratio=m_ratio, min_clusters=min_clusters,
                 norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size, importance):
        H, W = x_size
        B, N, C = x.shape
        res = x
        for blk in self.blocks:
            x, importance = blk(x, x_size, importance)
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return res + x, importance

    def flops(self, n_tokens, m):
        C = self.conv.in_channels
        fl = sum(blk.flops(n_tokens, m) for blk in self.blocks)
        fl += n_tokens * C * C * 9
        return fl


class Upsample(nn.Sequential):
    # Ported from upstream SAT: conv + pixelshuffle chain for 2^n and 3.
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super().__init__(*m)


@ARCH_REGISTRY.register()
class ProSAT(nn.Module):
    """ProSAT-Light default config (ProSAT.md §7): C=60, 4 groups x 4 blocks,
    6 heads, GDFN ratio 2.0, DTA 3% / min 16 clusters, MoD schedule
    1.0 -> 0.8 -> 0.6 -> 0.5 (warmup 2 layers).

    Routing is active from iteration 0 at the target capacity schedule (no
    iteration-driven ramp — matches ProMoD's convention).
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

        # --- 1. shallow feature extraction ---
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # --- 2. deep feature extraction ---
        self.norm_first = norm_layer(embed_dim)
        self.groups = nn.ModuleList()
        offset = 0
        for depth in depths:
            self.groups.append(ProSATGroup(
                embed_dim, depth, num_heads, ffn_ratio,
                capacity_ratios=self.capacity_schedule[offset:offset + depth],
                qkv_bias=qkv_bias, m_ratio=dta_m_ratio,
                min_clusters=dta_min_clusters, norm_layer=norm_layer))
            offset += depth
        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # --- 3. reconstruction (ProSAT.md §2: conv -> pixelshuffle -> conv) ---
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

    def forward_features(self, x):
        B, C, H, W = x.shape
        x_size = (H, W)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm_first(x)
        importance = x.new_ones(B, H * W)
        for group in self.groups:
            x, importance = group(x, x_size, importance)
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
        fl += n * 3 * self.embed_dim * 9                        # conv_first
        for group in self.groups:
            fl += group.flops(n, m)
        fl += n * self.embed_dim * self.embed_dim * 9           # conv_after_body
        fl += n * self.embed_dim * 64 * 9                       # conv_before_upsample
        scale = self.upscale
        if (scale & (scale - 1)) == 0:
            r = 1
            for _ in range(int(math.log(scale, 2))):
                fl += (n * r * r) * 64 * 64 * 4 * 9
                r *= 2
        elif scale == 3:
            fl += n * 64 * 64 * 9 * 9
        fl += (n * scale * scale) * 64 * 3 * 9                  # conv_last
        return fl


if __name__ == '__main__':
    model = ProSAT(upscale=2)
    params = sum(p.numel() for p in model.parameters())
    print(f'Params: {params / 1e6:.4f}M')
    print(f'Capacity schedule: {model.capacity_schedule}')
    print(f'FLOPs (640x360): {model.flops([640, 360]) / 1e9:.2f}G')
    x = torch.randn(1, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f'Forward: {tuple(x.shape)} -> {tuple(y.shape)}')
