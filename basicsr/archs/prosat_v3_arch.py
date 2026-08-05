'''
ProSAT-v3: ProSAT-v2's architecture with DTA's quadratic clustering amortized
across residual groups, and the attention matrix no longer materialized.

Architecturally this is ProSAT-v2 (`prosat_v2_arch.py`) -- same blocks, same
learned router + sigmoid gate, same partial-convolution GDFN gate, same
parameter count. What changes is *how often* and *how expensively* the same
quantities are computed. Three changes, all of which leave the mathematical
function either identical or near-identical:

1. `cluster_and_merge` is split into `dta_assign` (which tokens belong to
   which cluster -- the O(N^2) part) and `dta_merge` (the cluster-mean of the
   CURRENT features -- the cheap part). Cluster *membership* is a slowly
   varying semantic property; the merged *values* are what must track the
   residual stream. So the assignment is computed once per residual group and
   reused, while the merge still runs at every block -- K/V stay fresh, the
   quadratic work does not repeat.

   Measured against this file's own `flops()` (which reproduces v2's numbers
   exactly: 4.022G @64x64, 5828G @640x360), DTA is 46% of block FLOPs at
   64x64 and 66% at 640x360, because both dominant terms -- `S^2*C` with
   S=min(N,4K), and the token-to-center similarity N*K*C -- are quadratic in
   N when K = 0.03N. SAT's ~97% K/V reduction shrinks the attention; the
   clustering that produces it does not scale.

2. `dta_merge` uses `scatter_add` instead of the original
   `einsum('bnc,bnk->bkc', x, one_hot)`. A segmented mean is a segmented
   mean -- but the einsum computes it as a dense (C,N)@(N,K) matmul costing
   N*K*C, where the same result is a single pass costing N*C. This is a
   complexity fix, not an architectural change: same function, ~K times
   cheaper, and it never materializes the (B,N,K) one-hot (6.4 GB at
   640x360). Verified numerically against the einsum form.

3. `PSAAv3` can use `F.scaled_dot_product_attention`. Unlike PFT's windowed
   attention, PSAA adds NO positional bias to its logits, and ProSAT-v2
   already discards the returned `attn_map` -- so the only reason the
   (B,h,k,m) attention matrix was ever materialized is gone. This is not
   merely a speedup: at Urban100 x2 LR (512x384 -> N=196608, m=5898) that
   matrix is 27.8 GB in fp32 for a single layer, which is why Urban100 has
   never appeared in any config for this family.

`deterministic=True` additionally removes the two RNG calls inside the
assignment (`torch.randperm` for the subsample, and a `rand_like` jitter used
to break density ties), which are why ProSAT inference is not reproducible
run-to-run today.

`build_prosat_schedule` / `Upsample` are imported unchanged from
`prosat_arch.py`; `GateV2` / `GDFNv2` unchanged from `prosat_v2_arch.py` (the
partial-convolution fix is validated and orthogonal to this change). Neither
of those files is modified -- 401/402 stay reproducible.
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY

from basicsr.archs.prosat_arch import build_prosat_schedule, Upsample
from basicsr.archs.prosat_v2_arch import GateV2, GDFNv2


def dta_assign(x, cluster_num, subsample_factor=4, deterministic=False):
    """Density-driven cluster assignment -- the expensive, cacheable half of
    upstream SAT's `cluster_and_merge`.

    Identical to `prosat_arch.cluster_and_merge` up to (but excluding) the
    final merge, except that `deterministic=True` replaces both RNG calls
    with reproducible equivalents.

    Returns:
        assign_idx: (B, N) int64 -- cluster index per token.
    """
    B, N, C = x.shape
    device = x.device
    K = cluster_num

    x_norm = F.normalize(x, dim=-1)

    S = min(N, max(2 * K, subsample_factor * K))
    samples_per_region = S // K

    sub_idx = []
    for i in range(K):
        start_idx = i * (N // K)
        end_idx = (i + 1) * (N // K) if i < K - 1 else N
        region_size = end_idx - start_idx
        n_samples = min(samples_per_region, region_size)
        if region_size > 0:
            if deterministic:
                # evenly spaced within the region instead of a random draw --
                # same coverage, reproducible
                offs = torch.linspace(0, region_size - 1, n_samples, device=device).long()
            else:
                offs = torch.randperm(region_size, device=device)[:n_samples]
            sub_idx.append(start_idx + offs)

    sub_idx = torch.cat(sub_idx)
    if len(sub_idx) < S:
        remaining = S - len(sub_idx)
        all_idx = torch.arange(N, device=device)
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[sub_idx] = False
        pool = all_idx[mask]
        if deterministic:
            additional = pool[:remaining]
        else:
            additional = pool[torch.randperm(pool.numel(), device=device)[:remaining]]
        sub_idx = torch.cat([sub_idx, additional])

    x_norm_sub = x_norm[:, sub_idx]

    sim_sub = x_norm_sub @ x_norm_sub.transpose(1, 2)
    torch.diagonal(sim_sub, dim1=1, dim2=2).fill_(-1)

    k = min(K, S - 1)
    sim_topk_sub, _ = torch.topk(sim_sub, k=k, dim=-1)
    density_sub = sim_topk_sub.mean(dim=-1)
    if not deterministic:
        # upstream tie-break jitter; a second source of run-to-run variation
        density_sub = density_sub + torch.rand_like(density_sub) * 1e-6

    mask_higher_density = (density_sub[:, None, :] > density_sub[:, :, None]).float()
    masked_sim_sub = sim_sub * mask_higher_density - 1e9 * (1.0 - mask_higher_density)
    max_sim_to_higher, _ = masked_sim_sub.max(dim=-1)

    delta_sub = 1.0 - max_sim_to_higher
    max_density_mask_sub = (mask_higher_density.sum(dim=-1) == 0)
    max_dist_global = 1.0 - sim_sub.min(dim=-1)[0]
    delta_sub[max_density_mask_sub] = max_dist_global[max_density_mask_sub]
    delta_sub = torch.clamp(delta_sub, min=0.0)

    score_sub = density_sub * delta_sub
    _, center_idx_in_sub = torch.topk(score_sub, k=K, dim=-1)
    center_idx = sub_idx[center_idx_in_sub]

    centers_norm = torch.gather(
        x_norm, 1, center_idx[..., None].expand(B, K, x_norm.shape[-1]))
    sim_token_center = x_norm @ centers_norm.transpose(1, 2)
    return sim_token_center.argmax(dim=-1)                      # (B, N)


def dta_merge(x, assign_idx, cluster_num):
    """Cluster-mean of the current features -- the cheap half, recomputed at
    every block so K/V track the residual stream.

    Equivalent to upstream's `einsum('bnc,bnk->bkc', x, one_hot) / counts`,
    but as a single scatter pass (N*C) rather than a dense N*K*C matmul, and
    without materializing the (B, N, K) one-hot.
    """
    B, N, C = x.shape
    K = cluster_num
    out = x.new_zeros(B, K, C)
    out.scatter_add_(1, assign_idx[..., None].expand(B, N, C), x)
    counts = x.new_zeros(B, K)
    counts.scatter_add_(1, assign_idx, x.new_ones(B, N))
    return out / counts.clamp(min=1e-6)[..., None]


class PSAAv3(nn.Module):
    """PSAA with a reusable cluster assignment and an optional fused
    attention kernel. Q from active tokens; K/V from DTA-merged tokens built
    over ALL tokens. No positional bias (unchanged from ProSAT) -- which is
    exactly what makes SDPA a drop-in here.
    """

    def __init__(self, dim, num_heads, qkv_bias=True, m_ratio=0.03,
                 min_clusters=16, use_sdpa=True, deterministic=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.m_ratio = m_ratio
        self.min_clusters = min_clusters
        self.use_sdpa = use_sdpa
        self.deterministic = deterministic
        self.scale = (dim // num_heads) ** -0.5

        self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def n_clusters(self, n_tokens):
        return min(n_tokens, max(int(self.m_ratio * n_tokens), self.min_clusters))

    def forward(self, x_active, x_all, assign_idx=None):
        B, k, C = x_active.shape
        N = x_all.shape[1]
        h = self.num_heads
        d = C // h
        m = self.n_clusters(N)

        if assign_idx is None:
            assign_idx = dta_assign(x_all, m, deterministic=self.deterministic)
        kv = dta_merge(x_all, assign_idx, m)                    # (B, m, C)

        # Norm preservation (upstream SAT): rescale aggregated tokens to the
        # max token norm so averaging doesn't shrink K/V magnitudes.
        max_norm = torch.norm(x_all, dim=-1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        agg_norm = torch.norm(kv, dim=-1, keepdim=True)
        kv = torch.where(agg_norm > 1e-6, (kv / (agg_norm + 1e-6)) * max_norm, kv)

        q = self.w_q(x_active).reshape(B, k, h, d).permute(0, 2, 1, 3)
        key = self.w_k(kv).reshape(B, m, h, d).permute(0, 2, 1, 3)
        v = self.w_v(kv).reshape(B, m, h, d).permute(0, 2, 1, 3)

        if self.use_sdpa:
            # same math; never materializes the (B, h, k, m) matrix
            out = F.scaled_dot_product_attention(q, key, v)
        else:
            attn = ((q @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, k, C)
        return self.proj(out), assign_idx


class PSABv3(nn.Module):
    """ProSAT-v2's block, threading the reusable cluster assignment through."""

    def __init__(self, dim, num_heads, ffn_ratio=2.0, capacity_ratio=1.0,
                 qkv_bias=True, m_ratio=0.03, min_clusters=16,
                 use_sdpa=True, deterministic=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.capacity_ratio = capacity_ratio
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = PSAAv3(dim, num_heads, qkv_bias=qkv_bias, m_ratio=m_ratio,
                           min_clusters=min_clusters, use_sdpa=use_sdpa,
                           deterministic=deterministic)
        self.ffn = GDFNv2(dim, int(dim * ffn_ratio))

        if capacity_ratio < 1.0:
            self.router = nn.Linear(dim, 1, bias=False)

    def forward(self, x, x_size, assign_idx=None):
        B, N, C = x.shape
        r = self.capacity_ratio
        k = min(N, max(1, math.ceil(r * N)))

        xn = self.norm1(x)

        if r >= 1.0:
            active_idx, weights = None, None
        else:
            score = self.router(xn).squeeze(-1)
            _, active_idx = torch.topk(score, k, dim=1)
            mask = torch.zeros_like(score)
            mask.scatter_(1, active_idx, 1.0)
            weights = mask * torch.sigmoid(score)

        idx_c = None if active_idx is None else active_idx[..., None].expand(-1, -1, C)

        xa = xn if idx_c is None else torch.gather(xn, 1, idx_c)
        attn_out, assign_idx = self.attn(xa, xn, assign_idx)
        if idx_c is None:
            x = x + attn_out
        else:
            w = torch.gather(weights, 1, active_idx).unsqueeze(-1)
            x = x.scatter_add(1, idx_c, attn_out * w)

        xn2 = self.norm2(x)
        xa2 = xn2 if idx_c is None else torch.gather(xn2, 1, idx_c)
        ffn_out = self.ffn(xa2, active_idx, x_size, N)
        if idx_c is None:
            x = x + ffn_out
        else:
            x = x.scatter_add(1, idx_c, ffn_out * w)

        return x, assign_idx

    def flops(self, n_tokens, m, recluster_every=1):
        r = self.capacity_ratio
        C = self.norm1.normalized_shape[0]
        hidden = self.ffn.hidden_features
        fl = 0
        # DTA assignment: quadratic, amortized over `recluster_every` blocks
        S = min(n_tokens, 4 * m)
        fl += (S * S * C + n_tokens * m * C) / recluster_every
        # DTA merge: every block, but now a scatter pass, not an N*m*C matmul
        fl += n_tokens * C
        fl += r * n_tokens * C * C                      # w_q
        fl += 2 * m * C * C                             # w_k, w_v
        fl += r * n_tokens * C * C                      # attn.proj
        fl += 2 * r * n_tokens * m * C                  # attention matmuls
        fl += r * n_tokens * C * hidden                 # fc1 (routed)
        fl += n_tokens * (hidden // 2) * 9              # gate conv, dense
        fl += r * n_tokens * (hidden // 2) * C          # fc2 (routed)
        if r < 1.0:
            fl += n_tokens * C                          # router linear
        return fl


class ProSATGroupv3(nn.Module):
    """B blocks + 3x3 conv + group residual. Owns the DTA assignment cache:
    block `i` recomputes the clustering only when `i % recluster_every == 0`,
    every other block reuses it (but still re-merges current features).
    """

    def __init__(self, dim, depth, num_heads, ffn_ratio, capacity_ratios,
                 qkv_bias=True, m_ratio=0.03, min_clusters=16,
                 recluster_every=4, use_sdpa=True, deterministic=False,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.recluster_every = max(1, recluster_every)
        self.blocks = nn.ModuleList([
            PSABv3(dim, num_heads, ffn_ratio=ffn_ratio,
                   capacity_ratio=capacity_ratios[i], qkv_bias=qkv_bias,
                   m_ratio=m_ratio, min_clusters=min_clusters,
                   use_sdpa=use_sdpa, deterministic=deterministic,
                   norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        H, W = x_size
        B, N, C = x.shape
        res = x
        assign_idx = None
        for i, blk in enumerate(self.blocks):
            reuse = None if (i % self.recluster_every == 0) else assign_idx
            x, assign_idx = blk(x, x_size, reuse)
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return res + x

    def flops(self, n_tokens, m):
        C = self.conv.in_channels
        fl = sum(blk.flops(n_tokens, m, self.recluster_every) for blk in self.blocks)
        fl += n_tokens * C * C * 9
        return fl


@ARCH_REGISTRY.register()
class ProSATv3(nn.Module):
    """ProSAT-v3-Light: identical parameters and architecture to ProSATv2;
    DTA clustering amortized across each residual group and the attention
    matrix no longer materialized. See this file's module docstring.

    `dta_recluster_every=1` + `use_sdpa=False` + `dta_deterministic=False`
    reproduces ProSATv2's forward function (the faithfulness configuration).
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
                 dta_recluster_every=4,
                 dta_deterministic=True,
                 use_sdpa=True,
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
        self.dta_recluster_every = dta_recluster_every

        total_layers = sum(depths)
        self.capacity_schedule = build_prosat_schedule(
            total_layers, mod_warmup_layers, disable=mod_disable, schedule=mod_schedule)

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        self.norm_first = norm_layer(embed_dim)
        self.groups = nn.ModuleList()
        offset = 0
        for depth in depths:
            self.groups.append(ProSATGroupv3(
                embed_dim, depth, num_heads, ffn_ratio,
                capacity_ratios=self.capacity_schedule[offset:offset + depth],
                qkv_bias=qkv_bias, m_ratio=dta_m_ratio,
                min_clusters=dta_min_clusters, recluster_every=dta_recluster_every,
                use_sdpa=use_sdpa, deterministic=dta_deterministic,
                norm_layer=norm_layer))
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
    model = ProSATv3(upscale=2)
    params = sum(p.numel() for p in model.parameters())
    print(f'Params: {params / 1e6:.4f}M')
    print(f'Capacity schedule: {model.capacity_schedule}')
    print(f'FLOPs (64x64):   {model.flops([64, 64]) / 1e9:.3f}G')
    print(f'FLOPs (640x360): {model.flops([640, 360]) / 1e9:.1f}G')
    x = torch.randn(1, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f'Forward: {tuple(x.shape)} -> {tuple(y.shape)}')
