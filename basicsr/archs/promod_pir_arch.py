'''
ProMoD-PIR (PMDPIRModel): PFA-Informed Routing.

Where v1.0/v1.1 bolt a from-scratch router onto PFT, PIR is a redesign of
MoD that is co-designed with PFA (Progressive Focusing Attention, PFT's own
cross-layer sparse-attention cascade, see pft_arch.py). Two documented gaps
in v1.1 (basicsr/archs/promod_v1_1_arch.py) motivate this file, see
ARCH.md / this experiment's plan for the full investigation:

1. v1.1's router (`nn.Linear(dim, 1, bias=False)`) is learned entirely from
   scratch on `norm1(x)` and never reads `pfa_values`/`pfa_indices` — despite
   this project's own README implying it should. PIR's router adds a
   zero-initialized blend with a signal PFA already computes for free (see
   `PMDPIRTL._compute_routing_weights`).
2. v1.1's routed layers never call `attn_win.forward()`, so they never
   update PFA's cascade — every routed layer re-derives importance from a
   stale (or entirely absent) signal. PIR's routed layers keep the cascade
   alive for whichever tokens they do process (see `_refresh_pfa_active`).

**Load-bearing design constraint**: PFA's `pfa_values`/`pfa_indices` are
single tensors covering all `win_n` window positions uniformly — narrowing
the retained key-set (`num_topk` shrinking) for only some rows (e.g. only
the active ones in a routed layer) would leave the tensor with an
inconsistent width across rows. **PIR therefore requires the `num_topk`
schedule to only shrink at `capacity_ratio == 1.0` (dense) layers and stay
flat everywhere else** — enforced by an assertion in `PMDPIRModel.__init__`.
This means, unlike v1.1's progressively-shrinking schedule (1024→256→128→
64→32), PIR's key width shrinks once (at the warmup layers) and then stays
constant for the rest of the network. In exchange, once the width is fixed,
every routed layer can honor it: routed attention narrows keys down to that
fixed width too (v1.1's routed layers always attend over the full dense
`win_n` keys, ignoring pruning entirely) — a real efficiency gain at the
attention matmul that partly offsets losing v1.1's deeper progressive
shrink. Net FLOPs differ from v1.1's schedule and must be measured via
`.flops()`, not assumed.

Third change, independent of the above: **routing decisions (which tokens
are active) are computed once per shift chain per residual group and reused
by every routed layer in that group** — v1.1 recomputes `topk`/sort every
single routed layer, and this per-layer fixed overhead is the documented
cause of MoD's real-world speedup collapsing at inference resolution
(1.84x faster than dense at 64x64 training patches, but 0.82x — i.e.
slower — at 640x360). See `PMDPIRBB.forward`.

Same two ProSAT-informed guardrails as v1.1 (see that file's docstring for
the full incident history): fc1/act/dwconv/v_LePE always stay dense (never
gathered before a spatial conv); only fc2, attention's query side, and now
attention's key side (when pruning is active) are ever gathered.
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
    SMM_QmK, SMM_AmV,
)
from basicsr.archs.promod_arch import build_capacity_schedule
from basicsr.archs.promod_v1_1_arch import RoutedConvFFN


class PMDPIRTL(nn.Module):
    """PFT Transformer Layer with PFA-Informed Routing.

    Dense layers (capacity_ratio == 1.0) are bit-identical to PMDTL/PMDTLv1_1
    — they call attn_win.forward() unchanged, participating fully in PFA's
    cascade. Routed layers (capacity_ratio < 1.0) gather Q to the active
    token set (like v1.1) but, unlike v1.1, also narrow K/V to the incoming
    PFA key-set (when pruning is active) and write a refreshed pfa_values
    estimate back for the active rows, so downstream layers keep receiving
    a live signal instead of a frozen warmup-era snapshot.
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

        if capacity_ratio < 1.0:
            # `probe`: same independent scalar probe v1.1's router already is.
            # `pfa_alpha`: raw scalar, zero-init -- score is byte-identical to
            # v1.1's router at step 0; the network only leans on PFA's free
            # signal if training finds it useful. No sigmoid (no saturation
            # region to get stuck in), same reasoning as CLF's gate.
            self.probe = nn.Linear(dim, 1, bias=False)
            self.pfa_alpha = nn.Parameter(torch.zeros(1))

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.convlepe_kernel_size = convffn_kernel_size
        self.v_LePE = dwconv(hidden_features=dim, kernel_size=self.convlepe_kernel_size)

        # Always instantiated: holds relative_position_bias_table/scale/proj/eps
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

    def _compute_routing_weights(self, x, h, w, b, n, pfa_values, shift, cached_idx=None):
        """Routing math: independent probe + zero-init blend with PFA's own
        convergence signal (1 - peak attention mass -- a token whose
        attention is still diffuse hasn't converged and is a safer "keep
        active" candidate; a token whose attention has already sharply
        peaked has extracted what this cascade can give it).

        `cached_idx`, if given (a (active_idx, mask) pair from an earlier
        layer in the same residual group / shift chain), skips the topk+sort
        call entirely and reuses the same active positions -- amortizing the
        expensive data-dependent op across the group (see PMDPIRBB.forward).
        The gate weight itself (sigmoid(score)) is still recomputed fresh
        every layer -- cheap, and lets PFA's evolving signal actually matter.
        """
        scores = self.probe(x)  # [B, N, 1], x is norm1'd

        scores = scores.view(b, h, w, 1)
        if self.shift_size > 0:
            scores = torch.roll(scores, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        score_windows = window_partition(scores, self.window_size)  # [B*nW, ws, ws, 1]
        win_n = self.window_size * self.window_size
        score_windows = score_windows.view(-1, win_n)  # [B*nW, win_n]

        if pfa_values[shift] is not None:
            # pfa_values[shift]: [b_, heads, win_n, key_width] -- same b_/win_n
            # layout score_windows is already in (same shift, same window
            # convention), no extra reshape needed.
            concentration = pfa_values[shift].mean(dim=1).amax(dim=-1)  # [b_, win_n]
            pfa_signal = 1.0 - concentration
            score_windows = score_windows + self.pfa_alpha * pfa_signal

        k = max(1, int(math.ceil(self.capacity_ratio * win_n)))

        if cached_idx is not None:
            active_idx, mask = cached_idx
        else:
            _, active_idx = torch.topk(score_windows, k, dim=-1)  # [B*nW, k], window-local coords
            active_idx, _ = torch.sort(active_idx, dim=-1)  # restore left-to-right order within window
            mask = torch.zeros_like(score_windows)
            mask.scatter_(1, active_idx, 1.0)

        weights = mask * torch.sigmoid(score_windows)  # [B*nW, win_n]

        weights_full = weights.view(-1, self.window_size, self.window_size, 1)
        weights_full = window_reverse(weights_full, self.window_size, h, w)  # [B, H, W, 1]
        if self.shift_size > 0:
            weights_full = torch.roll(weights_full, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        weights_full = weights_full.reshape(b, n, 1)
        return weights_full, active_idx, k, (active_idx, mask)

    def forward(self, x, pfa_list, x_size, params, cached_idx=None):
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
            # --- Dense path: bit-identical to PMDTL/PMDTLv1_1, PFA chain intact ---
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
            return x, pfa_list, None

        # --- Routed path: real gather/scatter, PFA cascade kept alive ---
        weights, active_idx, k, idx_cache = self._compute_routing_weights(
            x, h, w, b, n, pfa_values, shift, cached_idx=cached_idx)

        num_heads = self.num_heads
        head_dim = c // num_heads
        b_ = x_windows.shape[0]

        qkvp = x_windows.reshape(b_, win_n, 4, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, kk, vv, v_lepe_w = qkvp[0], qkvp[1], qkvp[2], qkvp[3]  # each [b_, heads, win_n, head_dim]
        q = q * self.attn_win.scale

        idx_h = active_idx.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, head_dim)  # [b_,heads,k,hd]
        q_active = torch.gather(q, 2, idx_h)                # [b_, heads, k, head_dim] — real savings
        v_lepe_active = torch.gather(v_lepe_w, 2, idx_h)     # [b_, heads, k, head_dim]

        rpi = params['rpi_sa']
        rpb_full = self.attn_win.relative_position_bias_table[rpi.view(-1)].view(win_n, win_n, -1)
        rpb_full = rpb_full.permute(2, 0, 1).contiguous()              # [heads_or_1, win_n, win_n]
        rpb_full = rpb_full.unsqueeze(0).expand(b_, num_heads, -1, -1)  # [b_, heads, win_n, win_n]

        prev_indices = pfa_indices[shift]
        key_width = win_n if prev_indices is None else prev_indices.shape[-1]

        if prev_indices is None:
            # --- Case A: no key-pruning has happened yet for this shift chain
            # (PIR's num_topk schedule keeps this layer at the full window
            # width too -- see module docstring's constraint). Dense keys,
            # same computation v1.1's routed branch always does. ---
            attn_active = q_active @ kk.transpose(-2, -1)  # [b_, heads, k, win_n]

            idx_rpb = active_idx.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, win_n)
            rpb_active = torch.gather(rpb_full, 2, idx_rpb)  # [b_, heads, k, win_n]
            attn_active = attn_active + rpb_active

            if shift:
                mask_t = params['attn_mask']  # [nW, win_n, win_n]
                nw = mask_t.shape[0]
                mask_exp = mask_t.unsqueeze(0).expand(b_ // nw, -1, -1, -1).reshape(b_, win_n, win_n)
                idx_mask = active_idx.unsqueeze(-1).expand(-1, -1, win_n)
                mask_active = torch.gather(mask_exp, 1, idx_mask)
                attn_active = attn_active + mask_active.unsqueeze(1)

            attn_active = torch.softmax(attn_active, dim=-1)
            out = torch.einsum('bhij,bhijd->bhid', attn_active, vv) + v_lepe_active  # [b_,heads,k,hd]
        else:
            # --- Case B: keys already narrowed to `key_width` by an earlier
            # dense layer. Uses the SAME fused sparse-matmul kernels stock
            # PFT's own sparse-attention branch uses (pft_arch.py's
            # WindowAttention.forward, pfa_indices-not-None case) instead of
            # a plain gather+einsum: gathering K/V into a real
            # [b_,heads,k,key_width,head_dim] tensor materializes ~1.56GiB
            # per layer at real training batch size (measured OOM across the
            # 22 routed layers) -- SMM_QmK/SMM_AmV compute the per-row
            # sparse dot products without ever materializing that tensor. ---
            idx_row_k = active_idx.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, key_width)
            prev_idx_active = torch.gather(prev_indices, 2, idx_row_k)  # [b_,heads,k,key_width] -- index-only, small

            q_flat = q_active.contiguous().view(b_ * num_heads, k, head_dim)
            k_flat = kk.contiguous().view(b_ * num_heads, win_n, head_dim).transpose(-2, -1)  # full K, no gather
            idx_flat = prev_idx_active.contiguous().view(b_ * num_heads, k, key_width).int()
            attn_active = SMM_QmK.apply(q_flat, k_flat, idx_flat).view(b_, num_heads, k, key_width)

            rpb_active_rows = torch.gather(
                rpb_full, 2, active_idx.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, win_n))
            rpb_narrow = torch.gather(rpb_active_rows, -1, prev_idx_active)  # [b_,heads,k,key_width]
            attn_active = attn_active + rpb_narrow
            # No shift-mask re-application here -- matches stock PFT's own
            # sparse-attention branch (pft_arch.py lines 270-284), which
            # likewise skips it: a masked-out position would have scored a
            # large negative value at the dense layer that established this
            # key set, so it's already unlikely to have survived into
            # `prev_idx_active` in the first place.

            attn_active = torch.softmax(attn_active, dim=-1)

            v_flat = vv.contiguous().view(b_ * num_heads, win_n, head_dim)  # full V, no gather
            attn_flat = attn_active.contiguous().view(b_ * num_heads, k, key_width)
            out = SMM_AmV.apply(attn_flat, v_flat, idx_flat).view(b_, num_heads, k, head_dim)
            out = out + v_lepe_active
        out = out.transpose(1, 2).reshape(b_, k, c)
        out = self.attn_win.proj(out)  # [b_, k, c] — real savings

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

        num_windows = (h // self.window_size) * (w // self.window_size)
        k_total = k * num_windows
        _, active_idx_full = torch.topk(weights.reshape(b, n), k_total, dim=-1)
        active_idx_full, _ = torch.sort(active_idx_full, dim=-1)

        x_ffn = self.convffn(self.norm2(x), x_size, active_idx=active_idx_full) * weights
        x = x + x_ffn

        if cached_idx is None:
            # Cascade refresh only on the first routed layer of this shift
            # chain per group (same granularity as the routing decision
            # itself, cached via idx_cache -- see PMDPIRBB.forward). Doing
            # this every routed layer instead of once per group cloned a
            # full [b_,heads,win_n,key_width] tensor 22 times per forward
            # pass (~256MB each at real training batch size) and OOM'd at
            # batch_size_per_gpu=16 on an A100 -- all 22 had to stay alive
            # simultaneously for backward. Once per group cuts that to ~8.
            pfa_values[shift] = self._refresh_pfa_active(
                pfa_values, shift, attn_active, active_idx, num_heads, key_width, b_)

        pfa_list = [pfa_values, pfa_indices]  # indices unchanged — routing never shrinks them
        return x, pfa_list, idx_cache

    def _refresh_pfa_active(self, pfa_values, shift, attn_active, active_idx, num_heads, key_width, b_):
        """Combine this layer's freshly-computed attention (over whichever
        key set was actually attended to) with the incoming cascade value at
        the active rows, then scatter into a cloned pfa_values buffer.
        Inactive rows keep their exact prior value (those tokens' features
        didn't change either — they resolve to `shortcut` via the residual).
        """
        idx_scatter_val = active_idx.unsqueeze(1).unsqueeze(-1).expand(-1, num_heads, -1, key_width)

        if pfa_values[shift] is None:
            # First-ever value for this shift chain (shouldn't normally happen --
            # mod_warmup_layers guarantees a dense pass first, which always sets
            # pfa_values -- but handled for robustness/ablation configs with
            # warmup disabled). No prior value to combine with; inactive rows
            # start at exactly 0 (harmless -- the very next dense or routed
            # layer that reads them via `1 - concentration` in the router will
            # see a max-uncertainty signal, not a stale/misleading one).
            new_values = attn_active.new_zeros(b_, num_heads, win_n, key_width)
            new_values.scatter_(2, idx_scatter_val, attn_active)
            return new_values

        prev_val_active = torch.gather(pfa_values[shift], 2, idx_scatter_val)  # [b_,heads,k,key_width]
        combined = attn_active * prev_val_active
        eps = self.attn_win.eps
        combined = (combined + eps) / (combined.sum(-1, keepdim=True) + eps)

        new_values = pfa_values[shift].clone()
        new_values.scatter_(2, idx_scatter_val, combined)
        return new_values

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.input_resolution if input_resolution is None else input_resolution
        r = self.capacity_ratio
        win_n = self.window_size * self.window_size
        nw = h * w / win_n

        flops += self.dim * 3 * self.dim * h * w  # wqkv, dense

        if r >= 1.0:
            flops += nw * self.attn_win.flops(win_n)
        else:
            k = max(1, int(math.ceil(r * win_n)))
            key_width = self.attn_win.topk  # flat width every routed layer inherits/keeps (see docstring)
            head_dim = self.dim // self.num_heads
            flops += nw * self.num_heads * k * head_dim * key_width * 2  # QK^T + attn@V
            flops += nw * k * self.dim * self.dim  # proj, gathered rows only

        flops += h * w * self.dim * self.dim * self.mlp_ratio * (1.0 if r >= 1.0 else (1 + r))
        flops += h * w * self.dim * (self.convffn_kernel_size ** 2) * self.mlp_ratio  # dwconv, dense
        flops += h * w * self.dim * (self.convlepe_kernel_size ** 2)  # LePE, dense

        return flops


class PMDPIRBB(nn.Module):
    """Container of ProMoD-PIR transformer layers (PMDBB equivalent).

    Caches the (active_idx, mask) pair per shift (0/1) the first time a
    routed layer of that shift computes it, reusing the cache for every
    subsequent same-shift routed layer in this group — amortizing the
    expensive topk+sort call across the group instead of paying it every
    layer (the documented cause of MoD's real speedup collapsing at
    inference resolution). Cache is local to one forward() call (a plain
    dict, not a persistent buffer) and keyed by shift since shifted/
    unshifted windows are different partitions of the image and their
    window-local indices are not interchangeable.
    """

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
                PMDPIRTL(
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
        idx_cache = {0: None, 1: None}
        for layer in self.layers:
            shift = 1 if layer.shift_size > 0 else 0
            x, pfa_list, returned_cache = layer(x, pfa_list, x_size, params, cached_idx=idx_cache[shift])
            if returned_cache is not None:
                idx_cache[shift] = returned_cache
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


class PMDPIRB(nn.Module):
    """ProMoD-PIR group block (PMDB equivalent)."""

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

        self.residual_group = PMDPIRBB(
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
class PMDPIRModel(nn.Module):
    """ProMoD-PIR: PFT + PFA-Informed Routing. See module docstring at the
    top of this file for what changed vs v1.1 (promod_v1_1_arch.py) and why.
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

        # Load-bearing invariant (see module docstring): num_topk may only
        # shrink at capacity_ratio == 1.0 (dense) layers. Fail fast at
        # construction time instead of a silent shape-mismatch deep in a
        # training run.
        for i in range(1, total_layers):
            if num_topk[i] < num_topk[i - 1] and self.capacity_schedule[i] < 1.0:
                raise ValueError(
                    f'num_topk shrinks at layer {i} (from {num_topk[i-1]} to {num_topk[i]}) '
                    f'while capacity_schedule[{i}]={self.capacity_schedule[i]} < 1.0 (routed). '
                    'PIR requires num_topk to only shrink at dense (capacity_ratio==1.0) layers '
                    'and stay flat at every routed layer — see promod_pir_arch.py module docstring.')

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
            layer = PMDPIRB(
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
        """min/mean/max of pfa_alpha per routed layer, for smoke-test telemetry."""
        stats = {}
        for group in self.layers:
            for layer in group.residual_group.layers:
                if hasattr(layer, 'pfa_alpha'):
                    a = layer.pfa_alpha.detach()
                    stats[layer.layer_id] = (a.min().item(), a.mean().item(), a.max().item())
        return stats

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
    model = PMDPIRModel(
        upscale=2,
        img_size=64,
        embed_dim=52,
        depths=[2, 4, 6, 6, 6],
        num_heads=4,
        num_topk=[256] * 24,
        window_size=32,
        convffn_kernel_size=7,
        img_range=1.,
        mlp_ratio=1,
        upsampler='pixelshuffledirect',
        mod_warmup_layers=2,
        mod_capacity=0.48,
    )

    total = sum([param.nelement() for param in model.parameters()])
    print(f"Number of parameters: {total / 1e6:.3f}M")
    print(f"Capacity schedule: {model.capacity_schedule}")
    print(f"FLOPs (640x360): {model.flops([640, 360]) / 1e9:.2f}G")
    print(f"FLOPs (320x180): {model.flops([320, 180]) / 1e9:.2f}G")

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Gate stats (should all be exactly (0.0, 0.0, 0.0) at init): {model.get_gate_stats()}")
