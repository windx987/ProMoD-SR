"""
Benchmark: PFT-light vs ProMoD-light
Compares parameters, FLOPs, inference speed, and GPU memory.
Both models use the same config (embed_dim=52, depths=[2,4,6,6,6]).
"""

import torch
import time
import math

# Must import arch modules via basicsr to trigger registry
from basicsr.archs.pft_arch import PFT
from basicsr.archs.promod_arch import PMDModel
from basicsr.archs.promod_v1_1_arch import PMDGSModel
from basicsr.archs.promod_moe_arch import PMDMoEModel

# --------------------------------------------------------------------------- #
# Shared config (matches 101/201 light configs)
# --------------------------------------------------------------------------- #
SHARED_CFG = dict(
    upscale=2,
    in_chans=3,
    img_size=64,
    embed_dim=52,
    depths=[2, 4, 6, 6, 6],
    num_heads=4,
    num_topk=[
        1024, 1024,
        256, 256, 256, 256,
        128, 128, 128, 128, 128, 128,
        64, 64, 64, 64, 64, 64,
        32, 32, 32, 32, 32, 32,
    ],
    window_size=32,
    convffn_kernel_size=7,
    img_range=1.0,
    mlp_ratio=1,
    upsampler='pixelshuffledirect',
    resi_connection='1conv',
    use_checkpoint=False,
)

RESOLUTIONS = [
    (64,  64,  "64×64   (train patch)"),
    (128, 128, "128×128"),
    (256, 256, "256×256"),
    (320, 180, "320×180 (360p)"),
    (640, 360, "640×360 (720p half)"),
]

WARMUP_RUNS = 5
TIMED_RUNS  = 20
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_flops(model, h, w):
    """Use model's built-in flops() method."""
    try:
        return model.flops([h, w])
    except Exception:
        return None


def measure_latency(model, h, w, warmup=WARMUP_RUNS, runs=TIMED_RUNS):
    """Mean ± std inference latency in ms over `runs` timed forward passes."""
    x = torch.randn(1, 3, h, w, device=DEVICE)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(runs):
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    mean = sum(times) / len(times)
    std  = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))
    return mean, std


def measure_memory(model, h, w):
    """Peak GPU memory (MB) for one forward pass. CPU returns 0."""
    if DEVICE.type != 'cuda':
        return 0.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(1, 3, h, w, device=DEVICE)
    model.eval()
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def hr(char='─', width=88):
    print(char * width)


def main():
    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # Build models
    pft     = PFT(**SHARED_CFG).to(DEVICE)
    promod  = PMDModel(**SHARED_CFG, mod_warmup_layers=2).to(DEVICE)
    promodv11 = PMDGSModel(**SHARED_CFG, mod_warmup_layers=2).to(DEVICE)
    promodmoe = PMDMoEModel(**SHARED_CFG, mod_warmup_layers=2, num_experts=2).to(DEVICE)

    pft_params      = count_params(pft)
    promod_params   = count_params(promod)
    promodv11_params = count_params(promodv11)
    promodmoe_params = count_params(promodmoe)

    # ------------------------------------------------------------------ #
    # 1. Parameters
    # ------------------------------------------------------------------ #
    hr()
    print("PARAMETERS")
    hr()
    print(f"  PFT-light      : {pft_params:>10,}  ({pft_params/1e6:.3f}M)")
    print(f"  ProMoD-light   : {promod_params:>10,}  ({promod_params/1e6:.3f}M)")
    print(f"  ProMoDv1.1     : {promodv11_params:>10,}  ({promodv11_params/1e6:.3f}M)")
    print(f"  ProMoD-MoE(e2) : {promodmoe_params:>10,}  ({promodmoe_params/1e6:.3f}M)")

    # ------------------------------------------------------------------ #
    # 2. FLOPs
    # ------------------------------------------------------------------ #
    hr()
    print("FLOPs  (model.flops() — v1.0 is theoretical/mask-multiply, does not")
    print("        reflect actual execution; v1.1's is honest — see PMDTLv1_1.flops;")
    print("        MoE's is honest too but goes UP vs dense, not down — see PMDTLMoE.flops)")
    hr()
    print(f"  {'Resolution':<22}  {'PFT (G)':>10}  {'ProMoD (G)':>12}  {'v1.1 (G)':>10}  {'MoE(e2) (G)':>12}")
    hr('·')
    for h, w, label in RESOLUTIONS:
        pft_f      = measure_flops(pft, h, w)
        promod_f   = measure_flops(promod, h, w)
        v11_f      = measure_flops(promodv11, h, w)
        moe_f      = measure_flops(promodmoe, h, w)
        vals = [f'{v/1e9:>10.2f}' if v is not None else f'{"N/A":>10}' for v in (pft_f,)]
        vals2 = [f'{v/1e9:>12.2f}' if v is not None else f'{"N/A":>12}' for v in (promod_f,)]
        vals3 = [f'{v/1e9:>10.2f}' if v is not None else f'{"N/A":>10}' for v in (v11_f,)]
        vals4 = [f'{v/1e9:>12.2f}' if v is not None else f'{"N/A":>12}' for v in (moe_f,)]
        print(f"  {label:<22}  {vals[0]}  {vals2[0]}  {vals3[0]}  {vals4[0]}")

    # ------------------------------------------------------------------ #
    # 3. Inference latency
    # ------------------------------------------------------------------ #
    hr()
    print(f"INFERENCE LATENCY  (ms, mean ± std over {TIMED_RUNS} runs, batch=1)")
    hr()
    print(f"  {'Resolution':<22}  {'PFT (ms)':>14}  {'ProMoD (ms)':>14}  {'v1.1 (ms)':>14}  {'MoE(e2) (ms)':>14}")
    hr('·')
    for h, w, label in RESOLUTIONS:
        try:
            pft_mean, pft_std = measure_latency(pft, h, w)
            promod_mean, promod_std = measure_latency(promod, h, w)
            v11_mean, v11_std = measure_latency(promodv11, h, w)
            moe_mean, moe_std = measure_latency(promodmoe, h, w)
            print(f"  {label:<22}  {pft_mean:>7.1f}±{pft_std:>4.1f}  "
                  f"{promod_mean:>7.1f}±{promod_std:>4.1f}  "
                  f"{v11_mean:>7.1f}±{v11_std:>4.1f}  {moe_mean:>7.1f}±{moe_std:>4.1f}")
        except Exception as e:
            print(f"  {label:<22}  ERROR: {e}")

    # ------------------------------------------------------------------ #
    # 4. GPU memory
    # ------------------------------------------------------------------ #
    if DEVICE.type == 'cuda':
        hr()
        print("PEAK GPU MEMORY  (MB, inference, batch=1)")
        hr()
        print(f"  {'Resolution':<22}  {'PFT (MB)':>10}  {'ProMoD (MB)':>12}  {'v1.1 (MB)':>10}  {'MoE(e2) (MB)':>12}")
        hr('·')
        for h, w, label in RESOLUTIONS:
            try:
                pft_mem = measure_memory(pft, h, w)
                promod_mem = measure_memory(promod, h, w)
                v11_mem = measure_memory(promodv11, h, w)
                moe_mem = measure_memory(promodmoe, h, w)
                print(f"  {label:<22}  {pft_mem:>10.1f}  {promod_mem:>12.1f}  {v11_mem:>10.1f}  {moe_mem:>12.1f}")
            except Exception as e:
                print(f"  {label:<22}  ERROR: {e}")

    hr()
    print("NOTE: PFT/ProMoD FLOPs are from model.flops(); ProMoD's is theoretical and")
    print("      does not reflect actual execution (mask-multiply = same latency as PFT).")
    print("      ProMoDv1.1 (PMDGSModel) implements real gather/scatter execution —")
    print("      its FLOPs() is corrected accordingly and its latency column above")
    print("      should show a real speedup over PFT/ProMoD, not just a theoretical one.")
    print("      ProMoD-MoE (PMDMoEModel) adds a soft, fully-dense multi-expert FFN —")
    print("      this is a quality/capacity trade, NOT a FLOPs reduction: its FLOPs and")
    print("      latency are expected to be HIGHER than ProMoD, not lower.")
    hr()


if __name__ == '__main__':
    main()
