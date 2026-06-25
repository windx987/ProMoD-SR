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
from basicsr.archs.promod_arch import ProMoD

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
    pft    = PFT(**SHARED_CFG).to(DEVICE)
    promod = ProMoD(**SHARED_CFG, mod_warmup_layers=2).to(DEVICE)

    pft_params    = count_params(pft)
    promod_params = count_params(promod)

    # ------------------------------------------------------------------ #
    # 1. Parameters
    # ------------------------------------------------------------------ #
    hr()
    print("PARAMETERS")
    hr()
    print(f"  PFT-light    : {pft_params:>10,}  ({pft_params/1e6:.3f}M)")
    print(f"  ProMoD-light : {promod_params:>10,}  ({promod_params/1e6:.3f}M)")
    delta = promod_params - pft_params
    print(f"  Delta        : {delta:>+10,}  ({'same' if delta == 0 else f'{delta:+,}'})")

    # ------------------------------------------------------------------ #
    # 2. FLOPs
    # ------------------------------------------------------------------ #
    hr()
    print("FLOPs  (model.flops() — theoretical, does not reflect mask-multiply)")
    hr()
    print(f"  {'Resolution':<22}  {'PFT (G)':>10}  {'ProMoD (G)':>12}  {'Ratio':>8}")
    hr('·')
    for h, w, label in RESOLUTIONS:
        pft_f    = measure_flops(pft,    h, w)
        promod_f = measure_flops(promod, h, w)
        if pft_f is not None and promod_f is not None:
            ratio = promod_f / pft_f
            print(f"  {label:<22}  {pft_f/1e9:>10.2f}  {promod_f/1e9:>12.2f}  {ratio:>8.3f}×")
        else:
            print(f"  {label:<22}  {'N/A':>10}  {'N/A':>12}  {'N/A':>8}")

    # ------------------------------------------------------------------ #
    # 3. Inference latency
    # ------------------------------------------------------------------ #
    hr()
    print(f"INFERENCE LATENCY  (ms, mean ± std over {TIMED_RUNS} runs, batch=1)")
    hr()
    print(f"  {'Resolution':<22}  {'PFT (ms)':>14}  {'ProMoD (ms)':>14}  {'Speedup':>9}")
    hr('·')
    for h, w, label in RESOLUTIONS:
        try:
            pft_mean,    pft_std    = measure_latency(pft,    h, w)
            promod_mean, promod_std = measure_latency(promod, h, w)
            speedup = pft_mean / promod_mean
            print(f"  {label:<22}  {pft_mean:>7.1f}±{pft_std:>4.1f}  "
                  f"{promod_mean:>7.1f}±{promod_std:>4.1f}  {speedup:>8.3f}×")
        except Exception as e:
            print(f"  {label:<22}  ERROR: {e}")

    # ------------------------------------------------------------------ #
    # 4. GPU memory
    # ------------------------------------------------------------------ #
    if DEVICE.type == 'cuda':
        hr()
        print("PEAK GPU MEMORY  (MB, inference, batch=1)")
        hr()
        print(f"  {'Resolution':<22}  {'PFT (MB)':>10}  {'ProMoD (MB)':>12}  {'Ratio':>8}")
        hr('·')
        for h, w, label in RESOLUTIONS:
            try:
                pft_mem    = measure_memory(pft,    h, w)
                promod_mem = measure_memory(promod, h, w)
                ratio = promod_mem / pft_mem if pft_mem > 0 else float('nan')
                print(f"  {label:<22}  {pft_mem:>10.1f}  {promod_mem:>12.1f}  {ratio:>8.3f}×")
            except Exception as e:
                print(f"  {label:<22}  ERROR: {e}")

    hr()
    print("NOTE: FLOPs are from model.flops() which accounts for capacity_ratio r.")
    print("      Latency and memory reflect actual execution (mask-multiply = same as PFT).")
    print("      True speedup requires sparse gather/scatter implementation.")
    hr()


if __name__ == '__main__':
    main()
