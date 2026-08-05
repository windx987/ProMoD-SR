"""
Four-axis benchmark for the whole ProMoD-SR campaign: params, FLOPs, inference
latency, and peak GPU memory -- every architecture built from the SAME YAML it
was actually trained with, via basicsr's build_network.

Why all four axes: this project reports FLOPs reductions (503 -28.68%,
502 -20.18%, 403 -55% @640x360), but our own runs have contradicted the
FLOPs->speed assumption three times in three directions -- 501's real FLOPs cut
was *slower* at >=256x256, 503's -28.68% gave a 3.7x training speedup, and
403's -25% gave no speedup at all. RIB/SST (arXiv 2603.06738) makes the same
point from the opposite side: it spends 5.7x MORE FLOPs than PFT while being
~3x faster and using ~10x less memory, and states outright that "FLOPs
reductions do not necessarily reflect practical efficiency gains."

So FLOPs alone is not a result. Report it next to what it fails to predict.

Trained weights are NOT required here -- params/FLOPs/latency/memory are all
properties of the architecture. That matters because 301/304/401/501's
experiment directories no longer exist on either PVC; their configs do.

Usage:
    python benchmark.py                     # everything available on this box
    python benchmark.py --no-gpu            # Stage A: params + FLOPs only
    python benchmark.py --json out.json     # also write machine-readable output
    python benchmark.py --only 402,403      # subset

NOTE: run the GPU half on an IDLE node only. A node that is training has both
GPUs busy under DDP, and latency measured under contention is meaningless.
The reported std/mean is the check -- see --max-rsd.
"""

import argparse
import json
import math
import time
from copy import deepcopy

import torch
import yaml

from basicsr.archs import build_network

# --------------------------------------------------------------------------- #
# Every run in the campaign, pointing at the config it actually trained with.
# Adding a future experiment = adding one line here.
# --------------------------------------------------------------------------- #
RUNS = [
    ('304', 'options/train/304_PFTlight_muon_dense_SRx2.yml',             'PFT-light dense (baseline)'),
    ('321', 'options/train/321_ProMoD_light_SRx2_r0500.yml',              'Mask-multiply MoD r=0.5'),
    ('501', 'options/train/501_ProMoDv1_1_light_SRx2_scratch.yml',        'GS gather/scatter r=0.5 +warmup'),
    ('502', 'options/train/502_ProMoDv1_1_light_SRx2_r0480.yml',          'GS r=0.48'),
    ('503', 'options/train/503_ProMoDv1_1_light_SRx2_r0500_nowarmup.yml', 'GS r=0.5 no warmup'),
    ('504', 'options/train/504_ProMoD_MoE_light_SRx2_e2.yml',             'MoE e=2'),
    ('505', 'options/train/505_ProMoD_MoE_light_SRx2_e4.yml',             'MoE e=4'),
    ('601', 'options/train/601_ProMoD_CLF_light_SRx2_scratch.yml',        'CLF cross-layer fusion'),
    ('401', 'options/train/401_ProSAT_light_SRx2_scratch.yml',            'ProSAT (DTA global attn)'),
    ('402', 'options/train/402_ProSATv2_light_SRx2_scratch.yml',          'ProSAT-GR (gated router)'),
    ('403', 'options/train/403_ProSATv3_light_SRx2_scratch.yml',          'ProSAT-ADT (amortized DTA)'),
]

RESOLUTIONS = [
    (64,  64,  '64x64'),
    (256, 256, '256x256'),
    (640, 360, '640x360'),
]

WARMUP_RUNS = 5
TIMED_RUNS = 20


# --------------------------------------------------------------------------- #
# Measurement primitives (unchanged from the original benchmark.py)
# --------------------------------------------------------------------------- #
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_flops(model, h, w):
    """Use the model's own flops() -- each arch defines its honest accounting."""
    try:
        return model.flops([h, w])
    except Exception:
        return None


def measure_latency(model, h, w, device, warmup=WARMUP_RUNS, runs=TIMED_RUNS):
    """Mean, std (ms) over `runs` timed forward passes. None on OOM."""
    try:
        x = torch.randn(1, 3, h, w, device=device)
        model.eval()
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times = []
            for _ in range(runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
        mean = sum(times) / len(times)
        std = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))
        return mean, std
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def measure_memory(model, h, w, device):
    """Peak GPU memory (MB) for one forward. None on OOM, 0 on CPU."""
    if device.type != 'cuda':
        return 0.0
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(1, 3, h, w, device=device)
        model.eval()
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


# --------------------------------------------------------------------------- #
def build_from_config(cfg_path, device):
    """Build exactly the network the run trained with."""
    with open(cfg_path) as f:
        opt = yaml.safe_load(f)
    net = build_network(deepcopy(opt['network_g']))
    return net.to(device).eval()


def fmt(v, width, prec=2, scale=1.0, na='-'):
    return f'{na:>{width}}' if v is None else f'{v / scale:>{width}.{prec}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-gpu', action='store_true', help='params + FLOPs only')
    ap.add_argument('--json', metavar='PATH', help='write machine-readable results')
    ap.add_argument('--only', help='comma-separated run ids, e.g. 402,403')
    ap.add_argument('--max-rsd', type=float, default=5.0,
                    help='flag latency whose std/mean%% exceeds this (default 5)')
    args = ap.parse_args()

    device = torch.device('cpu' if args.no_gpu or not torch.cuda.is_available() else 'cuda')
    do_gpu = device.type == 'cuda'
    runs = RUNS
    if args.only:
        keep = {s.strip() for s in args.only.split(',')}
        runs = [r for r in RUNS if r[0] in keep]

    print(f'\nDevice: {device}' + (f'  ({torch.cuda.get_device_name(0)})' if do_gpu else ''))
    if not do_gpu:
        print('GPU measurements skipped -- params and FLOPs only (Stage A).')
    else:
        print('NOTE: latency/memory are only meaningful on an IDLE node. Check the RSD column.')

    results = {}
    for rid, cfg, desc in runs:
        try:
            model = build_from_config(cfg, device)
        except Exception as e:
            print(f'  [{rid}] BUILD FAILED: {type(e).__name__}: {e}')
            continue
        rec = {'desc': desc, 'config': cfg, 'params': count_params(model), 'res': {}}
        for h, w, label in RESOLUTIONS:
            entry = {'flops': measure_flops(model, h, w)}
            if do_gpu:
                mean, std = measure_latency(model, h, w, device)
                entry['latency_ms'] = mean
                entry['latency_std'] = std
                entry['peak_mem_mb'] = measure_memory(model, h, w, device)
            rec['res'][label] = entry
        results[rid] = rec
        del model
        if do_gpu:
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- tables
    print('\n' + '=' * 100)
    print('PARAMS + FLOPs   (FLOPs from each arch\'s own flops(); "-" = not implemented / OOM)')
    print('=' * 100)
    hdr = f'{"run":<5} {"architecture":<36} {"params":>10}'
    for _, _, label in RESOLUTIONS:
        hdr += f' {label + " (G)":>16}'
    print(hdr)
    print('-' * 100)
    for rid, rec in results.items():
        row = f'{rid:<5} {rec["desc"]:<36} {rec["params"] / 1e6:>9.4f}M'
        for _, _, label in RESOLUTIONS:
            row += ' ' + fmt(rec['res'][label]['flops'], 16, 2, 1e9)
        print(row)

    if do_gpu:
        print('\n' + '=' * 100)
        print(f'INFERENCE LATENCY (ms, batch=1, mean over {TIMED_RUNS} runs) '
              f'| RSD = std/mean%, >{args.max_rsd}% means the node was NOT idle')
        print('=' * 100)
        hdr = f'{"run":<5} {"architecture":<36}'
        for _, _, label in RESOLUTIONS:
            hdr += f' {label:>18}'
        print(hdr)
        print('-' * 100)
        noisy = []
        for rid, rec in results.items():
            row = f'{rid:<5} {rec["desc"]:<36}'
            for _, _, label in RESOLUTIONS:
                e = rec['res'][label]
                if e.get('latency_ms') is None:
                    row += f'{"OOM":>19}'
                else:
                    rsd = 100 * e['latency_std'] / e['latency_ms']
                    if rsd > args.max_rsd:
                        noisy.append((rid, label, rsd))
                    row += f'{e["latency_ms"]:>13.1f}±{rsd:>4.1f}%'
            print(row)
        if noisy:
            print(f'\n  !! {len(noisy)} measurement(s) exceeded {args.max_rsd}% RSD -- '
                  f'node was likely busy; re-run on an idle node before reporting:')
            for rid, label, rsd in noisy[:8]:
                print(f'     {rid} @ {label}: {rsd:.1f}%')

        print('\n' + '=' * 100)
        print('PEAK GPU MEMORY (MB, batch=1, inference)')
        print('=' * 100)
        hdr = f'{"run":<5} {"architecture":<36}'
        for _, _, label in RESOLUTIONS:
            hdr += f' {label:>16}'
        print(hdr)
        print('-' * 100)
        for rid, rec in results.items():
            row = f'{rid:<5} {rec["desc"]:<36}'
            for _, _, label in RESOLUTIONS:
                m = rec['res'][label].get('peak_mem_mb')
                row += f'{"OOM":>17}' if m is None else f'{m:>17.1f}'
            print(row)

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'device': str(device), 'gpu': torch.cuda.get_device_name(0) if do_gpu else None,
                       'results': results}, f, indent=1)
        print(f'\nWrote {args.json}')

    print('\nReminder: FLOPs is not a performance claim on its own. Pair every FLOPs')
    print('number with the latency/memory column before reporting it.\n')


if __name__ == '__main__':
    main()
