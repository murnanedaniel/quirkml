"""mu 60 size check: forward+loss+backward time and peak GPU memory of the current MaskFormer (pu0 pair model,
10-layer full-attention encoder, 4 decoder layers, 128 queries) on one synthetic event of N hits (bf16 autocast).
Hits: random positions; 40 targets with random masks of ~13 hits. Usage: bench_size.py CKPT N1 N2 ..."""
import sys
import time

import torch

from ml_tracking.experiments.quirk.model import QuirkModel

ckpt = sys.argv[1]
module = QuirkModel.load_from_checkpoint(ckpt, map_location="cuda")
net = module.model.train()
opt = torch.optim.AdamW(net.parameters(), lr=1e-6)
P = 128
for n in map(int, sys.argv[2:]):
    g = torch.Generator(device="cuda").manual_seed(0)
    x, y, z = (torch.rand(1, n, device="cuda", generator=g) * 2 - 1 for _ in range(3))
    r = torch.sqrt(x**2 + y**2); s = torch.sqrt(r**2 + z**2)
    inputs = {"sihit_x": x, "sihit_y": y, "sihit_z": z, "sihit_r": r, "sihit_s": s, "sihit_eta": torch.arctanh(z / s * 0.99),
              "sihit_phi": torch.atan2(y, x), "sihit_layer_id": torch.randint(2, 16, (1, n), device="cuda").float(),
              "sihit_valid": torch.ones(1, n, dtype=torch.bool, device="cuda")}
    valid = torch.zeros(1, P, dtype=torch.bool, device="cuda"); valid[:, :40] = True
    masks = torch.zeros(1, P, n, dtype=torch.bool, device="cuda")
    for i in range(40):
        masks[0, i, torch.randint(0, n, (13,), device="cuda")] = True
    targets = {"particle_valid": valid, "particle_sihit_valid": masks, "sihit_valid": inputs["sihit_valid"],
               "particle_is_quirk": torch.zeros(1, P, dtype=torch.bool, device="cuda"), "event_id": torch.zeros(1, device="cuda")}
    try:
        for it in range(3):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(inputs)
                _, _, losses = net.loss(out, targets)
                loss = sum(v for layer in losses.values() for task in layer.values() for v in task.values())
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); dt = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 2**30
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.cuda.synchronize(); t0 = time.time(); net(inputs); torch.cuda.synchronize(); ti = time.time() - t0
        print(f"N={n}: train step {dt:.3f} s, peak {mem:.1f} GB; inference {ti:.3f} s", flush=True)
    except torch.OutOfMemoryError:
        print(f"N={n}: OOM", flush=True); torch.cuda.empty_cache()
