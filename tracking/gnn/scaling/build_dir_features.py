"""tracking/direction/build_direction_features.py with the scaling-study paths (redigi input and feature output under
/pscratch/sd/d/danieltm/quirk_tracking/scaling/direction). Same code, same production-identity check.
  python build_dir_features.py <block> ...   (block b = events 1000b..1000b+999). Deletes the block's
  measurements.root files after a successful write (they are 330 MB per run)."""
import os, sys
from pathlib import Path
sys.path.insert(0, "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/direction")
import build_direction_features as B  # noqa: E402  (loads frames.npy from the Stage-A dir)

S = Path("/pscratch/sd/d/danieltm/quirk_tracking/scaling/direction")
B.D = S
B.OUT = S / "features"
for b in map(int, sys.argv[1:]):
    if (B.OUT / f"quirks.m200_L774_mu0.v1.direction.events{1000 * b}-{1000 * b + 999}.parquet").exists():
        print(f"block {b} exists"); continue
    b, nh, ns = B.build_block(b)
    print(f"block {b}: {nh} hits, identical to production (loc0, volume, sizes): {ns} ({ns / nh:.6f})", flush=True)
    for r in (2 * b, 2 * b + 1):
        os.remove(S / "redigi" / str(r) / "measurements.root")
