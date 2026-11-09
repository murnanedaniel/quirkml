"""Sum chunked evaluate.py outputs (OUT/chunk-*/summary.json) into OUT/summary_combined.json: efficiencies (quirk track,
quirk event, technical, technical_sm) from summed accepted/total, fake and duplicate rates from summed track classes,
per working point. Fails if the chunks do not cover [A, B) exactly once. Usage: combine_eval_chunks.py OUT A B"""
import json
import sys
from pathlib import Path

out, first, stop = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
chunks = sorted(out.glob("chunk-*/summary.json"), key=lambda p: int(p.parent.name.split("-")[1]))
covered = []
for p in chunks:
    a, b = (int(x) for x in p.parent.name.split("-")[1:3])
    covered.append((a, b + 1))
expect, pos = [], first
for a, b in covered:
    if a != pos:
        raise SystemExit(f"chunks do not cover [{first}, {stop}) contiguously: gap/overlap at {pos} vs {a}")
    pos = b
if pos != stop:
    raise SystemExit(f"chunks end at {pos}, expected {stop}")
EFF = ("quirk_track_efficiency", "quirk_event_efficiency", "technical_efficiency", "technical_sm_efficiency")
res = {"chunks": len(chunks), "events": [first, stop]}
for key in EFF:
    res[key] = {}
for key in ("track_classification_all_particles",):
    res[key] = {}
for p in chunks:
    s = json.loads(p.read_text())
    for key in EFF:
        for wp, v in s[key].items():
            r = res[key].setdefault(wp, {"accepted": 0.0, "total": 0.0})
            r["accepted"] += float(v.get("accepted") or 0); r["total"] += float(v.get("total") or 0)
    for wp, v in s["track_classification_all_particles"].items():
        r = res["track_classification_all_particles"].setdefault(wp, {k: 0 for k in ("tracks", "matched", "duplicate", "fake", "unassociated")})
        for k in r:
            r[k] += int(v[k])
for key in EFF:
    for wp, r in res[key].items():
        r["eff"] = r["accepted"] / r["total"] if r["total"] else None
for wp, r in res["track_classification_all_particles"].items():
    assoc = r["matched"] + r["duplicate"] + r["fake"]
    r["fake_rate"] = r["fake"] / assoc if assoc else None
    r["duplicate_rate"] = r["duplicate"] / assoc if assoc else None
(out / "summary_combined.json").write_text(json.dumps(res, indent=1))
for wp in ("very_loose_strict", "tight_strict", "very_loose", "tight"):
    if wp in res["quirk_track_efficiency"]:
        t = res["track_classification_all_particles"][wp]
        print(f"{wp}: quirk {res['quirk_track_efficiency'][wp]['eff']:.4f} ({res['quirk_track_efficiency'][wp]['accepted']:.0f}/"
              f"{res['quirk_track_efficiency'][wp]['total']:.0f})  tech_sm {res['technical_sm_efficiency'][wp]['eff']:.4f}  "
              f"fake {t['fake_rate']:.4f}  dup {t['duplicate_rate']:.4f}")
