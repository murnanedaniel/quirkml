"""Compare py_score.py with chunked ACTS counts, chunk by chunk (every chunk-A-B/summary.json under ACTS_OUT).
    python compare_py_acts.py PRED_DIR DATASET ACTS_OUT WORK_DIR
Prints one line per chunk and working point with any differing count; exits 1 on any difference, on a chunk directory
without summary.json, or when no chunk was compared. A cached Python summary is reused only if it was scored on the
same predictions, dataset and events; otherwise it is rescored."""
import json
import subprocess
import sys
from pathlib import Path

pred, ds, acts, work = sys.argv[1:5]
work = Path(work); work.mkdir(parents=True, exist_ok=True)
PY = sys.executable
SCORE = str(Path(__file__).with_name("py_score.py"))
SEL = {"quirk": "quirk_track_efficiency", "quirk_event": "quirk_event_efficiency", "technical": "technical_efficiency",
       "technical_sm": "technical_sm_efficiency"}
bad = 0
compared = 0
dirs = sorted((p for p in Path(acts).glob("chunk-*-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[1]))
missing = [d.name for d in dirs if not (d / "summary.json").is_file()]
if missing:
    sys.exit(f"ACTS chunks without summary.json under {acts}: {', '.join(missing)}")
if not dirs:
    sys.exit(f"no ACTS chunk directories under {acts}")


def cached_ok(path, a, b):
    if not path.exists():
        return False
    r = json.loads(path.read_text())
    return (r.get("events") == [a, b] and r.get("predictions") == str(Path(pred))
            and r.get("dataset") == str(Path(ds)) and "unassociated" in r.get("very_loose", {}).get("tracks", {}))


for d in dirs:
    a, b = int(d.name.split("-")[1]), int(d.name.split("-")[2]) + 1
    out = work / f"py_{a}_{b}.json"
    if not cached_ok(out, a, b):
        subprocess.run([PY, SCORE, pred, ds, str(a), str(b), str(out)], check=True, stdout=subprocess.DEVNULL)
    s, r = json.loads((d / "summary.json").read_text()), json.loads(out.read_text())
    compared += 1
    for wp in ("very_loose", "tight", "very_loose_strict", "tight_strict"):
        diffs = []
        for k, v in SEL.items():
            x = s[v][wp]
            if (int(x["accepted"]), int(x["total"])) != (r[wp][k]["accepted"], r[wp][k]["total"]):
                diffs.append(f"{k} acts {int(x['accepted'])}/{int(x['total'])} py {r[wp][k]['accepted']}/{r[wp][k]['total']}")
        t = s["track_classification_all_particles"][wp]
        for k in ("tracks", "matched", "duplicate", "fake", "unassociated"):
            if int(t[k]) != r[wp]["tracks"][k]:
                diffs.append(f"{k} acts {t[k]} py {r[wp]['tracks'][k]}")
        bad += bool(diffs)
        print(f"{d.name} {wp}: " + ("identical" if not diffs else "; ".join(diffs)), flush=True)
if not compared:
    sys.exit("no chunk was compared")
print(f"{compared} chunks compared, {bad} chunk/working-point rows differ", flush=True)
sys.exit(1 if bad else 0)
