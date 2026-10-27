"""Print the B2 / B3 summary (markdown) from quality.py result.json files: python summarize.py B2_ROOT MODEL..."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
print("| Model | cut | val cut | fake (own truth) | per-quirk | quirk event | coupled | technical_sm | tracks / ev |")
print("|---|---|---|---|---|---|---|---|---|")
for m in sys.argv[2:]:
    r = json.loads((root / m / "result.json").read_text())
    fc = "fake_rate" if r["own_truth"] == "normal" else "fake_rate_pair"
    row = lambda name, cut, x: print(f"| {m} | {name} | {cut} | {f(x[fc])} | {f(x['quirk_eff'])} | "  # noqa: E731
                                     f"{f(x['quirk_event_eff'])} | {f(x['coupled_eff'])} | {f(x['technical_sm_eff'])} | {x['tracks_per_event']:.1f} |")
    row("no cut", "-", r["b2"]["quality"]["no_cut_test"])
    if "min_hits_4_test" in r["b2"]["n_hits"]:
        row(">= 4 hits", "4", r["b2"]["n_hits"]["min_hits_4_test"])
    for name in [n for n in ("n_hits", "quality", "quality_qw", "quality_or_tag") if n in r["b2"]]:
        for op in r["b2"][name]["op"]:
            label = f"{'hit count' if name == 'n_hits' else name} @ val fake {op['target']}"
            if op.get("test"):
                c = op["cut"]
                row(label, "none needed" if c < -1e299 else (f"{c:.0f}" if name == "n_hits" else f"{c:.3f}"), op["test"])
            else:
                print(f"| {m} | {label} | not reached on val | | | | | | |")
print()
print("| Model | quirk-track definition | quirk-track eff val B / test | ordinary tracks tagged / event (test) | of ordinary tracks / event | events with >= 1 tagged ordinary |")
print("|---|---|---|---|---|---|")
for m in sys.argv[2:]:
    r = json.loads((root / m / "result.json").read_text())
    for k, v in r["b3"].items():
        if not isinstance(v, dict):
            continue
        print(f"| {m} | {k} | {v['quirk_track_eff_valB']:.3f} / {v['quirk_track_eff_test']:.3f} | "
              f"{v['ordinary_tagged_per_event_test']:.2f} | {v['ordinary_tracks_per_event_test']:.1f} | "
              f"{v['events_with_ordinary_tagged_test']:.3f} |")
