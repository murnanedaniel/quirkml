"""Collect the GNN data-scaling results: strict ACTS numbers (val and test) per arm, a CSV, a markdown table and the
efficiency-vs-training-events plot (log x), per-quirk and coupled, loose (> 50%) and tight (> 90%).
  python collect.py   (reads /pscratch/sd/d/danieltm/quirk_tracking/scaling/eval/*, writes scaling_results.{csv,md,png})
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

S = "/pscratch/sd/d/danieltm/quirk_tracking/scaling"
E = os.path.join(S, "eval")
# (task, training events, seed, steps, eval dir name)
ARMS = [
    ("per-quirk", 16000, 0, 64000, "ctrl_g2_q50_dir_s0"), ("per-quirk", 16000, 1, 64000, "ctrl_g2_q50_dir_s1"),
    ("coupled", 16000, 0, 64000, "ctrl_e2_pair_dir_s0"), ("coupled", 16000, 1, 64000, "ctrl_e2_pair_dir_s1"),
    ("coupled", 16000, 2, 64000, "ctrl_e2_pair_dir_s2"),
]
for tag in sorted(os.listdir(E)) if os.path.isdir(E) else []:
    if tag.startswith("ctrl_"):
        continue
    kind, n, seed = tag.split("_")[:3]  # e.g. q_50k_s0 / p_150k_s1 (runs/<tag>, early stop on strict val)
    sel = os.path.join(S, "runs", tag, "gnn", "best_sel.json")
    st = json.load(open(sel)) if os.path.exists(sel) else {}
    ARMS.append(("per-quirk" if kind == "q" else "coupled", int(n[:-1]) * 1000, int(seed[1:]),
                 (st.get("history") or [{"step": 0}])[-1]["step"], tag))


def get(path, *keys):
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        return None
    for k in keys:
        d = d[k]
    return d


def row(task, n, seed, steps, d):
    r = dict(task=task, train_events=n if n != 16000 else 16000, seed=seed, steps=steps, tag=d)
    base = os.path.join(E, d)
    qsrc = "evalpair" if task == "coupled" else "eval"
    for split in ("val", "test"):
        q = get(os.path.join(base, f"{split}_{qsrc}", "summary.json"), "quirk_track_efficiency")
        nt = os.path.join(base, f"{split}_eval", "summary.json")
        if q is None:
            continue
        for lvl, name in (("very_loose_strict", "loose"), ("tight_strict", "tight")):
            r[f"{split}_{name}"] = q[lvl]["eff"]
            r[f"{split}_{name}_err"] = q[lvl]["err"]
        if task == "per-quirk":
            r[f"{split}_event"] = get(nt, "quirk_event_efficiency", "very_loose_strict", "eff")
        r[f"{split}_fake"] = get(nt, "fake_rate", "very_loose_strict")
        r[f"{split}_tech_sm"] = get(nt, "technical_sm_efficiency", "very_loose_strict", "eff")
    sel = os.path.join(S, "runs", d, "gnn", "best_sel.json")
    if os.path.exists(sel):
        j = json.load(open(sel))
        r["best_step"], r["early_stopped"], r["best_val_python"] = j["best_step"], j["stopped"], j["best"]
    cut = os.path.join(base, "cut.txt")
    if os.path.exists(cut):
        r["cc_cut"] = open(cut).read().split()[-1]
    return r


def main():
    rows = [row(*a) for a in ARMS]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(S, "scaling_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    f = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    lines = ["| Task | Training events | Seed | Steps run / best | Test strict loose / tight | Quirk event (strict) | Fake (strict) | "
             "technical_sm (strict) | Val strict loose / tight |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["task"] != "per-quirk", r["train_events"], r["steps"], r["seed"])):
        lines.append(f"| {r['task']} | {r['train_events'] // 1000}k{' (control)' if r['tag'].startswith('ctrl_') else ''} | {r['seed']} | "
                     f"{r['steps'] // 1000}k / {(r['best_step'] // 1000) if 'best_step' in r else '-'}{'k' if 'best_step' in r else ''} | "
                     f"{f(r.get('test_loose'))} / {f(r.get('test_tight'))} | {f(r.get('test_event'))} | "
                     f"{f(r.get('test_fake'))} | {f(r.get('test_tech_sm'))} | {f(r.get('val_loose'))} / {f(r.get('val_tight'))} |")
    open(os.path.join(S, "scaling_results.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(rows)


def plot(rows):
    """Test strict efficiency vs training events. Lines: arms trained by the common rule (early stop on strict val,
    patience 3, cap 256k steps), mean over seeds. Squares: the existing 16k controls (64k-step cosine, not trained
    by the rule), mean over seeds. Error bars: binomial error, or the seed std when larger. Open circles: val."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"loose": "#2a78d6", "tight": "#eb6834"}

    def agg(rs, key):
        by = defaultdict(list)
        for r in rs:
            if key in r:
                by[r["train_events"]].append(r)
        xs = sorted(by)
        m = [np.mean([r[key] for r in by[x]]) for x in xs]
        e = [max(np.mean([r.get(key + "_err", 0) for r in by[x]]),
                 np.std([r[key] for r in by[x]], ddof=1) if len(by[x]) > 1 else 0) for x in xs]
        return xs, m, e

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, task in zip(axes, ("per-quirk", "coupled")):
        rule = [r for r in rows if r["task"] == task and not r["tag"].startswith("ctrl_")]
        ctrl = [r for r in rows if r["task"] == task and r["tag"].startswith("ctrl_")]
        for lvl in ("loose", "tight"):
            xs, m, e = agg(rule, f"test_{lvl}")
            if xs:
                ax.errorbar(xs, m, yerr=e, color=col[lvl], marker="o", ms=7, lw=2, capsize=3, label=f"{lvl} (> {50 if lvl == 'loose' else 90}%), test")
            xs, m, e = agg(rule, f"val_{lvl}")
            if xs:
                ax.plot(xs, m, color=col[lvl], marker="o", ms=7, mfc="white", lw=1, ls="--", label=f"{lvl}, val")
            xs, m, e = agg(ctrl, f"test_{lvl}")
            if xs:
                ax.errorbar([x * 0.93 for x in xs], m, yerr=e, color=col[lvl], marker="s", ms=8, mfc="white", ls="none",
                            capsize=3, label=f"{lvl}, test, 16k control (64k-step cosine)")
        ax.set_xscale("log")
        ax.set_xticks([16000, 50000, 150000, 400000])
        ax.set_xticklabels(["16k", "50k", "150k", "400k"])
        ax.minorticks_off()
        ax.set_xlim(12000, 200000 if max(r["train_events"] for r in rows) < 300000 else 500000)
        ax.set_ylim(0, 1)
        ax.set_xlabel("training events")
        ax.set_ylabel("strict efficiency (ACTS)")
        ax.set_title("per-quirk GNN, graph v3, walk + stitch" if task == "per-quirk" else "pair GNN, coupled, components")
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(S, "scaling_results.png"), dpi=150)
    print("plot", os.path.join(S, "scaling_results.png"))


if __name__ == "__main__":
    main()
