"""Stage A3 truth study + GNN v2 failure dig on m200_L774_mu0 test events 17000-19999 (runs 34-39).

Part 1 (truth): quirk hits made by quirk secondaries, multi-particle hits, a proposed truth rule and its effect on
the quirk denominators and the (R1 first-particle) oracle.
Part 2 (tracks): what the v2 tracks touching quirk hits are (counts, sizes, quirk A/B switches and where they
happen), coupled-quirk (pair) efficiency, event displays.

Matching here is a Python double-majority re-implementation (as quick_eval.py), not the ACTS evaluator.

  python dig.py --pred P/predictions.parquet --out DIR [--runs 34 35 36 37 38 39] [--workers 6]
"""

import argparse
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_events  # noqa: E402

RUNS_DIR = "/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/runs"
CACHE = "/pscratch/sd/d/danieltm/quirk_tracking/gnn/cache/m200_L774_mu0"
QPDG = 10000100
# simhit true_time is stored in ACTS units (mm/c): the values follow path length; 1 ns = 299.79 units
NS = 299.792458
EV_PER_RUN = 500
ARGS = None


def read_run(run):
    base = f"{RUNS_DIR}/{run}"
    H = pq.read_table(f"{base}/tracker_hits/tracker_hits_000000-100000.parquet",
                      columns=["event_id", "particle_ids", "simhit_ids", "volume_id", "layer_id", "surface_id"]).to_pydict()
    S = pq.read_table(f"{base}/tracker_simhits/tracker_simhits_000000-100000.parquet",
                      columns=["event_id", "particle_id", "true_time", "dE"]).to_pydict()
    P = pq.read_table(f"{base}/particles/particles_000000-100000.parquet",
                      columns=["event_id", "particle_id", "pdg_id", "parent_id", "energy", "px", "py", "pz"]).to_pydict()
    return H, S, P


def descendants(parent, roots):
    """{particle: (root quirk, generation)} for every descendant of the roots."""
    children = {}
    for i, p in enumerate(parent):
        if p >= 0:
            children.setdefault(int(p), []).append(i)
    out = {}
    for r in roots:
        stack = [(c, 1) for c in children.get(int(r), [])]
        while stack:
            c, gen = stack.pop()
            if c in out:
                continue
            out[c] = (int(r), gen)
            stack += [(cc, gen + 1) for cc in children.get(c, [])]
    return out


def match(tracks_hits, truth_hits, f):
    """Double majority at fraction f: some track with >= f of its hits in truth_hits and >= f of truth_hits."""
    tset = set(truth_hits)
    n = len(tset)
    best_eff, best_pur = 0.0, 0.0
    ok = False
    for th in tracks_hits:
        s = len(tset.intersection(th))
        if s == 0:
            continue
        eff, pur = s / n, s / len(th)
        if eff >= f and pur >= f:
            ok = True
        if eff > best_eff:
            best_eff, best_pur = eff, pur
    return ok, best_eff, best_pur


def turning_times(t, r, min_dr=2.0):
    """Times where r(t) of one quirk changes direction (a reversal must exceed min_dr mm); the time returned is
    that of the extreme hit."""
    o = np.argsort(t)
    t, r = t[o], r[o]
    out, sign = [], 0
    if len(r) == 0:
        return out
    ext_t, ext_r = t[0], r[0]
    for i in range(1, len(r)):
        d = r[i] - ext_r
        if sign == 0:
            if abs(d) >= min_dr:
                sign, ext_t, ext_r = np.sign(d), t[i], r[i]
        elif sign * d > 0:
            ext_t, ext_r = t[i], r[i]
        elif -sign * d >= min_dr:
            out.append(float(ext_t))
            sign, ext_t, ext_r = -sign, t[i], r[i]
    return out


def interp_pos(t, ts, pos):
    o = np.argsort(ts)
    ts, pos = ts[o], pos[o]
    return np.array([np.interp(t, ts, pos[:, k]) for k in range(3)])


def process_run(run):
    H, S, P = read_run(run)
    sidx = {e: i for i, e in enumerate(S["event_id"])}
    pidx = {e: i for i, e in enumerate(P["event_id"])}
    first_ev, last_ev = run * EV_PER_RUN, run * EV_PER_RUN + EV_PER_RUN - 1
    cache = {e["event_id"]: e for e in load_events(CACHE, first_ev, last_ev)}
    pred = pq.read_table(ARGS.pred, filters=[("event_id", ">=", first_ev), ("event_id", "<=", last_ev)]).to_pandas()
    pred = {int(e): g.sort_values("hit_id").predicted_track_id.to_numpy() for e, g in pred.groupby("event_id")}
    p1 = Counter()          # part-1 counters
    desc_rows, quirk_rows, multi_rows = [], [], []
    p2 = Counter()
    track_rows, switch_rows, same_rows, pair_rows, sep_all = [], [], [], [], []
    display_cands = []
    for k, loc in enumerate(H["event_id"]):
        if ARGS.max_events and k >= ARGS.max_events:
            break
        eid = run * EV_PER_RUN + int(loc)
        ev = cache[eid]
        js, jp = sidx[loc], pidx[loc]
        pdg = np.asarray(P["pdg_id"][jp])
        parent = np.asarray(P["parent_id"][jp])
        energy = np.asarray(P["energy"][jp])
        quirks = [int(q) for q in np.flatnonzero(np.abs(pdg) == QPDG)]
        desc = descendants(parent, quirks)
        sim_pid = np.asarray(S["particle_id"][js], np.int64)
        sim_t = np.asarray(S["true_time"][js], np.float32)
        sim_dE = np.asarray(S["dE"][js], np.float32)
        pids_l = H["particle_ids"][k]
        sims_l = H["simhit_ids"][k]
        n = len(pids_l)
        first = np.array([int(x[0]) if len(x) else -1 for x in pids_l])
        assert np.array_equal(first, ev["pid"]), eid
        t_hit = np.array([sim_t[s[0]] if len(s) else np.nan for s in sims_l])
        surf = (np.asarray(H["volume_id"][k], np.int64) * 10**8 + np.asarray(H["layer_id"][k], np.int64) * 10**5
                + np.asarray(H["surface_id"][k], np.int64))
        uniq = [list(dict.fromkeys(int(v) for v in x)) for x in pids_l]
        qset = set(quirks)
        p1["events"] += 1
        p1["hits"] += n
        p1["quirks"] += len(quirks)
        # hit categories
        rule = {q: [] for q in quirks}
        for i, u in enumerate(uniq):
            hasq = [q for q in u if q in qset]
            hasd = [q for q in u if q in desc]
            if len(u) > 1:
                p1["multi"] += 1
                if hasq:
                    p1["multi_with_quirk"] += 1
                    # dE per particle in this hit
                    dE = Counter()
                    for s in sims_l[i]:
                        dE[int(sim_pid[s])] += float(sim_dE[s])
                    top = max(dE, key=dE.get)
                    multi_rows.append(dict(n=len(u), both=len(hasq) == 2, first_is_quirk=u[0] in qset,
                                           first_is_top_dE=u[0] == top, other_is_desc=any(v in desc for v in u if v not in qset),
                                           other_pdg=[int(pdg[v]) for v in u if v not in qset][:3]))
                    if len(hasq) == 2:
                        p1["multi_both_quirks"] += 1
            for q in hasq:
                rule[q].append(i)
            if hasq:
                p1["hits_quirk_any"] += 1
                p1["hits_quirk_first"] += int(u[0] in qset)
                if hasd:
                    p1["hits_quirk_and_desc"] += 1
            elif hasd:
                p1["hits_desc_only"] += 1
                d0 = hasd[0]
                root, gen = desc[d0]
                qh = [j for j in range(n) if root in uniq[j]]
                same_surf = any(surf[j] == surf[i] and abs(t_hit[j] - t_hit[i]) < 1.0 * NS for j in qh)
                near = min([float(np.linalg.norm([ev["x"][j] - ev["x"][i], ev["y"][j] - ev["y"][i], ev["z"][j] - ev["z"][i]]))
                            for j in qh] or [np.nan])
                desc_rows.append(dict(pdg=int(pdg[d0]), gen=gen, energy=float(energy[d0]), same_surface=same_surf,
                                      dist_to_quirk_hit=near, labelled=int(u[0]), desc_hits=int((first == d0).sum())))
        # per-quirk truth comparison and oracle under the rule
        oracle = {}
        for i, f in enumerate(first):
            oracle.setdefault(int(f), []).append(i)
        oracle_tracks = [v for v in oracle.values()]
        for q in quirks:
            r1 = list(np.flatnonzero(first == q))
            rq = rule[q]
            row = dict(eid=eid, q=q, n_r1=len(r1), n_rule=len(rq), shared_other_quirk=sum(len([v for v in uniq[i] if v in qset]) == 2 for i in rq),
                       n_desc_only=sum(1 for i in range(n) if not (set(uniq[i]) & qset) and any(desc.get(v, (None,))[0] == q for v in uniq[i])))
            if len(rq) >= 3:
                for f, nm in ((0.5, "vl"), (0.9, "tight")):
                    row[f"oracle_{nm}"] = match(oracle_tracks, rq, f)[0]
            quirk_rows.append(row)
        # ---------------- part 2: tracks around the quirks
        lab = pred.get(eid)
        qq = [q for q in quirks if (first == q).sum() >= 3]
        if lab is None or len(qq) != 2:
            continue
        A, B = qq
        pos = np.stack([ev["x"], ev["y"], ev["z"]], 1).astype(np.float64)
        rr = np.hypot(pos[:, 0], pos[:, 1])
        hA, hB = np.flatnonzero(first == A), np.flatnonzero(first == B)
        pair = np.concatenate([hA, hB])
        tA, tB = t_hit[hA], t_hit[hB]
        turnA, turnB = turning_times(tA, rr[hA]), turning_times(tB, rr[hB])
        turns = np.array(turnA + turnB) if (turnA or turnB) else np.array([np.inf])
        p2["pairs"] += 1
        p2["turns"] += len(turnA) + len(turnB)
        # separation of the two quirks along time
        for tt in np.linspace(max(tA.min(), tB.min()), min(tA.max(), tB.max()), 20) if min(tA.max(), tB.max()) > max(tA.min(), tB.min()) else []:
            sep_all.append(float(np.linalg.norm(interp_pos(tt, tA, pos[hA]) - interp_pos(tt, tB, pos[hB]))))
        tids = [int(t) for t in np.unique(lab[pair]) if t >= 0]
        tracks = {t: np.flatnonzero(lab == t) for t in tids}
        all_tracks = [np.flatnonzero(lab == t) for t in np.unique(lab[lab >= 0])]
        # per-quirk
        for q, hq in ((A, hA), (B, hB)):
            ok5, be, bp = match(all_tracks, hq, 0.5)
            ok9 = match(all_tracks, hq, 0.9)[0]
            p2["quirks"] += 1
            p2["vl"] += ok5
            p2["tight"] += ok9
            p2["tight_fail_eff"] += (not ok9) and be < 0.9
            p2["tight_fail_pur"] += (not ok9) and be >= 0.9 and bp < 0.9
            p2["unassigned_hits"] += int((lab[hq] < 0).sum())
            p2["quirk_hits"] += len(hq)
        # coupled (pair) efficiency
        for f, nm in ((0.5, "05"), (0.9, "09")):
            single = match(all_tracks, pair, f)[0]
            major = [th for th in all_tracks if np.isin(th, pair).mean() >= f]
            u = np.unique(np.concatenate(major)) if major else np.array([], int)
            union = len(u) > 0 and np.isin(pair, u).mean() >= f
            p2[f"pair_single_{nm}"] += single
            p2[f"pair_union_{nm}"] += union
        both = match(all_tracks, hA, 0.5)[0] and match(all_tracks, hB, 0.5)[0]
        p2["both_quirks_vl"] += both
        nsw_event = 0
        for t, th in tracks.items():
            inA, inB = np.isin(th, hA), np.isin(th, hB)
            qh = th[inA | inB]
            qh = qh[np.argsort(t_hit[qh])]
            labs = np.where(np.isin(qh, hA), 0, 1)
            sw = np.flatnonzero(np.diff(labs) != 0)
            nsw_event += len(sw)
            track_rows.append(dict(eid=eid, n=len(th), nA=int(inA.sum()), nB=int(inB.sum()), n_other=int((~(inA | inB)).sum()),
                                   n_switch=len(sw), frac_major=max(inA.sum(), inB.sum()) / len(th)))
            for j in range(len(qh) - 1):
                a, b = qh[j], qh[j + 1]
                tm = 0.5 * (t_hit[a] + t_hit[b])
                d = float(np.linalg.norm(pos[a] - pos[b]))
                sep = float(np.linalg.norm(interp_pos(tm, tA, pos[hA]) - interp_pos(tm, tB, pos[hB])))
                rec = dict(t=float(tm), r=float(0.5 * (rr[a] + rr[b])), z=float(0.5 * (pos[a, 2] + pos[b, 2])), dist=d,
                           sep=sep, dt_turn=float(np.min(np.abs(turns - tm))), dt=float(abs(t_hit[b] - t_hit[a])))
                (switch_rows if labs[j] != labs[j + 1] else same_rows).append(rec)
        pair_rows.append(dict(eid=eid, n_tracks=len(tracks), n_switch=nsw_event, nA=len(hA), nB=len(hB),
                              vlA=match(all_tracks, hA, 0.5)[0], vlB=match(all_tracks, hB, 0.5)[0],
                              turns=len(turnA) + len(turnB)))
        display_cands.append(dict(eid=eid, A=A, B=B, n_switch=nsw_event, n_tracks=len(tracks),
                                  vl=int(match(all_tracks, hA, 0.5)[0]) + int(match(all_tracks, hB, 0.5)[0])))
    return dict(p1=p1, desc=desc_rows, quirk=quirk_rows, multi=multi_rows, p2=p2, tracks=track_rows, switches=switch_rows,
                same=same_rows, pairs=pair_rows, sep=sep_all, cands=display_cands)


def display(eid, A, B, pred_path, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = load_events(CACHE, eid, eid)[0]
    lab = pq.read_table(pred_path, filters=[("event_id", "=", eid)]).to_pandas().sort_values("hit_id").predicted_track_id.to_numpy()
    run, loc = divmod(eid, EV_PER_RUN)
    S = pq.read_table(f"{RUNS_DIR}/{run}/tracker_simhits/tracker_simhits_000000-100000.parquet",
                      columns=["event_id", "true_time"], filters=[("event_id", "=", loc)]).to_pydict()
    Hh = pq.read_table(f"{RUNS_DIR}/{run}/tracker_hits/tracker_hits_000000-100000.parquet",
                       columns=["event_id", "simhit_ids"], filters=[("event_id", "=", loc)]).to_pydict()
    st = np.asarray(S["true_time"][0])
    t = np.array([st[s[0]] for s in Hh["simhit_ids"][0]])
    x, y, z = ev["x"], ev["y"], ev["z"]
    r = np.hypot(x, y)
    pid = ev["pid"]
    hA, hB = pid == A, pid == B
    q = hA | hB
    tr = [int(v) for v in np.unique(lab[q]) if v >= 0]
    tr.sort(key=lambda v: -int(((lab == v) & q).sum()))
    track_cols = ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.2), facecolor="#fcfcfb")
    sel = q.copy()
    for v in tr:
        sel |= lab == v
    pad = 60
    for ax, (u, w, xl, yl) in zip(axs, [(x, y, "x [mm]", "y [mm]"), (z, r, "z [mm]", "r [mm]")]):
        ax.set_facecolor("#fcfcfb")
        lo_u, hi_u = u[sel].min() - pad, u[sel].max() + pad
        lo_w, hi_w = w[sel].min() - pad, w[sel].max() + pad
        box = (u > lo_u) & (u < hi_u) & (w > lo_w) & (w < hi_w)
        ax.scatter(u[box & ~q], w[box & ~q], s=4, c="#c3c2b7", lw=0, label="other hits", zorder=1)
        for k, v in enumerate(tr[:6]):
            th = np.flatnonzero(lab == v)
            th = th[np.argsort(t[th])]
            ax.plot(u[th], w[th], "-", lw=2, color=track_cols[k], alpha=0.85, zorder=2,
                    label=f"track {k + 1}: {int(hA[th].sum())} A + {int(hB[th].sum())} B + {int((~q[th]).sum())} other")
        for hm, c, nm, mk in ((hA, "#2a78d6", "quirk A", "o"), (hB, "#eb6834", "quirk B", "s")):
            ax.scatter(u[hm], w[hm], s=46, c=c, marker=mk, edgecolors="#fcfcfb", linewidths=1.5, zorder=3,
                       label=f"{nm} ({int(hm.sum())} hits)")
        un = q & (lab < 0)
        ax.scatter(u[un], w[un], s=90, facecolors="none", edgecolors="#0b0b0b", linewidths=1.2, zorder=4,
                   label="quirk hit not on any track")
        ax.set_xlim(lo_u, hi_u)
        ax.set_ylim(lo_w, hi_w)
        ax.set_xlabel(xl, color="#52514e")
        ax.set_ylabel(yl, color="#52514e")
        ax.grid(color="#e4e3df", lw=0.6)
        for s in ax.spines.values():
            s.set_color("#c3c2b7")
        ax.tick_params(colors="#52514e")
    axs[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    fig.suptitle(title, color="#0b0b0b", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def summarize(res, out):
    p1 = sum((r["p1"] for r in res), Counter())
    p2 = sum((r["p2"] for r in res), Counter())
    cat = lambda k: [x for r in res for x in r[k]]  # noqa: E731
    desc, quirk, multi = cat("desc"), cat("quirk"), cat("multi")
    tracks, sw, same, pairs, sep = cat("tracks"), cat("switches"), cat("same"), cat("pairs"), cat("sep")
    S = {}
    S["part1"] = dict(p1)
    S["part1"]["desc_only_by_pdg"] = Counter(d["pdg"] for d in desc).most_common(6)
    S["part1"]["desc_only_by_generation"] = sorted(Counter(d["gen"] for d in desc).items())
    S["part1"]["desc_only_same_surface_as_quirk_hit"] = float(np.mean([d["same_surface"] for d in desc])) if desc else None
    dd = np.array([d["dist_to_quirk_hit"] for d in desc])
    S["part1"]["desc_only_dist_to_nearest_quirk_hit_mm_q10_50_90"] = np.nanpercentile(dd, [10, 50, 90]).round(1).tolist() if len(dd) else None
    S["part1"]["desc_only_particle_hits_hist"] = sorted(Counter(min(d["desc_hits"], 5) for d in desc).items())
    S["part1"]["desc_only_energy_MeV_q10_50_90"] = (np.percentile([d["energy"] for d in desc], [10, 50, 90]) * 1000).round(2).tolist() if desc else None
    S["part1"]["multi_with_quirk_first_is_quirk"] = float(np.mean([m["first_is_quirk"] for m in multi]))
    S["part1"]["multi_with_quirk_first_is_top_dE"] = float(np.mean([m["first_is_top_dE"] for m in multi]))
    S["part1"]["multi_with_quirk_other_is_desc"] = float(np.mean([m["other_is_desc"] for m in multi if not m["both"]] or [0]))
    S["part1"]["multi_both_first_is_top_dE"] = float(np.mean([m["first_is_top_dE"] for m in multi if m["both"]] or [0]))
    qa = [q for q in quirk]
    n_r1 = np.array([q["n_r1"] for q in qa])
    n_rule = np.array([q["n_rule"] for q in qa])
    S["part1"]["quirk_hits_mean_r1_vs_rule"] = [float(n_r1.mean()), float(n_rule.mean())]
    S["part1"]["quirks_ge3_hits_r1_vs_rule"] = [int((n_r1 >= 3).sum()), int((n_rule >= 3).sum()), len(qa)]
    S["part1"]["quirks_gaining_hits_under_rule"] = int((n_rule > n_r1).sum())
    S["part1"]["quirk_desc_only_hits_per_quirk_mean"] = float(np.mean([q["n_desc_only"] for q in qa]))
    S["part1"]["quirk_desc_only_hits_frac_of_rule_hits"] = float(np.sum([q["n_desc_only"] for q in qa]) / n_rule.sum())
    sel = [q for q in qa if "oracle_vl" in q]
    S["part1"]["oracle_r1_assignment_under_rule_vl_tight"] = [float(np.mean([q["oracle_vl"] for q in sel])),
                                                              float(np.mean([q["oracle_tight"] for q in sel])), len(sel)]
    S["part2"] = dict(p2)
    for k in ["vl", "tight", "tight_fail_eff", "tight_fail_pur"]:
        S["part2"][k + "_rate"] = p2[k] / max(p2["quirks"], 1)
    for k in ["pair_single_05", "pair_union_05", "pair_single_09", "pair_union_09", "both_quirks_vl"]:
        S["part2"][k + "_rate"] = p2[k] / max(p2["pairs"], 1)
    S["part2"]["unassigned_quirk_hit_frac"] = p2["unassigned_hits"] / max(p2["quirk_hits"], 1)
    S["part2"]["tracks_per_pair_hist"] = sorted(Counter(min(p["n_tracks"], 8) for p in pairs).items())
    S["part2"]["tracks_touching_quirks_size_q10_50_90"] = np.percentile([t["n"] for t in tracks], [10, 50, 90]).tolist()
    S["part2"]["tracks_with_both_quirks_frac"] = float(np.mean([t["nA"] > 0 and t["nB"] > 0 for t in tracks]))
    S["part2"]["tracks_with_both_quirks_ge3_each_frac"] = float(np.mean([t["nA"] >= 3 and t["nB"] >= 3 for t in tracks]))
    S["part2"]["switches_per_pair_hist"] = sorted(Counter(min(p["n_switch"], 10) for p in pairs).items())
    S["part2"]["switches_per_mixed_track_mean"] = float(np.mean([t["n_switch"] for t in tracks if t["nA"] and t["nB"]] or [0]))
    S["part2"]["tracks_other_hits_frac"] = float(np.sum([t["n_other"] for t in tracks]) / np.sum([t["n"] for t in tracks]))
    S["part2"]["turning_points_per_pair"] = p2["turns"] / max(p2["pairs"], 1)
    for nm, rows in (("switch", sw), ("same_quirk_step", same)):
        if not rows:
            continue
        g = lambda k: np.array([x[k] for x in rows])  # noqa: E731
        S["part2"][nm] = dict(n=len(rows), r_q10_50_90=np.percentile(g("r"), [10, 50, 90]).round(0).tolist(),
                              absz_q10_50_90=np.percentile(np.abs(g("z")), [10, 50, 90]).round(0).tolist(),
                              t_ns_q10_50_90=(np.percentile(g("t"), [10, 50, 90]) / NS).round(2).tolist(),
                              hit_dist_mm_q10_50_90=np.percentile(g("dist"), [10, 50, 90]).round(1).tolist(),
                              quirk_sep_mm_q10_50_90=np.percentile(g("sep"), [10, 50, 90]).round(1).tolist(),
                              frac_sep_lt_10mm=float((g("sep") < 10).mean()), frac_sep_lt_30mm=float((g("sep") < 30).mean()),
                              frac_within_1ns_of_turn=float((g("dt_turn") < NS).mean()),
                              dt_between_hits_ns_q50=float(np.median(g("dt")) / NS))
    if sep:
        S["part2"]["pair_separation_all_times_mm_q10_50_90"] = np.percentile(sep, [10, 50, 90]).round(1).tolist()
        S["part2"]["pair_separation_frac_lt_10mm"] = float((np.array(sep) < 10).mean())
        S["part2"]["pair_separation_frac_lt_30mm"] = float((np.array(sep) < 30).mean())
    json.dump(S, open(os.path.join(out, "dig_summary.json"), "w"), indent=1, default=float)
    json.dump(dict(switches=sw, same=same), open(os.path.join(out, "switch_rows.json"), "w"), default=float)
    return S


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, nargs="+", default=[34, 35, 36, 37, 38, 39])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-events", type=int, default=0, help="per run, for tests")
    ARGS = ap.parse_args()
    os.makedirs(ARGS.out, exist_ok=True)
    with Pool(ARGS.workers) as pool:
        res = pool.map(process_run, ARGS.runs)
    S = summarize(res, ARGS.out)
    print(json.dumps(S, indent=1, default=float))
    cands = [c for r in res for c in r["cands"]]
    rng = np.random.default_rng(1)
    picks = []
    for nm, pool_ in (("fail_both", [c for c in cands if c["vl"] == 0]),
                      ("one_found", [c for c in cands if c["vl"] == 1]),
                      ("many_switches", sorted(cands, key=lambda c: -c["n_switch"])[:30]),
                      ("both_found", [c for c in cands if c["vl"] == 2])):
        if pool_:
            c = pool_[rng.integers(len(pool_))]
            picks.append((nm, c))
    for nm, c in picks:
        png = os.path.join(ARGS.out, f"display_{nm}_ev{c['eid']}.png")
        display(c["eid"], c["A"], c["B"], ARGS.pred, png,
                f"event {c['eid']} ({nm.replace('_', ' ')}): {c['n_tracks']} tracks touch quirk hits, "
                f"{c['n_switch']} A/B switches; quirks found at 50%: {c['vl']}/2")
        print("wrote", png)


if __name__ == "__main__":
    main()
