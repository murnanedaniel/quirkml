"""Tables from fisher.jsonl, fits/*.jsonl and mcmc/*.jsonl (quirk parameter study C1).

    python analyze.py [OUTDIR]   -> prints and writes OUTDIR/summary_{fisher,fits,mcmc}.txt
"""
import glob
import json
import sys

import numpy as np
import pandas as pd

O = sys.argv[1] if len(sys.argv) > 1 else "/pscratch/sd/d/danieltm/quirk_tracking/params"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def med(s):
    s = np.asarray(s, float); s = s[np.isfinite(s)]
    return np.median(s) if len(s) else np.nan


def rng(s):
    s = np.asarray(s, float); s = s[np.isfinite(s)]
    return "[%.3g, %.3g]" % tuple(np.percentile(s, [16, 84])) if len(s) else "-"


def hw(s):
    """half of the 16-84% width (a robust sigma)"""
    s = np.asarray(s, float); s = s[np.isfinite(s)]
    return 0.5 * np.diff(np.percentile(s, [16, 84]))[0] if len(s) > 2 else np.nan


def load(pattern):
    rows = []
    for f in glob.glob(pattern):
        for line in open(f):
            r = json.loads(line)
            if "error" not in r:
                rows.append(r)
    return rows


def fisher():
    R = load(f"{O}/fisher.jsonl")
    if not R:
        return ""
    out = []
    for floor in sorted({r["floor"] for r in R}):
        rows = []
        for (m, lam) in sorted({(r["m"], r["lam"]) for r in R}):
            ev = [r for r in R if r["m"] == m and r["lam"] == lam and r["floor"] == floor]
            g = lambda k, f: [e[k].get(f, np.nan) if isinstance(e.get(k), dict) else np.nan for e in ev]
            row = dict(m=m, lam=lam, n=len(ev), nhits=med([e["nhits_B3T"] for e in ev]))
            row["s_lnm"] = med(g("B3T_none_vfree", "s_lnm")); row["s_lnm_16_84"] = rng(g("B3T_none_vfree", "s_lnm"))
            row["s_lnL"] = med(g("B3T_none_vfree", "s_lnL"))
            row["s_ln(sig/m)"] = med(g("B3T_none_vfree", "s_ln_sig_over_m"))
            row["s_ln(m/L)"] = med(g("B3T_none_vfree", "s_ln_m_over_L"))
            row["rho"] = med(g("B3T_none_vfree", "rho"))
            row["soft_cos"] = med(g("B3T_none_vfree", "soft_cos"))
            row["s_lnm_vfix"] = med(g("B3T_none_vfix", "s_lnm"))
            for t in ("100ps", "30ps", "10ps"):
                row[f"s_lnm_{t}"] = med(g(f"B3T_{t}_vfree", "s_lnm"))
            row["s_lnL_10ps"] = med(g("B3T_10ps_vfree", "s_lnL"))
            row["B0_singular"] = np.mean([bool(e["B0_none_vfree"].get("singular", False)) for e in ev])
            row["B0_s_lnm"] = med(g("B0_none_vfree", "s_lnm"))
            row["B0_s_lnm_10ps"] = med(g("B0_10ps_vfree", "s_lnm"))
            row["B0_s_ln(sig/m)"] = med(g("B0_none_vfree", "s_ln_sig_over_m"))
            rows.append(row)
        df = pd.DataFrame(rows)
        out.append(f"== Fisher (local, at truth), floor {floor} mm: medians over events ==\n" + df.round(4).to_string(index=False))
    return "\n\n".join(out)


def prof_width(s, c):
    s = np.asarray(s); c = np.asarray(c, float)
    c = c - np.nanmin(c)
    i0 = int(np.nanargmin(c))
    w = []
    for side in (-1, 1):
        j = i0
        while 0 <= j + side < len(s) and c[j + side] < 1:
            j += side
        if not (0 <= j + side < len(s)):
            w.append(np.inf); continue
        # linear interpolation in sqrt(dchi2)
        a, b = np.sqrt(c[j]), np.sqrt(c[j + side])
        w.append(abs(s[j] + (s[j + side] - s[j]) * (1 - a) / (b - a) - s[i0]) if b != a else abs(s[j + side] - s[i0]))
    return 0.5 * (w[0] + w[1]), s[i0], c


def fits():
    R = load(f"{O}/fits/fit_*.jsonl")
    if not R:
        return ""
    rows = []
    for r in R:
        th = np.array(r["truth"])
        best = min(r["fits"], key=lambda f: f["chi2"])
        for f in r["fits"]:
            d = np.array(f["theta"]) - th
            row = dict(m=r["m"], lam=r["lam"], ev=r["ev"], hits=r["hits"], times=r["times"], f=f["f"], chi2=f["chi2"],
                       ndof=r["ndof"], chi2_truth=r["chi2_truth"], dlnm=d[0], dlnL=d[1], dlsm=2 * d[1] - d[0],
                       s_lnm=f.get("s_lnm", np.nan), s_lnL=f.get("s_lnL", np.nan),
                       fail=(f["chi2"] > best["chi2"] + 9) or not np.isfinite(f["chi2"]),
                       above_truth=f["chi2"] > r["chi2_truth"] + 9)
            if "profile" in r and f["f"] == min(x["f"] for x in r["fits"]) and \
                    all(np.all(np.isfinite(r["profile"][k]["chi2"])) for k in ("deg", "orth")):
                wd, s0, cd = prof_width(r["profile"]["deg"]["s"], r["profile"]["deg"]["chi2"])
                wo, so, co = prof_width(r["profile"]["orth"]["s"], r["profile"]["orth"]["chi2"])
                sd = np.asarray(r["profile"]["deg"]["s"])
                row.update(prof_deg=wd, prof_deg_min=s0, prof_orth=wo, prof_orth_min=so,
                           dchi2_deg_far=float(np.nanmin(cd[np.abs(sd - s0) >= 0.29])),
                           second_min=bool(np.any((cd < 4) & (np.abs(sd - s0) >= 0.15))))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{O}/fits/fits_table.csv", index=False)
    out = []
    for (hits, times), g in df.groupby(["hits", "times"]):
        tab = []
        for (m, lam), h in g.groupby(["m", "lam"]):
            row = dict(m=m, lam=lam, n=h.ev.nunique())
            for f, hf in h.groupby("f"):
                ok = hf[~hf.fail]
                tag = f"{int(round(f * 100))}%"
                row[f"fail_{tag}"] = hf.fail.mean()
                row[f"bias_lnm_{tag}"] = ok.dlnm.median(); row[f"res_lnm_{tag}"] = hw(ok.dlnm)
                row[f"bias_lnL_{tag}"] = ok.dlnL.median(); row[f"res_lnL_{tag}"] = hw(ok.dlnL)
                row[f"res_ln(sig/m)_{tag}"] = hw(ok.dlsm)
                row[f"pull_lnm_{tag}"] = hw(ok.dlnm / ok.s_lnm)
                row[f"hess_s_lnm_{tag}"] = ok.s_lnm.median()
            row["chi2/ndof"] = (h.chi2 / h.ndof).median()
            if "prof_deg" in h:
                p = h.dropna(subset=["prof_deg"])
                row["prof_s_deg"] = med(p.prof_deg); row["prof_s_orth"] = med(p.prof_orth)
                row["prof_deg_min_bias"] = med(p.prof_deg_min); row["prof_deg_min_spread"] = hw(p.prof_deg_min)
                row["frac_second_min"] = p.second_min.mean()
            tab.append(row)
        out.append(f"== fits: hits={hits}, times={times} (fail = chi2 > best start + 9; res = half 16-84% width of fit - truth over events) ==\n"
                   + pd.DataFrame(tab).round(4).to_string(index=False))
    return "\n\n".join(out)


def mcmc():
    R = load(f"{O}/mcmc/*.jsonl")
    if not R:
        return ""
    rows = []
    for r in R:
        th = r["truth"]
        lo, hi = r["q_lnm"][1], r["q_lnm"][3]
        lo2, hi2 = r["q_lnm"][0], r["q_lnm"][4]
        rows.append(dict(m=r["m"], lam=r["lam"], hits=r["hits"], ev=r["ev"], acc=r["acc"], std_lnm=r["std_lnm"], std_lnL=r["std_lnL"],
                         gn_s_lnm=r["gn_s_lnm"], rho=r["rho"], hw68_lnm=0.5 * (hi - lo), hw95_lnm=0.25 * (hi2 - lo2),
                         hw68_lsm=0.5 * (r["q_ln_sig_over_m"][3] - r["q_ln_sig_over_m"][1]),
                         cover68=lo <= th[0] <= hi, cover95=lo2 <= th[0] <= hi2, sec=r["sec"]))
    df = pd.DataFrame(rows)
    tab = df.groupby(["hits", "m", "lam"]).agg(n=("ev", "size"), acc=("acc", "median"), mcmc_hw68_lnm=("hw68_lnm", "median"),
                                              mcmc_hw95_over_2=("hw95_lnm", "median"), gn_s_lnm=("gn_s_lnm", "median"),
                                              hw68_ln_sig_over_m=("hw68_lsm", "median"), rho=("rho", "median"),
                                              cover68=("cover68", "mean"), cover95=("cover95", "mean"), sec=("sec", "median"))
    return "== MCMC (flat priors; medians over events; cover = fraction of events whose 68/95% interval holds the truth ln m) ==\n" + tab.round(4).to_string()


if __name__ == "__main__":
    for name, fn in (("fisher", fisher), ("fits", fits), ("mcmc", mcmc)):
        s = fn()
        if s:
            open(f"{O}/summary_{name}.txt", "w").write(s + "\n")
            print(s + "\n")
