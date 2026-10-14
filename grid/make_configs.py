"""Write ColliderML stage configs for one grid point from the pilot templates.

    python make_configs.py --mass 400 --lambda-ev 3000 --mu 200 --events 50 \
        --base <run tree base> --quirkgen-json <quirk-gen events.json> --out <config dir>

Templates: colliderml/pilot_m400_L1000_mu0 (mu = 0) and colliderml/pilot_m400_L1000_mu200 (mu > 0).
The generated configs still use interactive mode and debug_output_dir; read the ColliderML runbook
before turning them into batch production configs.
"""
import argparse, json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TPL0 = HERE.parent / "colliderml" / "pilot_m400_L1000_mu0"
TPLP = HERE.parent / "colliderml" / "pilot_m400_L1000_mu200"
OLD_BASE = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_erc/out/stage3/pilot_tree"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, required=True, help="GeV")
    ap.add_argument("--lambda-ev", type=float, required=True)
    ap.add_argument("--mu", type=int, default=0)
    ap.add_argument("--events", type=int, required=True)
    ap.add_argument("--base", required=True, help="run tree base (replaces the pilot tree)")
    ap.add_argument("--campaign", default="quirk_pilot")
    ap.add_argument("--version", default="v0")
    ap.add_argument("--quirkgen-json", required=True, help="quirk-gen events.json for this mass")
    ap.add_argument("--max-boost", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = json.load(open(a.quirkgen_json))
    if abs(meta["mass_gev"] - a.mass) > 1e-6:
        raise SystemExit(f"events.json is for mass {meta['mass_gev']} GeV, not {a.mass}")
    sigma = a.lambda_ev ** 2 / 1.973269804e-4 / 1e6          # MeV/mm, Lambda^2 / hbar c
    dataset = f"m{a.mass:g}_L{a.lambda_ev:g}_mu{a.mu}"
    tpl = TPL0 if a.mu == 0 else TPLP
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in sorted(tpl.glob("*.yaml")):
        s = f.read_text()
        s = s.replace(OLD_BASE, a.base).replace("m400_L1000_mu0", dataset).replace("m400_L1000_mu200", dataset)
        s = s.replace('campaign: "quirk_pilot"', f'campaign: "{a.campaign}"').replace('version: "v0"', f'version: "{a.version}"')
        s = s.replace("/quirk_pilot/", f"/{a.campaign}/").replace("/v0/runs/", f"/{a.version}/runs/")
        s = re.sub(r"^events: \d+", f"events: {a.events}", s, flags=re.M)
        s = re.sub(r"^run_size: \d+", f"run_size: {a.events}", s, flags=re.M)
        s = re.sub(r"^chunk_size: \d+", f"chunk_size: {a.events}", s, flags=re.M)
        s = re.sub(r"^pileup: \d+", f"pileup: {a.mu}", s, flags=re.M)
        s = re.sub(r"Mass: [0-9.e+]+ ", f"Mass: {a.mass * 1e3:.1f} ", s)
        s = re.sub(r"PdgId: \d+", f"PdgId: {meta['pdgid']}", s)
        s = re.sub(r"Charge: [0-9.]+", f"Charge: {float(meta['charge']):.1f}", s)
        s = re.sub(r"StringForce: [0-9.e+]+ +#.*", f"StringForce: {sigma:.6g}   # MeV/mm, Lambda^2 / hbar c for Lambda = {a.lambda_ev:g} eV", s)
        s = re.sub(r"MaxBoost: [0-9.e+-]+", f"MaxBoost: {a.max_boost:g}", s)
        (out / f.name).write_text(s)
    print(f"wrote {len(list(out.glob('*.yaml')))} configs for {dataset} to {out}")
    print(f"next: put the converted quirk-gen file at {a.base}/{a.campaign}/{dataset}/{a.version}/runs/0/events_hepmc2.hepmc")


if __name__ == "__main__":
    main()
