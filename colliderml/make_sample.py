"""Write the ColliderML stage configs for one quirk sample from a settings file.

    python3 colliderml/make_sample.py my_sample.yaml

Writes <work_dir>/configs/<dataset>/ with the three stage configs and sample.env (the resolved
settings, read by gen/run_quirkgen.sh and colliderml/run_sample.sh), and creates the run
directory <work_dir>/runs/<dataset>/. Templates: colliderml/templates/. Stage 1 is merge_smear for
mu = 0 and pythia_generation with merging for mu > 0. The configs use the paths inside the
container (/work = work_dir, /workspace = colliderml-production). Needs only python3.
"""
import argparse
import os
import shlex
import sys
from pathlib import Path

QUIRKML = Path(__file__).resolve().parent.parent
TEMPLATES = QUIRKML / "colliderml" / "templates"
HBARC_MEV_MM = 1.973269804e-10      # hbar c in MeV mm
PDGID, CHARGE = 10000100, 1.0       # quirk-gen: +10000100 has charge +1 (checked against events.json by run_sample.sh)
KEYS = ("work_dir", "mass_gev", "lambda_ev", "mu", "events", "seed", "max_boost")


def read_settings(path):
    """The settings file is flat 'key: value  # comment' lines (a YAML subset), read without pyyaml."""
    s = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            k, _, v = line.partition(":")
            s[k.strip()] = v.strip().strip("'\"")
    missing = [k for k in KEYS if k not in s]
    if missing:
        sys.exit(f"{path}: missing {', '.join(missing)}")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("settings")
    a = ap.parse_args()
    s = read_settings(a.settings)

    work = Path(os.path.expanduser(os.path.expandvars(s["work_dir"]))).resolve()
    lam, mass, mu, events, seed = float(s["lambda_ev"]), float(s["mass_gev"]), int(s["mu"]), int(s["events"]), int(s["seed"])
    dataset = f"m{mass:g}_L{lam:g}_mu{mu}"
    if not (work / "colliderml-production" / "scripts").is_dir():
        sys.exit(f"no colliderml-production checkout in {work}: run colliderml/setup.sh {work} first")

    values = dict(
        DATASET=dataset, EVENTS=events, MU=mu,
        SEED_GEN=seed + 41, SEED_SIM=seed + 42, SEED_DIGI=seed + 43,
        MASS_MEV=f"{mass * 1e3:.1f}", CHARGE=CHARGE, PDGID=PDGID,
        STRINGFORCE=f"{lam ** 2 / HBARC_MEV_MM / 1e12:.6g}", LAMBDA_EV=f"{lam:g}",
        MAX_BOOST=f"{float(s['max_boost']):g}",
    )
    stage1 = "1_merge_smear.yaml" if mu == 0 else "1_pythia_pileup_merge.yaml"
    names = [stage1, "2_simulation.yaml", "3_digitization.yaml"]

    cfg_dir = work / "configs" / dataset
    run_dir = work / "runs" / dataset
    cfg_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        (cfg_dir / n).write_text((TEMPLATES / n).read_text().format(**values))

    env = dict(WORK_DIR=work, DATASET=dataset, MASS_GEV=f"{mass:g}", LAMBDA_EV=f"{lam:g}", MU=mu,
               EVENTS=events, SEED=seed, PDGID=PDGID, STAGES=" ".join(names))
    (cfg_dir / "sample.env").write_text("".join(f"{k}={shlex.quote(str(v))}\n" for k, v in env.items()))

    print(f"dataset   {dataset}  (string force {values['STRINGFORCE']} MeV/mm)")
    print(f"configs   {cfg_dir}")
    print(f"run dir   {run_dir}")
    print("next:")
    print(f"  bash {QUIRKML}/gen/run_quirkgen.sh {cfg_dir}")
    print(f"  bash {QUIRKML}/colliderml/run_sample.sh {cfg_dir}")


if __name__ == "__main__":
    main()
