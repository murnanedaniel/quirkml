"""Write the ColliderML stage configs for one quirk sample from a settings file.

    python colliderml/make_sample.py my_sample.yaml

Writes <work_dir>/configs/<dataset>/ with the four stage configs and sample.env (the resolved
settings, read by gen/run_quirkgen.sh and colliderml/run_sample.sh), and creates the run
directory. Templates: colliderml/templates/. Stage 1 is merge_smear for mu = 0 and
pythia_generation with merging for mu > 0.
"""
import argparse
import os
import shlex
import sys
from pathlib import Path

import yaml

QUIRKML = Path(__file__).resolve().parent.parent
TEMPLATES = QUIRKML / "colliderml" / "templates"
HBARC_MEV_MM = 1.973269804e-10      # hbar c in MeV mm
PDGID, CHARGE = 10000100, 1.0       # quirk-gen: +10000100 has charge +1 (checked against events.json by run_sample.sh)
MOUNTED = ("/global/cfs/cdirs/m4958/",)


def expand(p):
    return os.path.expandvars(str(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("settings")
    a = ap.parse_args()
    s = yaml.safe_load(open(a.settings))

    work = Path(expand(s["work_dir"]))
    colliderml = work / "colliderml-production"
    lam, mass, mu, events, seed = float(s["lambda_ev"]), float(s["mass_gev"]), int(s["mu"]), int(s["events"]), int(s["seed"])
    dataset = f"m{mass:g}_L{lam:g}_mu{mu}"
    runs_base = work / "runs"
    run_dir = runs_base / s["campaign"] / dataset / str(s["version"]) / "runs" / "0"
    cfg_dir = work / "configs" / dataset

    scratch = os.environ.get("PSCRATCH", "")
    for name, p in (("work_dir", work), ("cache_dir", Path(expand(s["cache_dir"]))), ("quirkml", QUIRKML)):
        if not (str(p).startswith(MOUNTED) or (scratch and str(p).startswith(scratch))):
            sys.exit(f"{name} = {p} is not under /global/cfs/cdirs/m4958 or $PSCRATCH, so the container can't see it")
    if not (colliderml / "scripts" / "cli" / "run_stage.py").exists():
        sys.exit(f"no colliderml-production checkout at {colliderml}: run colliderml/setup.sh {work} first")

    values = dict(
        CAMPAIGN=s["campaign"], DATASET=dataset, VERSION=s["version"],
        CACHE_DIR=expand(s["cache_dir"]), RUNS_BASE=runs_base, RUN_DIR=run_dir,
        COLLIDERML=colliderml, QUIRKML=QUIRKML,
        EVENTS=events, MU=mu,
        SEED_GEN=seed + 41, SEED_SIM=seed + 42, SEED_DIGI=seed + 43,
        MASS_MEV=f"{mass * 1e3:.1f}", CHARGE=CHARGE, PDGID=PDGID,
        STRINGFORCE=f"{lam ** 2 / HBARC_MEV_MM / 1e12:.6g}", LAMBDA_EV=f"{lam:g}",
        MAX_BOOST=f"{float(s['max_boost']):g}",
    )
    stage1 = "1_merge_smear.yaml" if mu == 0 else "1_pythia_pileup_merge.yaml"
    names = [stage1, "2_simulation.yaml", "3_digitization.yaml", "4_package_parquet.yaml"]

    cfg_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        (cfg_dir / n).write_text((TEMPLATES / n).read_text().format(**values))

    env = dict(QUIRKML=QUIRKML, WORK_DIR=work, COLLIDERML=colliderml, CONFIG_DIR=cfg_dir, RUN_DIR=run_dir,
               DATASET=dataset, MASS_GEV=f"{mass:g}", LAMBDA_EV=f"{lam:g}", MU=mu, EVENTS=events, SEED=seed,
               PDGID=PDGID, QUIRKGEN_SCRATCH=expand(s["quirkgen_scratch"]), STAGES=" ".join(names))
    (cfg_dir / "sample.env").write_text("".join(f"{k}={shlex.quote(str(v))}\n" for k, v in env.items()))

    print(f"dataset   {dataset}  (string force {values['STRINGFORCE']} MeV/mm)")
    print(f"configs   {cfg_dir}")
    print(f"run dir   {run_dir}")
    print("next:")
    print(f"  bash {QUIRKML}/gen/run_quirkgen.sh {cfg_dir}")
    print(f"  bash {QUIRKML}/colliderml/run_sample.sh {cfg_dir}     # on a compute node, see docs/tutorial.md")


if __name__ == "__main__":
    main()
