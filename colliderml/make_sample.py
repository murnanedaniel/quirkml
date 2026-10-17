"""Write the ColliderML stage configs for one quirk sample from a settings file.

    python3 colliderml/make_sample.py my_sample.yaml

Writes <work_dir>/configs/<dataset>/ with the stage configs and sample.env (the resolved
settings, read by colliderml/run_stage.sh), and creates the runs directory
<output_base_dir>/quirks/<dataset>/<version>/runs/ (ColliderML production's
<campaign>/<dataset>/<version>/runs/<N> layout, campaign "quirks"; output_base_dir defaults to work_dir). Templates:
colliderml/templates/. Stage 1 is merge_smear for mu = 0 and pythia_generation with merging for
mu > 0. The configs hold no seed: colliderml/run_stage.sh gives each run the seed
<dataset>_<version>_run<N>, as production's run_stage.py does. The configs use the paths inside
the container (/work = work_dir, /data = output_base_dir, /workspace = colliderml-production). Needs only python3.
"""
import argparse
import os
import shlex
import sys
from pathlib import Path

QUIRKML = Path(__file__).resolve().parent.parent
TEMPLATES = QUIRKML / "colliderml" / "templates"
HBARC_MEV_MM = 1.973269804e-10      # hbar c in MeV mm
PDGID, CHARGE = 10000100, 1.0       # quirk-gen: +10000100 has charge +1 (checked against events.json by run_stage.sh)
CAMPAIGN = "quirks"
KEYS = ("work_dir", "mass_gev", "lambda_ev", "mu", "runs", "events_per_run", "version",
        "sim_threads", "digi_threads", "chunk_size", "max_boost")


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

    def resolve(value):                             # relative paths are relative to the settings file
        path = Path(os.path.expanduser(os.path.expandvars(value)))
        return (path if path.is_absolute() else Path(a.settings).resolve().parent / path).resolve()
    work = resolve(s["work_dir"])
    out_base = resolve(s.get("output_base_dir") or s["work_dir"])
    lam, mass, mu = float(s["lambda_ev"]), float(s["mass_gev"]), int(s["mu"])
    runs, per_run, version = int(s["runs"]), int(s["events_per_run"]), s["version"]
    dataset = f"m{mass:g}_L{lam:g}_mu{mu}"
    if not (work / "colliderml-production" / "scripts").is_dir():
        sys.exit(f"no colliderml-production checkout in {work}: run colliderml/setup.sh {work} first")
    if runs < 1 or per_run < 1:
        sys.exit("runs and events_per_run must be at least 1")

    values = dict(
        CAMPAIGN=CAMPAIGN, DATASET=dataset, VERSION=version, EVENTS_PER_RUN=per_run, MU=mu,
        SIM_THREADS=int(s["sim_threads"]), DIGI_THREADS=int(s["digi_threads"]),
        CHUNK_SIZE=int(s["chunk_size"]),
        MASS_MEV=f"{mass * 1e3:.1f}", CHARGE=CHARGE, PDGID=PDGID,
        STRINGFORCE=f"{lam ** 2 / HBARC_MEV_MM / 1e12:.6g}", LAMBDA_EV=f"{lam:g}",
        MAX_BOOST=f"{float(s['max_boost']):g}",
    )
    stage1 = "1_merge_smear.yaml" if mu == 0 else "1_pythia_pileup_merge.yaml"
    names = [stage1, "2_simulation.yaml", "3_digitization.yaml", "4_package_parquet.yaml"]

    cfg_dir = work / "configs" / dataset
    runs_dir = out_base / CAMPAIGN / dataset / version / "runs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    if runs_dir.is_dir() and any(runs_dir.iterdir()):
        print(f"note: {runs_dir} already has runs; run_stage.sh keeps finished runs (it skips them)")
    runs_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        (cfg_dir / n).write_text((TEMPLATES / n).read_text().format(**values))

    env = dict(WORK_DIR=work, OUTPUT_BASE=out_base, CAMPAIGN=CAMPAIGN, DATASET=dataset, VERSION=version,
               RUNS_DIR=f"{CAMPAIGN}/{dataset}/{version}/runs", MASS_GEV=f"{mass:g}",
               LAMBDA_EV=f"{lam:g}", MU=mu, RUNS=runs, EVENTS_PER_RUN=per_run,
               SIM_THREADS=values["SIM_THREADS"], DIGI_THREADS=values["DIGI_THREADS"],
               PDGID=PDGID, STAGE1=stage1)
    (cfg_dir / "sample.env").write_text("".join(f"{k}={shlex.quote(str(v))}\n" for k, v in env.items()))

    print(f"dataset   {dataset}  (string force {values['STRINGFORCE']} MeV/mm)")
    print(f"size      {runs} runs x {per_run} events = {runs * per_run} events")
    print(f"configs   {cfg_dir}")
    print(f"runs      {runs_dir}/<N>")
    print("next, one stage at a time over all runs (add --parallel K to run K runs at once):")
    for stage in ("generation", "stage1", "simulation", "digitization", "package"):
        print(f"  bash {QUIRKML}/colliderml/run_stage.sh {cfg_dir} {stage}")

if __name__ == "__main__":
    main()
