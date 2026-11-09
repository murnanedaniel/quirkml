"""B1 strict fake decomposition (fakes/decompose.py, same categories) for mu 60 test predictions, normal truth.
Reference numbers from the Python strict scoring (py_score.py json) instead of an ACTS run.
  python decompose60.py OUT_DIR NAME=PRED_DIR:PYSCORE_JSON ...
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fakes"))
import common as C  # noqa: E402
import decompose as D  # noqa: E402

D.FIRST, D.STOP = 0, 2000
C.SRC = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/parquet_r1_mu60/m200_L774_mu60"
D.C.SRC = C.SRC
D.MODELS = {}
tmp = Path(sys.argv[1]) / "pyref"
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1)
    pdir, pj = rest.split(":")
    r = json.load(open(pj))["very_loose_strict"]
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({"track_classification_all_particles": {"very_loose_strict": r["tracks"]},
                                                "technical_sm_efficiency": {"very_loose_strict": r["technical_sm"]}}))
    D.MODELS[name] = (pdir, str(d))
sys.argv = [sys.argv[0], sys.argv[1]]
D.main()
