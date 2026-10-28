"""ROC AUC of the quirk GNN edge scores, from the score histograms written by edge_eval.py (raw/*.npz, 1000 bins).

AUC over the candidate edges in each model's graph (true edges missing from the graph are not included).
Positives = the model's training-target edges; negatives = edges joining two different particles (for pair GNNs the
quirk pair is one particle). Same-particle edges that are not training targets (zero loss weight) are left out.
Three AUCs: all edges; edges with at least one quirk hit; quirk separation (per-quirk GNNs: same-quirk target edges
against edges joining quirk A and quirk B; pair GNNs: pair target edges against edges joining a quirk and another
particle). Ties inside a bin count one half.
    python tracking/gnn/edge_auc.py [split: test|val200|val800]
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_eval as E  # noqa: E402

RAW = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/results/edge_eval/raw"
OUT = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/results/edge_eval/auc.json"


def auc(pos, neg):
    """pos, neg: histograms over score bins (ascending). P(score_pos > score_neg) + 0.5 P(equal bin)."""
    cneg = np.cumsum(neg) - neg  # negatives in lower bins
    return float((pos * (cneg + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    res = {}
    for model, spec in E.MODELS.items():
        kind = spec[1]
        H = sum(np.load(f)[f"H_{split}"][0] for f in sorted(glob.glob(os.path.join(RAW, f"{model}_*.npz"))))
        tgt, same = set(E.true_classes(kind, "target")), set(E.true_classes(kind, "same"))
        g = lambda cs: sum(H[E.CI[c]] for c in cs)  # noqa: E731
        out = {}
        for name, sel in (("all", E.CLASSES), ("quirk_hit", E.QTOUCH)):
            out[name] = auc(g([c for c in sel if c in tgt]), g([c for c in sel if c not in same]))
        if kind == "perquirk":
            out["quirk_separation"] = auc(g(["q_chain_pc", "q_chain_npc"]), g(["cross_pc", "cross_npc"]))
        else:
            out["quirk_separation"] = auc(g([c for c in E.true_classes(kind, "target") if c in E.QTOUCH]),
                                          g(["quirk_ord"]))
        res[model] = out
        print(model, json.dumps(out))
    json.dump({"split": split, "definition": __doc__, "auc": res}, open(OUT.replace(".json", f"_{split}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
