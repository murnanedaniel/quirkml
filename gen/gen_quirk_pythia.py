#!/usr/bin/env python3
"""Quirk-pair events with full SM ISR + MPI, Pythia8 Hidden Valley, HepMC3 out.

Process: HiddenValley:ffbar2EvEvbar (q qbar -> gamma*/Z* -> Ev Evbar, code 4931),
Ev = colourless spin-1/2 fermion with charge -1, stable, no HV shower or
fragmentation.  In the output Ev -> -QPDG and Evbar -> +QPDG (charge kept), so
QuirkDDG4 (pdg 4900101, quirk+ = +pdg) picks the pair up.

All status-1 particles are written, grouped by production vertex; each vertex
gets a fictitious status-2 mother so every vertex has an incoming particle.
Units GEV, MM.  Vertex smearing as ColliderML hard_scatter (xy 12.5 um, z 55.5 mm,
t 0.185 ns).  A truth sidecar (.npz) holds the quirk 4-momenta at production,
the primary vertex, and the recoil = sum over all other final-state particles.

  python gen_quirk_pythia.py --mass 400 --nevents 2000 --seed 1 --out q.hepmc3
"""
import argparse
import numpy as np
import pythia8

EV = 4900011
MOTHER = 4900111  # fictitious neutral mother, status 2 (not tracked)
C_MM_PER_NS = 299.792458


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, required=True, help="GeV")
    ap.add_argument("--nevents", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ecm", type=float, default=14000.0)
    ap.add_argument("--qpdg", type=int, default=4900101)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-smear", action="store_true")
    a = ap.parse_args()

    p = pythia8.Pythia("", False)
    for s in [
        "Beams:eCM = %g" % a.ecm,
        "HiddenValley:ffbar2EvEvbar = on",
        "HiddenValley:spinFv = 1",          # spin 1/2
        "HiddenValley:FSR = off",
        "HiddenValley:fragment = off",
        "%d:m0 = %g" % (EV, a.mass),
        "%d:mWidth = 0" % EV,
        "%d:mayDecay = off" % EV,
        "PartonLevel:ISR = on",
        "PartonLevel:MPI = on",
        "HadronLevel:all = on",
        "ParticleDecays:limitTau0 = on",    # as ColliderML: Geant4 decays the rest
        "ParticleDecays:tau0Max = 10",
        "Random:setSeed = on",
        "Random:seed = %d" % a.seed,
        "Next:numberCount = 0",
        "Next:numberShowEvent = 0",
        "Next:numberShowInfo = 0",
        "Next:numberShowProcess = 0",
    ]:
        p.readString(s)
    p.init()
    rng = np.random.default_rng(a.seed + 7919)

    truth = {k: [] for k in ["qplus", "qminus", "vtx", "recoil", "nfinal", "sigma_mb"]}
    nw = 0
    with open(a.out, "w") as f:
        f.write("HepMC::Version 3.02.06\nHepMC::Asciiv3-START_EVENT_LISTING\n")
        while nw < a.nevents:
            if not p.next():
                continue
            ev = p.event
            if a.no_smear:
                v0 = np.zeros(4)
            else:
                v0 = np.array([rng.normal(0, 0.0125), rng.normal(0, 0.0125),
                               rng.normal(0, 55.5), rng.normal(0, 0.185) * C_MM_PER_NS])
            groups, qp, qm, rec, nfin = {}, None, None, np.zeros(4), 0
            for i in range(ev.size()):
                pa = ev[i]
                if not pa.isFinal():
                    continue
                pid = pa.id()
                mom = np.array([pa.px(), pa.py(), pa.pz(), pa.e()])
                if abs(pid) == EV:
                    pid = -a.qpdg if pid > 0 else a.qpdg
                    if pid > 0:
                        qp = mom
                    else:
                        qm = mom
                else:
                    rec += mom
                nfin += 1
                # production vertex: Pythia mm (and mm/c for time) + smeared PV
                x = (round(pa.xProd() + v0[0], 9), round(pa.yProd() + v0[1], 9),
                     round(pa.zProd() + v0[2], 9), round(pa.tProd() + v0[3], 9))
                groups.setdefault(x, []).append((pid, mom, pa.m()))
            if qp is None or qm is None:
                continue
            npart = sum(len(g) + 1 for g in groups.values())
            f.write(f"E {nw} {len(groups)} {npart}\nU GEV MM\n")
            pidx, vidx = 0, 0
            for x, plist in groups.items():
                tot = sum(m for _, m, _ in plist)
                mm = np.sqrt(max(tot[3] ** 2 - tot[0] ** 2 - tot[1] ** 2 - tot[2] ** 2, 0.0))
                pidx += 1; mother = pidx; vidx -= 1
                f.write(f"P {mother} 0 {MOTHER} {tot[0]:.9e} {tot[1]:.9e} {tot[2]:.9e} {tot[3]:.9e} {mm:.9e} 2\n")
                f.write(f"V {vidx} 0 [{mother}] @ {x[0]:.9e} {x[1]:.9e} {x[2]:.9e} {x[3]:.9e}\n")
                for pid, m4, mass in plist:
                    pidx += 1
                    f.write(f"P {pidx} {vidx} {pid} {m4[0]:.9e} {m4[1]:.9e} {m4[2]:.9e} {m4[3]:.9e} {mass:.9e} 1\n")
            truth["qplus"].append(qp); truth["qminus"].append(qm)
            truth["vtx"].append(v0); truth["recoil"].append(rec)
            truth["nfinal"].append(nfin); truth["sigma_mb"].append(p.infoPython().sigmaGen())
            nw += 1
        f.write("HepMC::Asciiv3-END_EVENT_LISTING\n")
    np.savez(a.out.replace(".hepmc3", "_truth.npz"), mass_GeV=a.mass, seed=a.seed,
             **{k: np.array(v) for k, v in truth.items()})
    print(f"wrote {a.out}: {nw} events, sigma = {truth['sigma_mb'][-1]*1e9:.4f} pb")


if __name__ == "__main__":
    main()
