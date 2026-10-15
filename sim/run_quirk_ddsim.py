#!/usr/bin/env python3
"""Self-contained ddsim driver that runs the QuirkDDG4 plugin.

Injects the QuirkPhysics constructor via SIM.physics.setupUserPhysics, runs
one thread (so that the QUIRK_* diagnostic files below are written by one
thread; the plugin itself runs multi-threaded in the ColliderML route), and
writes edm4hep output.  Works on any dd4hep compact geometry (the
simplified barrel in ../geometry, or the full ODD).

Build & path setup first:  source ../build.sh  (or env.sh) so the plugin's
libQuirkDDG4.so + .components are on DD4HEP_LIBRARY_PATH.

Example:
  ddsim is not used directly; run this with the LCG/dd4hep python:
    python run_quirk_ddsim.py \
        --compact ../geometry/QuirkBarrel.xml \
        --input quirks.hepmc3 --output quirks_edm4hep.root \
        --pdg 4900101 --mass 1000 --string-force 5068
"""
import os
import sys
import argparse
from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import MeV, GeV, mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compact", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="quirks_edm4hep.root")
    ap.add_argument("--events", type=int, default=1)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--pdg", type=int, default=4900101)
    ap.add_argument("--mass", type=float, default=1000.0, help="GeV")
    ap.add_argument("--charge", type=float, default=1.0)
    ap.add_argument("--string-force", type=float, default=5068.0,
                    help="MeV/mm (= Lambda^2/hbarc; Lambda=1000 eV -> 5068)")
    ap.add_argument("--first-string", type=float, default=1e-6, help="mm")
    ap.add_argument("--max-boost", type=float, default=1e-2)
    ap.add_argument("--no-em", action="store_true",
                    help="disable multiple scattering, bremsstrahlung and pair production on the "
                         "quirks; ionisation stays on (it makes the hits) unless --no-ionisation")
    ap.add_argument("--no-quirk-physics", action="store_true", help="diagnostic: plain ddsim")
    ap.add_argument("--no-ionisation", action="store_true",
                    help="with --no-em: also drop ionisation (pure string transport for validation; "
                         "no energy deposits, so no tracker hits)")
    a = ap.parse_args()

    # DD4hepSimulation.run() re-parses sys.argv; strip our args so it does not
    # mistake them for a compact file, and use absolute paths.
    compact = os.path.abspath(a.compact)
    inp = os.path.abspath(a.input)
    outp = os.path.abspath(a.output)
    sys.argv = [sys.argv[0]]

    SIM = DD4hepSimulation()
    SIM.compactFile = [compact]      # run() iterates this as a list
    SIM.inputFiles = [inp]
    SIM.outputFile = outp
    SIM.numberOfEvents = a.events
    SIM.skipNEvents = a.skip
    SIM.runType = "batch"
    SIM.numberOfThreads = 1           # REQUIRED: shared quirk string state

    SIM.part.minimalKineticEnergy = 1.0 * MeV
    # Use the full-truth particle handler so that secondaries (delta-ray
    # electrons) which leave a tracker hit are KEPT as distinct MCParticles and
    # their hits stay linked to THEM -- not silently reassigned to the quirk
    # parent (which the default Geant4TCUserParticleHandler does).  The handler
    # reads tracker_region_rmax/zmax constants from the compact.
    SIM.part.userParticleHandler = "Geant4FullTruthParticleHandler"
    SIM.part.keepAllParticles = False
    # the exotic quirk pdg (default 4900101) is not in the default
    # reject/zeroTime PDG sets, so leave those at their defaults.
    SIM.physics.list = "FTFP_BERT"

    pdg = a.pdg
    mass_mev = a.mass * GeV / MeV    # value in MeV
    charge = a.charge
    sf = a.string_force
    fs = a.first_string
    mb = a.max_boost
    no_em = a.no_em

    def setupQuirks(kernel):
        from DDG4 import PhysicsList
        seq = kernel.physicsList()
        q = PhysicsList(kernel, "QuirkPhysics/Quirks")
        q.Mass = float(mass_mev)         # MeV
        q.Charge = float(charge)
        q.PdgId = int(pdg)
        q.StringForce = float(sf)        # MeV/mm
        q.FirstStringLen = float(fs)     # mm
        q.MaxBoost = float(mb)
        q.MaxMergeT = 1e-6
        q.MaxMergeMag = 1e-6
        q.EnableEM = not no_em
        q.EnableIonisation = not a.no_ionisation
        q.enableUI()
        seq.adopt(q)
        return None

    if not a.no_quirk_physics:
        SIM.physics.setupUserPhysics(setupQuirks)

    # Optional: dump every step of every track (quirks + all secondaries) for
    # trajectory diagnostics.  Active only when QUIRK_STEPDUMP_FILE is set.
    if os.environ.get("QUIRK_STEPDUMP_FILE"):
        SIM.action.step = [{"name": "Geant4StepDump/StepDump"}]

    print(f"[run_quirk_ddsim] compact={a.compact}")
    print(f"[run_quirk_ddsim] quirk pdg=+-{pdg} mass={a.mass} GeV "
          f"stringForce={sf} MeV/mm  threads=1")
    SIM.run()


if __name__ == "__main__":
    main()
