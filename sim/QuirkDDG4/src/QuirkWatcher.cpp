/*
  Copyright (C) 2002-2024 CERN for the benefit of the ATLAS collaboration
*/

#include "G4TrackStatus.hh"
#include "G4LorentzVector.hh"
#include "G4SystemOfUnits.hh"

#include "Quirk.h"
#include "InfracolorForce.h"
#include "QuirkWatcher.h"

// Diagnostics (all env-controlled, no-op unless the variable is set):
//   QUIRK_WATCH_DEBUG=1   -> log suspend/kill hand-off decisions (capped)
//   QUIRK_DUMP_FILE=path  -> append the FULL per-step quirk state (every step)
//   QUIRK_ZMAX_CM=value   -> stop tracking a quirk once |z| exceeds this (so
//                            accuracy sweeps don't coast the pair out to 40 m)
#include <cstdlib>
#include <fstream>
namespace {
  int qwatch_dbg = -1;
  long qwatch_count = 0;
  bool qwatchOn() {
    if (qwatch_dbg < 0) { const char* e = std::getenv("QUIRK_WATCH_DEBUG");
                          qwatch_dbg = (e && e[0] == '1') ? 1 : 0; }
    return qwatch_dbg == 1;
  }
  std::ofstream* qdump = nullptr;
  int qdump_init = 0;
  std::ofstream* dumpStream() {
    if (!qdump_init) {
      qdump_init = 1;
      const char* f = std::getenv("QUIRK_DUMP_FILE");
      if (f && f[0]) {
        qdump = new std::ofstream(f);
        (*qdump) << "trackID,pdg,step,t_ns,x_mm,y_mm,z_mm,px_MeV,py_MeV,pz_MeV,E_MeV\n";
      }
    }
    return qdump;
  }
  double zmaxMM = -2;
  double zmaxCut() {  // mm, or <0 if disabled
    if (zmaxMM < -1) { const char* e = std::getenv("QUIRK_ZMAX_CM");
                       zmaxMM = (e && e[0]) ? std::atof(e) * CLHEP::cm : -1.0; }
    return zmaxMM;
  }
  double rmaxMM = -2;
  double rmaxCut() {  // mm, or <0 if disabled
    if (rmaxMM < -1) { const char* e = std::getenv("QUIRK_RMAX_CM");
                       rmaxMM = (e && e[0]) ? std::atof(e) * CLHEP::cm : -1.0; }
    return rmaxMM;
  }
}


QuirkWatcher::QuirkWatcher() : G4VProcess(G4String("QuirkWatcher")) {
    enableAtRestDoIt = false;
    enableAlongStepDoIt = false;
}

QuirkWatcher::~QuirkWatcher() {}

G4double QuirkWatcher::PostStepGetPhysicalInteractionLength(
    const G4Track&, //track
    G4double, //previousStepSize
    G4ForceCondition* condition
) {
    *condition = StronglyForced;
    return DBL_MAX;
}

G4VParticleChange* QuirkWatcher::PostStepDoIt(
    const G4Track& track,
    const G4Step& //stepData
) {
    // Get infracolor string
    auto part_nc = // track should really be non-const in Geant4 interface
      const_cast<G4ParticleDefinition*>(track.GetParticleDefinition());
    Quirk* quirkDef = dynamic_cast<Quirk*>(part_nc);
    if (quirkDef == 0) {
        G4Exception("QuirkWatcher::PostStepDoIt", "NonQuirk", FatalErrorInArgument, "QuirkWatcher run on non-quirk particle");
        std::abort();
    }
    InfracolorForce& string = quirkDef->GetStringIn();

    // Full per-step state dump (every step) for the error-accumulation analysis.
    if (std::ofstream* d = dumpStream()) {
        const G4ThreeVector& x = track.GetPosition();
        const G4ThreeVector& p = track.GetMomentum();
        (*d) << track.GetTrackID() << ',' << track.GetParticleDefinition()->GetPDGEncoding()
             << ',' << track.GetCurrentStepNumber()
             << ',' << track.GetGlobalTime()/CLHEP::ns
             << ',' << x.x()/CLHEP::mm << ',' << x.y()/CLHEP::mm << ',' << x.z()/CLHEP::mm
             << ',' << p.x()/CLHEP::MeV << ',' << p.y()/CLHEP::MeV << ',' << p.z()/CLHEP::MeV
             << ',' << track.GetTotalEnergy()/CLHEP::MeV << '\n';
    }

    if (track.GetCurrentStepNumber() > 1 && !string.IsSourceInitialized()) {
        string.Clear();
        string.GetReactionForce()->Clear();
        G4Exception(
            "QuirkWatcher::PostStepDoIt",
            "QuirkMissingPartner",
            EventMustBeAborted,
            "QuirkWatcher: missing partner for quirk"
        );
    }

    // Update track status
    G4TrackStatus stat = track.GetTrackStatus();
    if (stat == fStopButAlive) {
        stat = fAlive;
    }
    if (stat == fAlive || stat == fSuspend) {
        G4bool passControl = !string.HasNextStringVector();
        //G4bool passControl = !string.HasNextStringVector() || (string.GetSumStrings().t() < string.GetReactionForce()->GetSumStrings().t());
        if (passControl) {
            if (string.IsSourceAlive()) {
                stat = fSuspend;
            } else {
                stat = fStopAndKill;
            }
            // Always log KILL (rare, important); cap SUSPEND spam.
            bool isKill = (stat != fSuspend);
            if (qwatchOn() && (isKill || qwatch_count < 50)) {
                if (!isKill) ++qwatch_count;
                G4cout << "QWATCH " << track.GetParticleDefinition()->GetParticleName()
                       << " trk=" << track.GetTrackID()
                       << " step=" << track.GetCurrentStepNumber()
                       << " r=" << track.GetPosition().perp()/CLHEP::cm << "cm"
                       << " z=" << track.GetPosition().z()/CLHEP::cm << "cm"
                       << " nStrThis=" << string.GetNStrings()
                       << " nStrPartner=" << string.GetReactionForce()->GetNStrings()
                       << " srcAlive=" << string.IsSourceAlive()
                       << " -> " << (stat==fSuspend ? "SUSPEND" : "KILL") << G4endl;
            }
        }
    }
    // Region cut: once the quirk is well past the tracker in z, stop tracking it
    // (avoids coasting the pair out to the 40 m world edge).
    double zc = zmaxCut();
    if (zc > 0 && std::abs(track.GetPosition().z()) > zc) {
        stat = fStopAndKill;
    }
    double rc = rmaxCut();
    if (rc > 0 && track.GetPosition().perp() > rc) {
        stat = fStopAndKill;
    }

    if (stat == fStopAndKill || stat == fKillTrackAndSecondaries) {
        string.TrackKilled();
    }

    m_particleChange.Initialize(track);
    m_particleChange.ProposeTrackStatus(stat);
    return &m_particleChange;
}
