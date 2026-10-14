//==========================================================================
//  Diagnostic DDG4 stepping action: dump the per-step state of EVERY track
//  (quirks AND all secondaries: delta-ray electrons, photons, ...) so the
//  true stepped trajectories of all particles can be drawn.
//
//  Active only when the env var QUIRK_STEPDUMP_FILE is set.  Add it to a run
//  with:  SIM.action.step = [{"name": "Geant4StepDump/StepDump"}]
//==========================================================================
#include <DDG4/Geant4SteppingAction.h>
#include <DDG4/Factories.h>

#include <G4Step.hh>
#include <G4Track.hh>
#include <G4StepPoint.hh>
#include <G4SystemOfUnits.hh>

#include <fstream>
#include <cstdlib>

namespace quirks {

  class Geant4StepDump : public dd4hep::sim::Geant4SteppingAction {
  public:
    Geant4StepDump(dd4hep::sim::Geant4Context* ctxt, const std::string& nam)
      : dd4hep::sim::Geant4SteppingAction(ctxt, nam)
    {
      const char* f = std::getenv("QUIRK_STEPDUMP_FILE");
      if (f && f[0]) {
        m_out.open(f);
        m_out << "trackID,parentID,pdg,step,x_mm,y_mm,z_mm,p_MeV\n";
      }
    }
    virtual ~Geant4StepDump() { if (m_out.is_open()) m_out.close(); }

    virtual void operator()(const G4Step* step, G4SteppingManager*) override {
      if (!m_out.is_open()) return;
      const G4Track* t = step->GetTrack();
      const G4ThreeVector& x = step->GetPostStepPoint()->GetPosition();
      m_out << t->GetTrackID() << ',' << t->GetParentID() << ','
            << t->GetParticleDefinition()->GetPDGEncoding() << ','
            << t->GetCurrentStepNumber() << ','
            << x.x()/CLHEP::mm << ',' << x.y()/CLHEP::mm << ',' << x.z()/CLHEP::mm << ','
            << t->GetMomentum().mag()/CLHEP::MeV << '\n';
    }

  private:
    std::ofstream m_out;
  };
}

DECLARE_GEANT4ACTION_NS(quirks, Geant4StepDump)
