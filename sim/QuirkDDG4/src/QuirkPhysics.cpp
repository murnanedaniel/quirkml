//==========================================================================
//  DDG4 physics constructor for the ATLAS-style infracolor-string quirk.
//
//  This replaces the Athena/Gaudi QuirkPhysicsTool with a dd4hep::sim
//  Geant4PhysicsConstructor plugin so the *exact* HyperbolaStepper /
//  QuirkTransportation generator can run inside ddsim / DDG4.
//
//  Modelled on DDG4/plugins/Geant4ExtraParticles.cpp.  The quirk parameters
//  (mass, charge, PDG id, string tension, ...) are taken from DDG4 properties
//  instead of the Athena quirks_setup.txt file.  The two Quirk particles are
//  created in constructParticle(); QuirkTransportation + the EM processes +
//  QuirkWatcher are installed in constructProcess(), which also installs the
//  QuirkStackingAction and the extra waiting stack that make the quirk and
//  antiquirk co-evolve their shared string (exactly as QuirkPhysicsTool did).
//==========================================================================

#include <DDG4/Geant4PhysicsConstructor.h>
#include <DDG4/Factories.h>

#include "Quirk.h"
#include "InfracolorForce.h"
#include "QuirkTransportation.h"
#include "QuirkWatcher.h"
#include "QuirkStackingAction.h"

#include <G4Version.hh>
#include <G4ProcessManager.hh>
#include <G4RunManager.hh>
#include <G4EventManager.hh>
#include <G4hMultipleScattering.hh>
#include <G4hIonisation.hh>
#include <G4hBremsstrahlung.hh>
#include <G4hPairProduction.hh>
#include <G4SystemOfUnits.hh>

namespace dd4hep {
  namespace sim {

    /// DDG4 physics constructor that adds the quirk pair and its dynamics.
    class QuirkPhysics : public Geant4PhysicsConstructor {
    public:
      QuirkPhysics() = delete;
      QuirkPhysics(const QuirkPhysics&) = delete;
      QuirkPhysics(Geant4Context* ctxt, const std::string& nam)
        : Geant4PhysicsConstructor(ctxt, nam)
      {
        declareProperty("Mass",           m_mass);            // MeV
        declareProperty("Charge",         m_charge);          // e
        declareProperty("PdgId",          m_pdgid);
        declareProperty("StringForce",    m_stringForce);     // MeV/mm (= Lambda^2/hbarc)
        declareProperty("FirstStringLen", m_firstStringLen);  // mm
        declareProperty("MaxBoost",       m_maxBoost);
        declareProperty("MaxMergeT",      m_maxMergeT);        // mm
        declareProperty("MaxMergeMag",    m_maxMergeMag);      // mm
        declareProperty("EnableEM",       m_enableEM);         // EM processes on quirks
        declareProperty("EnableIonisation", m_enableIoni);     // ionisation when EnableEM is false (needed for hits)
      }
      virtual ~QuirkPhysics() = default;

      /// Create the quirk+ / quirk- particles with their cross-linked strings.
      virtual void constructParticle(Constructor& /* ctor */) override {
        info("+++ Quirk: m=%.1f MeV  q=%.1f e  pdg=%d  stringForce=%.3f MeV/mm",
             m_mass, m_charge, m_pdgid, m_stringForce);

        // quirk+ must be created first; the quirk- ctor cross-links the two
        // InfracolorForce objects (see Quirk::Quirk).
        Quirk* q1 = new Quirk(
            "quirk+", m_mass * CLHEP::MeV, /*width*/ 0.0, +m_charge * CLHEP::eplus,
            /*2*spin*/ 1, /*parity*/ 0, /*C*/ 0, /*2*I*/ 0, /*2*I3*/ 0, /*G*/ 0,
            /*type*/ "lepton", /*lepton*/ +1, /*baryon*/ 0, +m_pdgid,
            /*stable*/ true, /*lifetime*/ -1.0, /*decay*/ nullptr, /*shortlived*/ false);
        new Quirk(
            "quirk-", m_mass * CLHEP::MeV, /*width*/ 0.0, -m_charge * CLHEP::eplus,
            /*2*spin*/ 1, /*parity*/ 0, /*C*/ 0, /*2*I*/ 0, /*2*I3*/ 0, /*G*/ 0,
            /*type*/ "lepton", /*lepton*/ -1, /*baryon*/ 0, -m_pdgid,
            /*stable*/ true, /*lifetime*/ -1.0, /*decay*/ nullptr, /*shortlived*/ false);

        InfracolorForce& s = q1->GetStringIn();
        s.SetStringForce(m_stringForce * CLHEP::MeV / CLHEP::mm);
        s.SetFirstStringLength(m_firstStringLen * CLHEP::mm);
        s.SetMaxBoost(m_maxBoost);
        s.SetMaxMergeT(m_maxMergeT * CLHEP::mm);
        s.SetMaxMergeMag(m_maxMergeMag * CLHEP::mm);
      }

      /// Replace transportation for the quirks and wire up the stacking.
      virtual void constructProcess(Constructor& ctor) override {
        G4ParticleTable::G4PTblDicIterator* it = ctor.particleIterator();
        it->reset();
        int nq = 0;
        while ((*it)()) {
          G4ParticleDefinition* particle = it->value();
          if (dynamic_cast<Quirk*>(particle) != nullptr) {
            ++nq;
            G4ProcessManager* pmanager = particle->GetProcessManager();
            while (pmanager->GetProcessListLength() != 0) pmanager->RemoveProcess(0);
            G4VProcess* aTransportation = new QuirkTransportation;
            pmanager->AddProcess(aTransportation);
            pmanager->SetProcessOrderingToFirst(aTransportation, idxAlongStep);
            pmanager->SetProcessOrderingToFirst(aTransportation, idxPostStep);
            if (m_enableEM) {
              pmanager->AddProcess(new G4hMultipleScattering, -1, 1, 1);
              pmanager->AddProcess(new G4hIonisation,         -1, 2, 2);
              pmanager->AddProcess(new G4hBremsstrahlung,     -1, 3, 3);
              pmanager->AddProcess(new G4hPairProduction,     -1, 4, 4);
              pmanager->AddProcess(new QuirkWatcher,          -1, -1, 5);
            } else {
              // diagnostic: keep ionisation (required to create the
              // SimTrackerHits) but drop multiple scattering + radiative
              // processes, isolating whether the transverse braid scatter is
              // multiple scattering or the transport/stepper itself.
              if (m_enableIoni) {
                pmanager->AddProcess(new G4hIonisation,       -1, 1, 1);
                pmanager->AddProcess(new QuirkWatcher,        -1, -1, 2);
              } else {
                // pure string transport (validation only): no energy loss, so no tracker hits
                pmanager->AddProcess(new QuirkWatcher,        -1, -1, 1);
              }
            }
          }
        }
        info("+++ Quirk: installed QuirkTransportation on %d quirk species", nq);

        // The quirk and antiquirk must co-evolve their shared string: an extra
        // waiting stack + QuirkStackingAction force the partner to wait while
        // each quirk lays out string, mirroring QuirkPhysicsTool.
        G4RunManager* rm = G4RunManager::GetRunManager();
        rm->SetNumberOfAdditionalWaitingStacks(1);
        G4UserStackingAction* dflt =
            G4EventManager::GetEventManager()->GetUserStackingAction();
        rm->SetUserAction(new QuirkStackingAction(dflt));
        info("+++ Quirk: installed QuirkStackingAction (+1 waiting stack); "
             "run ddsim single-threaded (threads=1).");
      }

    private:
      double m_mass{500000.0};      // MeV  (default 500 GeV)
      double m_charge{1.0};         // e
      int    m_pdgid{4900101};      // exotic id (not 15: tau is in zeroTimePDGs)
      double m_stringForce{5068.0}; // MeV/mm (Lambda=1000 eV -> 5068)
      double m_firstStringLen{1.0e-6};
      double m_maxBoost{1.0e-2};
      double m_maxMergeT{1.0e-6};
      double m_maxMergeMag{1.0e-6};
      bool   m_enableEM{true};      // ionisation/MS/brems/pair on the quirks
      bool   m_enableIoni{true};    // with EnableEM false: keep ionisation (true) or none at all
    };
  }
}

#include <DDG4/Factories.h>
using dd4hep::sim::QuirkPhysics;
DECLARE_GEANT4ACTION(QuirkPhysics)
