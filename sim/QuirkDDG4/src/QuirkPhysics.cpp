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
#include "QuirkPairProductionModel.h"

#include <G4Version.hh>
#include <G4ProcessManager.hh>
#include <G4RunManager.hh>
#include <G4EventManager.hh>
#include <G4Threading.hh>
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
        declareProperty("EnableBremsstrahlung",  m_enableBrem); // with EnableEM: G4hBremsstrahlung on the quirks
        declareProperty("EnablePairProduction",  m_enablePair); // with EnableEM: G4hPairProduction on the quirks
        declareProperty("PairProductionOwnTables", m_pairOwnTables); // pair production with QuirkPairProductionModel
      }
      virtual ~QuirkPhysics() = default;

      /// Create the quirk+ / quirk- particles (their strings are made per thread, see Quirk.h).
      virtual void constructParticle(Constructor& /* ctor */) override {
        info("+++ Quirk: m=%.1f MeV  q=%.1f e  pdg=%d  stringForce=%.3f MeV/mm",
             m_mass, m_charge, m_pdgid, m_stringForce);

        // quirk+ must be created first; the quirk- ctor links the two as
        // partners (see Quirk::Quirk).
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

        Quirk::StringParams s;
        s.stringForce = m_stringForce * CLHEP::MeV / CLHEP::mm;
        s.firstStringLength = m_firstStringLen * CLHEP::mm;
        s.maxBoost = m_maxBoost;
        s.maxMergeT = m_maxMergeT * CLHEP::mm;
        s.maxMergeMag = m_maxMergeMag * CLHEP::mm;
        q1->SetStringParams(s);
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
              // Brems and pair production can be left out (docs/colliderml.md); with both
              // on, the process list is the one of the earlier versions.
              int order = 3;
              if (m_enableBrem) { pmanager->AddProcess(new G4hBremsstrahlung, -1, order, order); ++order; }
              if (m_enablePair) {
                auto* pair = new G4hPairProduction;
                // Geant4 11.3's own model has no sampling tables for the quirks and stops the
                // run at their first pair production (em0033); see QuirkPairProductionModel.h.
                if (m_pairOwnTables) pair->SetEmModel(new QuirkPairProductionModel(particle));
                pmanager->AddProcess(pair, -1, order, order); ++order;
              }
              pmanager->AddProcess(new QuirkWatcher,          -1, -1, order);
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
        info("+++ Quirk: installed QuirkTransportation on %d quirk species (EM %s, brems %s, pair production %s)",
             nq, m_enableEM ? "on" : "off", (m_enableEM && m_enableBrem) ? "on" : "off",
             (m_enableEM && m_enablePair) ? (m_pairOwnTables ? "on, own tables" : "on") : "off");

        // The quirk and antiquirk must co-evolve their shared string: an extra
        // waiting stack + QuirkStackingAction force the partner to wait while
        // each quirk lays out string, mirroring QuirkPhysicsTool. Stacking is
        // per event, so in a multithreaded run it goes on every worker thread
        // (constructProcess runs on each of them); the master has no events.
        if (G4Threading::IsMultithreadedApplication() && G4Threading::IsMasterThread()) {
          info("+++ Quirk: master thread, QuirkStackingAction goes on the workers");
          return;
        }
        G4RunManager* rm = G4RunManager::GetRunManager();
        rm->SetNumberOfAdditionalWaitingStacks(1);
        G4UserStackingAction* dflt =
            G4EventManager::GetEventManager()->GetUserStackingAction();
        rm->SetUserAction(new QuirkStackingAction(dflt));
        info("+++ Quirk: installed QuirkStackingAction (+1 waiting stack) on thread %d%s",
             G4Threading::G4GetThreadId(), dflt ? ", wrapping the existing stacking action" : "");
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
      bool   m_enableBrem{true};    // with EnableEM true: hadron bremsstrahlung on the quirks
      bool   m_enablePair{true};    // with EnableEM true: e+e- pair production on the quirks
      bool   m_pairOwnTables{false}; // pair production with QuirkPairProductionModel (own sampling tables)
    };
  }
}

#include <DDG4/Factories.h>
using dd4hep::sim::QuirkPhysics;
DECLARE_GEANT4ACTION(QuirkPhysics)
