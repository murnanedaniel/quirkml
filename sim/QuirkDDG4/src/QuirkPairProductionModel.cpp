#include "QuirkPairProductionModel.h"

#include <G4AutoLock.hh>
#include <G4ElementData.hh>
#include <G4ElementDataRegistry.hh>
#include <G4Log.hh>
#include <G4ParticleDefinition.hh>
#include <cmath>

namespace { G4Mutex quirkPairMutex = G4MUTEX_INITIALIZER; }

QuirkPairProductionModel::QuirkPairProductionModel(const G4ParticleDefinition* p)
  : G4hPairProductionModel(p, "hPairProd") {}

// Same steps as G4MuPairProductionModel::Initialise (Geant4 11.3), except that the
// sampling tables are looked up and stored under the quirk's own name.
void QuirkPairProductionModel::Initialise(const G4ParticleDefinition* p, const G4DataVector& cuts) {
  SetParticle(p);
  if (nullptr == fParticleChange) {
    fParticleChange = GetParticleChangeForLoss();
    if (0 == nbine) {
      emin = std::max(lowestKinEnergy, LowEnergyLimit());
      emax = std::max(HighEnergyLimit(), emin * 2);
      nbine = std::size_t(nYBinPerDecade * std::log10(emax / emin));
      if (nbine < 3) { nbine = 3; }
      ymin = G4Log(minPairEnergy / emin);
      dy = -ymin / G4double(nbiny);
    }
  }
  if (lowestKinEnergy >= HighEnergyLimit()) { return; }
  if (p != particle) { return; }

  const G4String name = "quirkEEPairProd_" + p->GetParticleName();
  G4AutoLock l(&quirkPairMutex);
  fElementData = G4ElementDataRegistry::Instance()->GetElementDataByName(name);
  if (nullptr == fElementData) {
    // Normally on the master, before the workers start. G4ElementData registers
    // itself with an empty name, so it is named at once.
    fElementData = new G4ElementData(NZDATPAIR);
    fElementData->SetName(name);
    MakeSamplingTables();
  }
  l.unlock();
  if (IsMaster()) { InitialiseElementSelectors(p, cuts); }
}
