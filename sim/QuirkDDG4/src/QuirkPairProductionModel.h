//==========================================================================
//  e+e- pair production model for the quirks, with its own sampling tables.
//
//  Geant4 11.3's G4MuPairProductionModel (base of G4hPairProductionModel) keeps
//  its sampling tables in a G4ElementData found by name in the global
//  G4ElementDataRegistry. The name is set only for e, mu, pi, K and p; for any
//  other particle it stays "", and GetElementDataByName("") returns the first
//  G4ElementData that was never given a name, which belongs to another part of
//  Geant4. The quirk model then never builds tables of its own, and the first
//  pair-production interaction on a quirk stops the run with em0033
//  (G4MuPairProductionModel::DataCorrupted). This model differs from
//  G4hPairProductionModel only in Initialise: the tables are built under a
//  name of their own ("quirkEEPairProd_<particle>"), on the master thread.
//==========================================================================
#ifndef QUIRKPAIRPRODUCTIONMODEL_H
#define QUIRKPAIRPRODUCTIONMODEL_H

#include <G4hPairProductionModel.hh>

class QuirkPairProductionModel : public G4hPairProductionModel {
public:
  explicit QuirkPairProductionModel(const G4ParticleDefinition* p);
  ~QuirkPairProductionModel() override = default;
  void Initialise(const G4ParticleDefinition*, const G4DataVector&) override;
};

#endif
