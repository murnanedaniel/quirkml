/*
  Copyright (C) 2002-2024 CERN for the benefit of the ATLAS collaboration
*/

#ifndef QUIRK_H
#define QUIRK_H

#include "G4ParticleDefinition.hh"

class InfracolorForce;

class Quirk : public G4ParticleDefinition {
public:
    Quirk(
        const G4String&     aName,        G4double            mass,
        G4double            width,        G4double            charge,   
        G4int               iSpin,        G4int               iParity,   
        G4int               iConjugation, G4int               iIsospin,   
        G4int               iIsospin3,    G4int               gParity,
        const G4String&     pType,        G4int               lepton,     
        G4int               baryon,       G4int               encoding,
        G4bool              stable,       G4double            lifetime,
        G4DecayTable        *decaytable,  G4bool              shortlived,
        const G4String&     subType = "",
        G4int               anti_encoding = 0,
        G4double            magneticMoment = 0.0
    );
    ~Quirk();

    // String parameters, shared by the quirk and its antiquirk. Set once, before the run.
    struct StringParams {
        G4double stringForce;
        G4double firstStringLength;
        G4double maxBoost;
        G4double maxMergeT;
        G4double maxMergeMag;
    };
    void SetStringParams(const StringParams& params);

    // The string pulling on this quirk in the event being simulated by the calling thread.
    // Particle definitions are shared by all Geant4 worker threads, so the string state is
    // kept per thread: each thread holds its own linked pair for this quirk and its partner.
    InfracolorForce& GetStringIn() const;

    // Clear the calling thread's strings, so that an event never starts with string
    // left over from the previous event on that thread (called by QuirkStackingAction).
    static void ClearStrings();

    Quirk (const Quirk&) = delete;
    Quirk& operator= (const Quirk&) = delete;

private:
    Quirk* m_partner;
    StringParams m_params;
    G4bool m_hasParams;
};

#endif
