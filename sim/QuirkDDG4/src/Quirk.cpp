/*
  Copyright (C) 2002-2024 CERN for the benefit of the ATLAS collaboration
*/

#include <map>
#include <memory>

#include "G4ParticleTable.hh"
#include "InfracolorForce.h"
#include "Quirk.h"

Quirk::Quirk(
    const G4String&     aName,        G4double            mass,
    G4double            width,        G4double            charge,   
    G4int               iSpin,        G4int               iParity,   
    G4int               iConjugation, G4int               iIsospin,   
    G4int               iIsospin3,    G4int               gParity,
    const G4String&     pType,        G4int               lepton,     
    G4int               baryon,       G4int               encoding,
    G4bool              stable,       G4double            lifetime,
    G4DecayTable        *decaytable,  G4bool              shortlived,
    const G4String&     subType,
    G4int               anti_encoding,
    G4double            magneticMoment
) :
    G4ParticleDefinition(
	aName,mass,width,charge,iSpin,iParity,
        iConjugation,iIsospin,iIsospin3,gParity,pType,
        lepton,baryon,encoding,stable,lifetime,decaytable,
        shortlived,subType,anti_encoding,magneticMoment
    ),
    m_partner(0),
    m_params(),
    m_hasParams(false)
{
    G4ParticleTable* particleTable = GetParticleTable();
    G4ParticleDefinition* partner = particleTable->FindParticle(GetAntiPDGEncoding());
    if (partner != 0) {
        Quirk* qpartner = dynamic_cast<Quirk*>(partner);
        if (qpartner == 0) {
            G4Exception("Quirk::Quirk", "NonQuirkAntiparticle", FatalErrorInArgument, "Antiparticle of quirk must be a quirk");
            std::abort();
        }
        m_partner = qpartner;
        qpartner->m_partner = this;
    }
}

Quirk::~Quirk() {}

void Quirk::SetStringParams(const StringParams& params) {
    m_params = params;
    m_hasParams = true;
    if (m_partner != 0) {
        m_partner->m_params = params;
        m_partner->m_hasParams = true;
    }
}

namespace {
    // One linked pair of strings per thread and quirk pair, created on first use in that thread.
    typedef std::map<const Quirk*, std::unique_ptr<InfracolorForce> > StringMap;
    G4ThreadLocal StringMap* strings = 0;
}

void Quirk::ClearStrings() {
    if (strings == 0) return;
    for (StringMap::iterator it = strings->begin(); it != strings->end(); ++it) it->second->Clear();
}

InfracolorForce& Quirk::GetStringIn() const {
    if (strings == 0) strings = new StringMap();
    StringMap::iterator it = strings->find(this);
    if (it != strings->end()) return *it->second;
    if (m_partner == 0) {
        G4Exception("Quirk::GetStringIn", "NoAntiQuirk", FatalErrorInArgument, "Quirk has no antiquirk");
        std::abort();
    }
    InfracolorForce* mine = new InfracolorForce();
    InfracolorForce* theirs = new InfracolorForce();
    mine->SetReactionForce(theirs);
    theirs->SetReactionForce(mine);
    if (m_hasParams) {   // the setters apply to both ends of the string
        mine->SetStringForce(m_params.stringForce);
        mine->SetFirstStringLength(m_params.firstStringLength);
        mine->SetMaxBoost(m_params.maxBoost);
        mine->SetMaxMergeT(m_params.maxMergeT);
        mine->SetMaxMergeMag(m_params.maxMergeMag);
    }
    (*strings)[this].reset(mine);
    (*strings)[m_partner].reset(theirs);
    return *mine;
}
