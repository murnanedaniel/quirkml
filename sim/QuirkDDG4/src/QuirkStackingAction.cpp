/*
  Copyright (C) 2002-2017 CERN for the benefit of the ATLAS collaboration
*/

#include "QuirkStackingAction.h"

#include "Quirk.h"

#include "G4Track.hh"
#include "G4Threading.hh"
#include "G4ios.hh"

QuirkStackingAction::QuirkStackingAction(G4UserStackingAction* defaultStackingAction):
    m_firstTrack(true),
    m_defaultStackingAction(defaultStackingAction)
{}

QuirkStackingAction::~QuirkStackingAction() {
    delete m_defaultStackingAction;
}

G4ClassificationOfNewTrack QuirkStackingAction::ClassifyNewTrack(const G4Track* aTrack) {
    G4ClassificationOfNewTrack defaultClassification = fUrgent;
    if (m_defaultStackingAction) {
        m_defaultStackingAction->SetStackManager(stackManager);
        defaultClassification = m_defaultStackingAction->ClassifyNewTrack(aTrack);
    }
    const Quirk* quirkDef = dynamic_cast<const Quirk*>(aTrack->GetParticleDefinition());
    if (quirkDef == 0 || (defaultClassification != fUrgent && defaultClassification != fWaiting)) {
        return defaultClassification;
    } else if (m_firstTrack) {
        static G4ThreadLocal G4bool announced = false;   // once per thread: shows the action is active there
        if (!announced) {
            announced = true;
            G4cout << "QuirkStackingAction: first quirk classified on thread " << G4Threading::G4GetThreadId() << G4endl;
        }
        m_firstTrack = false;
        return fWaiting;
    } else {
        return fWaiting_1;
    }
}

void QuirkStackingAction::NewStage() {
    if (m_defaultStackingAction) {
        m_defaultStackingAction->SetStackManager(stackManager);
        m_defaultStackingAction->NewStage();
    }
}

void QuirkStackingAction::PrepareNewEvent() {
    if (m_defaultStackingAction) {
        m_defaultStackingAction->SetStackManager(stackManager);
        m_defaultStackingAction->PrepareNewEvent();
    }
    m_firstTrack = true;
    Quirk::ClearStrings();
}
