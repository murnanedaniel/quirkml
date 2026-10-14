/*
  Copyright (C) 2002-2024 CERN for the benefit of the ATLAS collaboration
*/

#include "DummyEquation.h"

DummyEquation::DummyEquation() :
  G4EquationOfMotion(nullptr),
  m_dummyField (std::make_unique<G4UniformMagField>(0, 0, 0))
{
  SetFieldObj(m_dummyField.get());
}
