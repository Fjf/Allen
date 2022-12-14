/*****************************************************************************\
* (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#ifndef ALLEN_STANDALONE
#include "Gaudi/MonitoringHub.h"
#endif

struct StreamServiceLocator {
  static StreamServiceLocator* get()
  {
    static StreamServiceLocator instance;
    return &instance;
  }

#ifndef ALLEN_STANDALONE
  Gaudi::Monitoring::Hub& monitoringHub() { return m_monitoringHub; }
#endif

private:
  StreamServiceLocator() = default;

  ~StreamServiceLocator() = default;

#ifndef ALLEN_STANDALONE
  Gaudi::Monitoring::Hub m_monitoringHub {};
#endif
};
