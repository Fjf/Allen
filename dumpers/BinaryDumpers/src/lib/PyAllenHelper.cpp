/*****************************************************************************\
* (c) Copyright 2000-2019 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the GNU General Public      *
* Licence version 3 (GPL Version 3), copied verbatim in the file "COPYING".   *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include <Dumpers/IUpdater.h>
#include <GaudiKernel/Service.h>

Allen::NonEventData::IUpdater* cast_updater( IService* updater_svc ) {
  return dynamic_cast<Allen::NonEventData::IUpdater*>( updater_svc );
}
