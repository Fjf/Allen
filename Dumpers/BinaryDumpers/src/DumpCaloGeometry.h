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
#pragma once

// Include files
#include "DumpGeometry.h"
#include <CaloDet/DeCalorimeter.h>

/** @class DumpCaloGeometry
 *  Dump calo geometry information
 *
 *  @author Roel Aaij
 *  @date   2019-02-21
 */
class DumpCaloGeometry final : public DumpGeometry<DeCalorimeter> {
public:
  DumpCaloGeometry(std::string name, ISvcLocator* loc) : DumpGeometry<DeCalorimeter> {std::move(name), loc, ""} {}

protected:
  DumpUtils::Dumps dumpGeometry() const override;
};