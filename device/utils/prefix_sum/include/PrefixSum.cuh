/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "BackendCommon.h"
#include "VeloEventModel.cuh"
#include "SciFiEventModel.cuh"
#include "UTEventModel.cuh"

__global__ void prefix_sum_reduce(unsigned* dev_main_array, unsigned* dev_auxiliary_array, const unsigned array_size);

__global__ void prefix_sum_single_block(unsigned* dev_total_sum, unsigned* dev_array, const unsigned array_size);

__global__ void prefix_sum_scan(unsigned* dev_main_array, unsigned* dev_auxiliary_array, const unsigned array_size);
