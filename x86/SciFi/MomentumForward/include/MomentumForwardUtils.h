#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include <cassert>

#include "SciFiDefinitions.cuh"
#include "SciFiEventModel.cuh"

void get_offset_and_n_hits_for_layer(
  const int first_zone, 
  const SciFi::HitCount& scifi_hit_count, 
  const float y, 
  int& n_hits, 
  int& zone_offset);

MiniState state_at_z(const MiniState state, const float z);

float y_at_z(const MiniState state, const float z);
