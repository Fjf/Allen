#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <algorithm>
#include <fstream>

#include <cassert>

#include "SciFiDefinitions.cuh"
#include "SciFiEventModel.cuh"
#include "LookingForwardConstants.h"
#include "MomentumForwardUtils.h"
#include "BinarySearch.cuh"

#include <functional>

struct SciFiWindowsParams {
  float dx_slope = 4000000;
  float dx_min = 200;
  float dx_weight = 0.5;
  float tx_slope = 4000000;
  float tx_min = 200;
  float tx_weight = 0.5;
  float max_window_layer0 = 600;
  float max_window_layer1 = 10; // 20;
  float max_window_layer2 = 10; // 20;
  float max_window_layer3 = 20; // 40;
  float chi2_cut = 100;         // 40;
};

class Window_stat {
public:
  Window_stat() {}

  Window_stat(int num_hits, float x_center, float dx)
  {
    this->num_hits = num_hits;
    this->x_center = x_center;
    this->x_min = x_center - dx;
    this->x_max = x_center + dx;
  }

  Window_stat(int num_hits, float x_center, float x_min, float x_max)
  {
    this->num_hits = num_hits;
    this->x_center = x_center;
    this->x_min = x_min;
    this->x_max = x_max;
  }

  int num_hits;
  float x_center;
  float x_min, x_max;
};

float x_at_z(const MiniState& state, const float z);

MiniState propagate_state_from_velo(const MiniState& UT_state, float qop, int layer);

bool select_hits(
  const MiniState& velo_UT_state,
  float UT_qop,
  unsigned int UT_track_index,
  const SciFi::Hits& hits,
  const SciFi::HitCount& hit_count,
  const int station,
  std::vector<SciFi::TrackHits>& track_candidate,
  std::array<std::vector<Window_stat>, 4>& window_stats,
  const SciFiWindowsParams& window_params);

float dx_calc(const MiniState& state, float qop, const SciFiWindowsParams& window_params);

std::tuple<int, int> find_x_in_window(
  const SciFi::Hits& hits,
  const int num_hits,
  const int zone_offset,
  const float x_min,
  const float x_max);

float linear_propagation(float x_0, float tx, float dz);

float get_chi_2(
  const std::vector<float>& x,
  const std::vector<float>& y,
  std::function<float(float)> expected_function);

void linear_regression(const std::vector<float>& x, const std::vector<float>& y, float& m, float& q, float& chi_2);
