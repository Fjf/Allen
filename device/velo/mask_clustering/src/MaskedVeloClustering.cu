/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <MEPTools.h>
#include <MaskedVeloClustering.cuh>
#include <VeloTools.cuh>

INSTANTIATE_ALGORITHM(velo_masked_clustering::velo_masked_clustering_t)

/**
 * @brief Makes a 8-connectivity mask
 * @details For a pixel x, constructs its 8-connectivity mask +:
 *
 *          +++
 *          +x+
 *          +++
 *
 *          Using the shift operation, it is possible to make the
 *          connectivity mask of the 64 bits in the forming cluster with few operations.
 */
__device__ uint64_t make_8con_mask(const uint64_t cluster)
{
  return cluster | ((cluster & 0x7FFF7FFF7FFF7FFF) << 1) | ((cluster & 0xFFFEFFFEFFFEFFFE) << 15) | (cluster << 16) |
         ((cluster & 0xFFFCFFFCFFFCFFFC) << 17) | ((cluster & 0xFFFEFFFEFFFEFFFE) >> 1) |
         ((cluster & 0x7FFF7FFF7FFF7FFF) >> 15) | (cluster >> 16) | ((cluster & 0x3FFF3FFF3FFF3FFF) >> 17);
}

/**
 * @brief Makes a connectivity mask to the east of the current cluster.
 */
__device__ uint64_t mask_east(const uint64_t cluster)
{
  const auto mask = (cluster >> 48);
  return mask | ((mask & 0x7FFF7FFF7FFF7FFF) << 1) | (mask >> 1);
}

/**
 * @brief Makes a connectivity mask to the west of the current cluster.
 */
__device__ uint64_t mask_west(const uint64_t cluster)
{
  const auto mask = (cluster << 48);
  return mask | (mask << 1) | ((mask & 0xFFFEFFFEFFFEFFFE) >> 1);
}

/**
 * @brief Makes a cluster with a pixel map and a starting pixel.
 */
__device__ std::tuple<uint64_t, uint32_t, uint32_t, uint32_t> make_cluster(
  const uint64_t pixel_map,
  const uint64_t starting_pixel,
  const unsigned col_lower_limit,
  const unsigned row_lower_limit)
{
  uint64_t current_cluster = 0;
  uint64_t next_cluster = starting_pixel;

  // Extend cluster until the cluster does not update anymore
  while (current_cluster != next_cluster) {
    current_cluster = next_cluster;
    next_cluster = pixel_map & make_8con_mask(current_cluster);
  }

  // Fetch the number of pixels in the cluster
  const unsigned n = __popcll(current_cluster);

  // Get the weight of the cluster in x
  const unsigned x = col_lower_limit * n + __popcll(current_cluster & 0x00000000FFFF0000) +
                     __popcll(current_cluster & 0x0000FFFF00000000) * 2 +
                     __popcll(current_cluster & 0xFFFF000000000000) * 3;

  // Get the weight of the cluster in y
  const unsigned y =
    row_lower_limit * n + __popcll(current_cluster & 0x0002000200020002) +
    __popcll(current_cluster & 0x0004000400040004) * 2 + __popcll(current_cluster & 0x0008000800080008) * 3 +
    __popcll(current_cluster & 0x0010001000100010) * 4 + __popcll(current_cluster & 0x0020002000200020) * 5 +
    __popcll(current_cluster & 0x0040004000400040) * 6 + __popcll(current_cluster & 0x0080008000800080) * 7 +
    __popcll(current_cluster & 0x0100010001000100) * 8 + __popcll(current_cluster & 0x0200020002000200) * 9 +
    __popcll(current_cluster & 0x0400040004000400) * 10 + __popcll(current_cluster & 0x0800080008000800) * 11 +
    __popcll(current_cluster & 0x1000100010001000) * 12 + __popcll(current_cluster & 0x2000200020002000) * 13 +
    __popcll(current_cluster & 0x4000400040004000) * 14;

  return {current_cluster, x, y, n};
}

/**
 * @brief Helper function to print contents of pixel array.
 */
__device__ void print_array(const uint32_t* p, const int row = -1, const int col = -1)
{
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 6; ++c) {
      if (r == row && c == col) {
        printf("x");
      }
      else {
        const int temp_sp_col = c / 2;
        const bool temp_pixel = (p[temp_sp_col] >> (16 * (c % 2) + (r % 16))) & 0x01;
        printf("%i", temp_pixel);
      }
      if (((c + 1) % 2) == 0) printf(" ");
    }
    printf("\n");
    if (((r + 1) % 4) == 0) printf("\n");
  }
  printf("\n");
}

/**
 * @brief Helper function to print contents of 64-bit forming cluster.
 */
__device__ void print_array_64(const uint64_t p, const int row = -1, const int col = -1)
{
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 4; ++c) {
      if (r == row && c == col) {
        printf("x");
      }
      else {
        const bool temp_pixel = (p >> ((c * 16) + r)) & 0x01;
        printf("%i", temp_pixel);
      }
      if (((c + 1) % 2) == 0) printf(" ");
    }
    printf("\n");
    if (((r + 1) % 4) == 0) printf("\n");
  }
  printf("\n");
}

template<int decoding_version>
__device__ void no_neighbour_sp(
  unsigned const* module_pair_cluster_start,
  uint8_t const* dev_velo_sp_patterns,
  Velo::Clusters& velo_cluster_container,
  unsigned* module_pair_cluster_num,
  const float* dev_velo_sp_fx,
  const float* dev_velo_sp_fy,
  VeloGeometry const& g,
  int const module_pair_number,
  unsigned const cluster_start,
  Velo::VeloRawBank<decoding_version> const& raw_bank)
{
  // need to double up as two indedependet sensors in the bank
  const float* ltg0;
  uint32_t n_sp;

  if constexpr (decoding_version == 2 || decoding_version == 3) {
    ltg0 = g.ltg + g.n_trans * raw_bank.sensor_pair();
    n_sp = raw_bank.count;
  }
  else {
    ltg0 = g.ltg + g.n_trans * raw_bank.sensor_index0();
    n_sp = raw_bank.size / 4;
  }

  for (unsigned sp_index = 0; sp_index < n_sp; ++sp_index) {
    // Decode sp
    const uint32_t sp_word = raw_bank.word[sp_index];
    const uint32_t sp_addr = (sp_word & 0x007FFF00U) >> 8;
    const uint32_t no_sp_neighbours = sp_word & 0x80000000U;
    uint32_t sensor_index;

    if constexpr (decoding_version == 2 || decoding_version == 3) {
      sensor_index = raw_bank.sourceID;
    }
    else {
      const uint32_t sensorBit = (sp_word & 0x800000U) ? (0x1U << 22) : 0x0U;
      sensor_index = sensorBit ? raw_bank.sensor_index1() : raw_bank.sensor_index0();
    }

    // There are no neighbours, so compute the number of pixels of this superpixel
    if (no_sp_neighbours) {
      // Look up pre-generated patterns
      const int32_t sp_row = sp_addr & 0x3FU;
      const int32_t sp_col = (sp_addr >> 6);
      const uint8_t sp = sp_word & 0xFFU;

      const uint32_t idx = dev_velo_sp_patterns[sp];
      const uint32_t chip = sp_col >> (VP::ChipColumns_division - 1);

      {
        // there is always at least one cluster in the super
        // pixel. look up the pattern and add it.
        const uint32_t row = idx & 0x03U;
        const uint32_t col = (idx >> 2) & 1;
        const uint32_t cx = sp_col * 2 + col;
        const uint32_t cy = sp_row * 4 + row;

        const unsigned cid = get_channel_id(sensor_index, chip, cx & VP::ChipColumns_mask, cy);

        const float fx = dev_velo_sp_fx[sp * 2];
        const float fy = dev_velo_sp_fy[sp * 2];
        const float local_x = g.local_x[cx] + fx * g.x_pitch[cx];
        const float local_y = (cy + 0.5f + fy) * Velo::Constants::pixel_size;

        const unsigned cluster_num = atomicAdd(module_pair_cluster_num + module_pair_number, 1);

#if ALLEN_DEBUG
        const auto module_estimated_num =
          module_pair_cluster_start[module_pair_number + 1] - module_pair_cluster_start[module_pair_number];
        assert(cluster_num <= module_estimated_num);
#else
        _unused(module_pair_cluster_start);
#endif

        float gx, gy, gz;
        if constexpr (decoding_version == 2 || decoding_version == 3) {
          gx = (ltg0[0] * local_x + ltg0[1] * local_y + ltg0[9]);
          gy = (ltg0[3] * local_x + ltg0[4] * local_y + ltg0[10]);
          gz = (ltg0[6] * local_x + ltg0[7] * local_y + ltg0[11]);
        }
        else {
          const uint32_t sensorBit = (sp_word & 0x800000U) ? (0x1U << 22) : 0x0U;
          const float* ltg1 = g.ltg + g.n_trans * raw_bank.sensor_index1();
          if (sensorBit) {
            gx = (ltg1[0] * local_x + ltg1[1] * local_y + ltg1[9]);
            gy = (ltg1[3] * local_x + ltg1[4] * local_y + ltg1[10]);
            gz = (ltg1[6] * local_x + ltg1[7] * local_y + ltg1[11]);
          }
          else {
            gx = (ltg0[0] * local_x + ltg0[1] * local_y + ltg0[9]);
            gy = (ltg0[3] * local_x + ltg0[4] * local_y + ltg0[10]);
            gz = (ltg0[6] * local_x + ltg0[7] * local_y + ltg0[11]);
          }
        }

        const auto cluster_index = cluster_start + cluster_num;
        velo_cluster_container.set_id(cluster_index, get_lhcb_id(cid));
        velo_cluster_container.set_x(cluster_index, gx);
        velo_cluster_container.set_y(cluster_index, gy);
        velo_cluster_container.set_z(cluster_index, gz);
        velo_cluster_container.set_phi(cluster_index, hit_phi_16(gx, gy));
      }

      // if there is a second cluster for this pattern
      // add it as well.
      if (idx & 8) {
        const uint32_t row = (idx >> 4) & 3;
        const uint32_t col = (idx >> 6) & 1;
        const uint32_t cx = sp_col * 2 + col;
        const uint32_t cy = sp_row * 4 + row;

        unsigned cid = get_channel_id(sensor_index, chip, cx & VP::ChipColumns_mask, cy);

        const float fx = dev_velo_sp_fx[sp * 2 + 1];
        const float fy = dev_velo_sp_fy[sp * 2 + 1];
        const float local_x = g.local_x[cx] + fx * g.x_pitch[cx];
        const float local_y = (cy + 0.5f + fy) * Velo::Constants::pixel_size;

        const unsigned cluster_num = atomicAdd(module_pair_cluster_num + module_pair_number, 1);

#if ALLEN_DEBUG
        const auto module_estimated_num =
          module_pair_cluster_start[module_pair_number + 1] - module_pair_cluster_start[module_pair_number];
        assert(cluster_num <= module_estimated_num);
#endif

        float gx, gy, gz;
        if constexpr (decoding_version == 2 || decoding_version == 3) {
          gx = (ltg0[0] * local_x + ltg0[1] * local_y + ltg0[9]);
          gy = (ltg0[3] * local_x + ltg0[4] * local_y + ltg0[10]);
          gz = (ltg0[6] * local_x + ltg0[7] * local_y + ltg0[11]);
        }
        else {
          const uint32_t sensorBit = (sp_word & 0x800000U) ? (0x1U << 22) : 0x0U;
          const float* ltg1 = g.ltg + g.n_trans * raw_bank.sensor_index1();
          if (sensorBit) {
            gx = (ltg1[0] * local_x + ltg1[1] * local_y + ltg1[9]);
            gy = (ltg1[3] * local_x + ltg1[4] * local_y + ltg1[10]);
            gz = (ltg1[6] * local_x + ltg1[7] * local_y + ltg1[11]);
          }
          else {
            gx = (ltg0[0] * local_x + ltg0[1] * local_y + ltg0[9]);
            gy = (ltg0[3] * local_x + ltg0[4] * local_y + ltg0[10]);
            gz = (ltg0[6] * local_x + ltg0[7] * local_y + ltg0[11]);
          }
        }
        const auto cluster_index = cluster_start + cluster_num;
        velo_cluster_container.set_id(cluster_index, get_lhcb_id(cid));
        velo_cluster_container.set_x(cluster_index, gx);
        velo_cluster_container.set_y(cluster_index, gy);
        velo_cluster_container.set_z(cluster_index, gz);
        velo_cluster_container.set_phi(cluster_index, hit_phi_16(gx, gy));
      }
    }
  }
}

template<int decoding_version>
__device__ void rest_of_clusters(
  unsigned const* module_pair_cluster_start,
  Velo::Clusters velo_cluster_container,
  unsigned* module_pair_cluster_num,
  VeloGeometry const& g,
  uint32_t const candidate,
  Velo::VeloRawBank<decoding_version> const& raw_bank)
{
  const auto sp_index = candidate >> 11;
  const auto sensor_number = (candidate >> 3) & 0xFF;
  const auto module_pair_number = sensor_number / 8;
  const auto starting_pixel_location = candidate & 0x7;
  const float* ltg = g.ltg + g.n_trans * sensor_number;
  uint32_t n_sp;

  if constexpr (decoding_version == 2 || decoding_version == 3) {
    n_sp = raw_bank.count;
  }
  else {
    n_sp = raw_bank.size / 4;
  }

  const uint32_t sp_word = raw_bank.word[sp_index];
  const uint32_t sp_addr = (sp_word & 0x007FFF00U) >> 8;
  // Note: In the code below, row and col are int32_t (not unsigned)
  //       This is not a bug
  const int32_t sp_row = sp_addr & 0x3FU;
  const int32_t sp_col = sp_addr >> 6;

  // Find candidates that follow this condition:
  // For pixel +, all pixels - should *not* be populated
  // - -
  // + -
  //   -

  // Load the following SPs,
  // where x is the SP containing the possible candidates, o are other SPs:
  // ooooo
  // oooxo
  // ooooo
  // ooooo
  //
  // Each column of SPs are in one uint32_t
  // Order is from left to right
  //
  // 0: o 1: o 2: o 3: o 4: o
  //    o    o    o    x    o
  //    o    o    o    o    o
  //    o    o    o    o    o
  //
  // Order inside an uint32_t is from bottom to top. Eg. 3:
  // 3: o
  // 2: x
  // 1: o
  // 0: o
  uint32_t pixel_array[5] = {0, 0, 0, 0, 0};

  // sp limits to load
  const int32_t sp_row_lower_limit = sp_row - 2;
  const int32_t sp_row_upper_limit = sp_row + 1;
  const int32_t sp_col_lower_limit = sp_col - 3;
  const int32_t sp_col_upper_limit = sp_col + 1;

  // Row limits
  const int32_t row_lower_limit = sp_row_lower_limit * 4;
  const int32_t col_lower_limit = sp_col_lower_limit * 2;

  // Load SPs
  // Note: We will pick up the current one,
  //       no need to add a special case
  for (unsigned k = 0; k < n_sp; ++k) {
    const uint32_t other_sp_word = raw_bank.word[k];
    const uint32_t other_no_sp_neighbours = other_sp_word & 0x80000000U;
    if constexpr (decoding_version > 3) {
      const uint32_t otherSensorBit = (other_sp_word & 0x800000U) ? (0x1U) : 0x0U;
      // Check if the other SP id from the same sensor in the pair in this bank
      if (otherSensorBit != (sensor_number & 0x1)) continue;
    }
    if (!other_no_sp_neighbours) {
      const uint32_t other_sp_addr = (other_sp_word & 0x007FFF00U) >> 8;
      const int32_t other_sp_row = other_sp_addr & 0x3FU;
      const int32_t other_sp_col = (other_sp_addr >> 6);
      const uint8_t other_sp = other_sp_word & 0xFFU;

      if (
        other_sp_row >= sp_row_lower_limit && other_sp_row <= sp_row_upper_limit &&
        other_sp_col >= sp_col_lower_limit && other_sp_col <= sp_col_upper_limit) {
        const int relative_row = other_sp_row - sp_row_lower_limit;
        const int relative_col = other_sp_col - sp_col_lower_limit;

        // Note: Order is:
        // 15 31
        // 14 30
        // 13 29
        // 12 28
        // 11 27
        // 10 26
        //  9 25
        //  8 24
        //  7 23
        //  6 22
        //  5 21
        //  4 20
        //  3 19
        //  2 18
        //  1 17
        //  0 16
        pixel_array[relative_col] |= (other_sp & 0X0F) << (4 * relative_row) | (other_sp & 0XF0)
                                                                                 << (12 + 4 * relative_row);
      }
    }
  }

  // Pixel array is now populated
  // Work with candidate on starting_pixel_location
  const auto sp_relative_row = starting_pixel_location & 0x3;
  const auto sp_relative_col = starting_pixel_location < 4;
  const uint32_t col = sp_col * 2 + sp_relative_col;

  // Work with a 64-bit number
  const uint64_t starting_pixel = ((uint64_t)(0x01 << (11 - sp_relative_row)) << (16 * (col & 0x01))) << 32;
  const uint64_t pixel_map = (((uint64_t) pixel_array[3]) << 32) | pixel_array[2];

  // Make cluster with mask clustering method
  const auto cluster = make_cluster(pixel_map, starting_pixel, col_lower_limit + 4, row_lower_limit);

  // Check if there are any hits with precedence, in which case discard the cluster,
  // as another candidate will generate it. Hits with precedence are:
  // * Hits to the east of the starting pixel
  // * Hits to the north of the starting pixel
  const uint64_t hits_with_precedence =
    // Hits to the east, populated in the first 16 bits
    (mask_east(std::get<0>(cluster)) & pixel_array[4]) |
    // Hits in the current cluster with precedence in the latter 16 bits
    (std::get<0>(cluster) & (starting_pixel ^ -starting_pixel));

  // Keep the cluster if:
  // * There are no hits with precedence
  // * There are no pixels in the north-most or the south-most row.
  //   If that were the case, the cluster very likely is not complete.
  bool keep_cluster = hits_with_precedence == 0 && !__popcll(std::get<0>(cluster) & 0x8000800080008000) &&
                      !__popcll(std::get<0>(cluster) & 0x0001000100010001);

  if (keep_cluster) {
    unsigned x = std::get<1>(cluster);
    unsigned y = std::get<2>(cluster);
    unsigned n = std::get<3>(cluster);

    // Interrogate if cluster needs to be extended to the west
    if (__popcll(std::get<0>(cluster) & 0xFFFF)) {
      const uint64_t west_pixel_map = (((uint64_t) pixel_array[1]) << 32) | pixel_array[0];
      const uint64_t west_start_pixel = mask_west(std::get<0>(cluster)) & west_pixel_map;

      // Extend the cluster to the west
      const auto cluster_extension = make_cluster(west_pixel_map, west_start_pixel, col_lower_limit, row_lower_limit);

      // Update x, y, n
      x += std::get<1>(cluster_extension);
      y += std::get<2>(cluster_extension);
      n += std::get<3>(cluster_extension);

      // If there were pixels in the north-most row, the south-most row, or the west-most column,
      // the cluster very likely is not complete, so don't keep it.
      keep_cluster &= !__popcll(std::get<0>(cluster_extension) & 0x8000800080008000) &&
                      !__popcll(std::get<0>(cluster_extension) & 0xFFFF) &&
                      !__popcll(std::get<0>(cluster) & 0x0001000100010001);
    }

    // Make the clusters that for sure are good clusters
    if (keep_cluster) {
      const unsigned cx = x / n;
      const unsigned cy = y / n;

      const float fx = x / static_cast<float>(n) - cx;
      const float fy = y / static_cast<float>(n) - cy;

      // store target (3D point for tracking)
      const uint32_t chip = cx >> VP::ChipColumns_division;
      const unsigned cid = get_channel_id(sensor_number, chip, cx & VP::ChipColumns_mask, cy);

      const float local_x = g.local_x[cx] + fx * g.x_pitch[cx];
      const float local_y = (cy + 0.5f + fy) * Velo::Constants::pixel_size;

      const unsigned cluster_num = atomicAdd(module_pair_cluster_num + module_pair_number, 1);

#if ALLEN_DEBUG
      const auto module_estimated_num =
        module_pair_cluster_start[module_pair_number + 1] - module_pair_cluster_start[module_pair_number];
      assert(cluster_num <= module_estimated_num);
#endif

      const float gx = ltg[0] * local_x + ltg[1] * local_y + ltg[9];
      const float gy = ltg[3] * local_x + ltg[4] * local_y + ltg[10];
      const float gz = ltg[6] * local_x + ltg[7] * local_y + ltg[11];

      const unsigned cluster_start = module_pair_cluster_start[module_pair_number];

      const auto cluster_index = cluster_start + cluster_num;
      velo_cluster_container.set_id(cluster_index, get_lhcb_id(cid));
      velo_cluster_container.set_x(cluster_index, gx);
      velo_cluster_container.set_y(cluster_index, gy);
      velo_cluster_container.set_z(cluster_index, gz);
      velo_cluster_container.set_phi(cluster_index, hit_phi_16(gx, gy));
    }
  }
}

template<int decoding_version, bool mep_layout>
__global__ void velo_masked_clustering_kernel(
  velo_masked_clustering::Parameters parameters,
  const unsigned event_start,
  const VeloGeometry* dev_velo_geometry,
  const uint8_t* dev_velo_sp_patterns,
  const float* dev_velo_sp_fx,
  const float* dev_velo_sp_fy)
{
  const unsigned number_of_events = parameters.dev_number_of_events[0];
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];

  const unsigned* module_pair_cluster_start =
    parameters.dev_offsets_estimated_input_size + event_number * Velo::Constants::n_module_pairs;
  unsigned* module_pair_cluster_num =
    parameters.dev_module_pair_cluster_num + event_number * Velo::Constants::n_module_pairs;
  unsigned number_of_candidates = parameters.dev_module_pair_candidate_num[event_number];
  const unsigned* cluster_candidates =
    parameters.dev_cluster_candidates + parameters.dev_candidates_offsets[event_number];

  // Local pointers to parameters.dev_velo_cluster_container
  const unsigned estimated_number_of_clusters =
    parameters.dev_offsets_estimated_input_size[Velo::Constants::n_module_pairs * number_of_events];
  auto velo_cluster_container = Velo::Clusters {parameters.dev_velo_cluster_container, estimated_number_of_clusters};
  parameters.dev_velo_clusters[event_number] = velo_cluster_container;

  // Load Velo geometry (assume it is the same for all events)
  const VeloGeometry& g = *dev_velo_geometry;
  const auto velo_raw_event = Velo::RawEvent<decoding_version, mep_layout> {parameters.dev_velo_raw_input,
                                                                            parameters.dev_velo_raw_input_offsets,
                                                                            parameters.dev_velo_raw_input_sizes,
                                                                            parameters.dev_velo_raw_input_types,
                                                                            event_number + event_start};

  // process no neighbour sp
  for (unsigned raw_bank_number = threadIdx.x; raw_bank_number < velo_raw_event.number_of_raw_banks();
       raw_bank_number += blockDim.x) {
    const auto raw_bank = velo_raw_event.raw_bank(raw_bank_number);

    if (raw_bank.type != LHCb::RawBank::VP && raw_bank.type != LHCb::RawBank::Velo) continue;

    unsigned module_pair_number = 0;
    if constexpr (decoding_version == 2 || decoding_version == 3) {
      module_pair_number = raw_bank.sensor_pair() / 8;
    }
    else {
      module_pair_number = raw_bank.sensor_index0() / 8;
    }
    const unsigned cluster_start = module_pair_cluster_start[module_pair_number];

    no_neighbour_sp<decoding_version>(
      module_pair_cluster_start,
      dev_velo_sp_patterns,
      velo_cluster_container,
      module_pair_cluster_num,
      dev_velo_sp_fx,
      dev_velo_sp_fy,
      g,
      module_pair_number,
      cluster_start,
      raw_bank);
  }

  __syncthreads();

  // Process rest of clusters
  for (unsigned candidate_number = threadIdx.x; candidate_number < number_of_candidates;
       candidate_number += blockDim.x) {
    const uint32_t candidate = cluster_candidates[candidate_number];
    uint8_t sensor_number = (candidate >> 3) & 0xFF;

    assert(sensor_number < Velo::Constants::n_sensors);

    const auto raw_bank_number = parameters.dev_velo_bank_index[sensor_number];
    const auto raw_bank = velo_raw_event.raw_bank(raw_bank_number);
    rest_of_clusters<decoding_version>(
      module_pair_cluster_start, velo_cluster_container, module_pair_cluster_num, g, candidate, raw_bank);
  }
}

void velo_masked_clustering::velo_masked_clustering_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_module_cluster_num_t>(
    arguments, first<host_number_of_events_t>(arguments) * Velo::Constants::n_module_pairs);
  set_size<dev_velo_cluster_container_t>(
    arguments, first<host_total_number_of_velo_clusters_t>(arguments) * Velo::Clusters::element_size);
  set_size<dev_velo_clusters_t>(arguments, first<host_number_of_events_t>(arguments));
}

void velo_masked_clustering::velo_masked_clustering_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_module_cluster_num_t>(arguments, 0, context);

  auto const bank_version = first<host_raw_bank_version_t>(arguments);

  if (bank_version < 0) return; // no VP banks present in data

  // Selector from layout
  auto kernel_fn = (bank_version == 2) ?
                     (runtime_options.mep_layout ? global_function(velo_masked_clustering_kernel<2, true>) :
                                                   global_function(velo_masked_clustering_kernel<2, false>)) :
                     (bank_version == 3) ?
                     (runtime_options.mep_layout ? global_function(velo_masked_clustering_kernel<3, true>) :
                                                   global_function(velo_masked_clustering_kernel<3, false>)) :
                     (runtime_options.mep_layout ? global_function(velo_masked_clustering_kernel<4, true>) :
                                                   global_function(velo_masked_clustering_kernel<4, false>));

  kernel_fn(dim3(size<dev_event_list_t>(arguments)), property<block_dim_t>(), context)(
    arguments,
    std::get<0>(runtime_options.event_interval),
    constants.dev_velo_geometry,
    constants.dev_velo_sp_patterns.data(),
    constants.dev_velo_sp_fx.data(),
    constants.dev_velo_sp_fy.data());
}
