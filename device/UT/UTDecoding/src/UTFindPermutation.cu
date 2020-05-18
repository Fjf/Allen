#include "UTFindPermutation.cuh"
#include "FindPermutation.cuh"
#include <cstdio>

void ut_find_permutation::ut_find_permutation_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_ut_hit_permutations_t>(arguments, first<host_accumulated_number_of_ut_hits_t>(arguments));
}

void ut_find_permutation::ut_find_permutation_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  HostBuffers&,
  cudaStream_t& cuda_stream,
  cudaEvent_t&) const
{
  device_function(ut_find_permutation)(
    dim3(first<host_number_of_selected_events_t>(arguments), constants.host_unique_x_sector_layer_offsets[4]),
    property<block_dim_t>(),
    cuda_stream)(arguments, constants.dev_unique_x_sector_layer_offsets.data());
}

__global__ void ut_find_permutation::ut_find_permutation(
  ut_find_permutation::Parameters parameters,
  const uint* dev_unique_x_sector_layer_offsets)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const uint sector_group_number = blockIdx.y;
  const uint number_of_unique_x_sectors = dev_unique_x_sector_layer_offsets[4];

  const UT::HitOffsets ut_hit_offsets {
    parameters.dev_ut_hit_offsets, event_number, number_of_unique_x_sectors, dev_unique_x_sector_layer_offsets};

  UT::ConstPreDecodedHits ut_pre_decoded_hits {
    parameters.dev_ut_pre_decoded_hits, parameters.dev_ut_hit_offsets[number_of_events * number_of_unique_x_sectors]};

  const uint sector_group_offset = ut_hit_offsets.sector_group_offset(sector_group_number);
  const uint sector_group_number_of_hits = ut_hit_offsets.sector_group_number_of_hits(sector_group_number);

  // Load yBegin into a shared memory container
  // TODO: Find a proper maximum and cover corner cases
  __shared__ float s_y_begin[UT::Decoding::ut_max_hits_shared_sector_group];

  if (sector_group_number_of_hits > 0) {
    __syncthreads();
    assert(sector_group_number_of_hits < UT::Decoding::ut_max_hits_shared_sector_group);

    for (uint i = threadIdx.x; i < sector_group_number_of_hits; i += blockDim.x) {
      s_y_begin[i] = ut_pre_decoded_hits.sort_key(sector_group_offset + i);
    }

    __syncthreads();

    // Sort according to the natural order in s_y_begin
    // Store the permutation found into parameters.dev_ut_hit_permutations
    find_permutation(
      0,
      sector_group_offset,
      sector_group_number_of_hits,
      parameters.dev_ut_hit_permutations,
      [&](const int a, const int b) -> int { return (s_y_begin[a] > s_y_begin[b]) - (s_y_begin[a] < s_y_begin[b]); });
  }
}
