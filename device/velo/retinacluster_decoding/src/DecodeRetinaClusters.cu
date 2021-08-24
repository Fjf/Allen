#include <MEPTools.h>
#include <DecodeRetinaClusters.cuh>

void decode_retinaclusters::decode_retinaclusters_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_module_cluster_num_t>(
                                     arguments, first<host_number_of_selected_events_t>(arguments) * Velo::Constants::n_module_pairs);
  set_size<dev_velo_cluster_container_t>(arguments,
                                         first<host_total_number_of_velo_clusters_t>(arguments) * Velo::Clusters::element_size);
  set_size<dev_offsets_module_pair_cluster_t>(arguments,
                                         first<host_number_of_selected_events_t>(arguments) * Velo::Constants::n_module_pairs + 1);
}

void decode_retinaclusters::decode_retinaclusters_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers&,
  cudaStream_t& cuda_stream,
  cudaEvent_t&) const
{
  initialize<dev_module_cluster_num_t>(arguments, 0, cuda_stream);
  initialize<dev_offsets_module_pair_cluster_t>(arguments, 0, cuda_stream);

  if (runtime_options.mep_layout) {
    global_function(decode_retinaclusters_mep)(
      dim3(first<host_number_of_selected_events_t>(arguments)), property<block_dim_t>(), cuda_stream)(arguments, constants.dev_velo_geometry);
  } else {
    global_function(decode_retinaclusters)(
      dim3(first<host_number_of_selected_events_t>(arguments)), property<block_dim_t>(), cuda_stream)(arguments, constants.dev_velo_geometry);
  }
}

__device__ void put_retinaclusters_into_container(
  Velo::Clusters velo_cluster_container,
  VeloGeometry const& g,
  uint const cluster_start,
  VeloRawBank const& raw_bank)
{
  const float* ltg = g.ltg + g.n_trans * raw_bank.sensor_index;

  for (uint rc_index = threadIdx.y; rc_index < raw_bank.count; rc_index += blockDim.y) {
    // Decode cluster
    const uint32_t word = raw_bank.word[rc_index];

    const uint32_t cx = ( word >> 14 ) & 0x3FF;
    const float    fx = ( ( word >> 11 ) & 0x7 ) / 8.f;
    const uint32_t cy = ( word >> 3 ) & 0xFF;
    const float    fy = ( word & 0x7FF ) / 8.f;

    const uint32_t chip = cx >> VP::ChipColumns_division;
    const float local_x = g.local_x[cx] + fx * g.x_pitch[cx];
    const float local_y = (0.5f + fy) * Velo::Constants::pixel_size;

    const float gx = ( ltg[0] * local_x + ltg[1] * local_y + ltg[9] );
    const float gy = ( ltg[3] * local_x + ltg[4] * local_y + ltg[10] );
    const float gz = ( ltg[6] * local_x + ltg[7] * local_y + ltg[11] );
    const uint cid = get_channel_id(raw_bank.sensor_index, chip, cx & VP::ChipColumns_mask, cy);

    velo_cluster_container.set_x(cluster_start + rc_index, gx);
    velo_cluster_container.set_y(cluster_start + rc_index, gy);
    velo_cluster_container.set_z(cluster_start + rc_index, gz);
    velo_cluster_container.set_id(cluster_start + rc_index, get_lhcb_id(cid));

  }
}


__global__ void decode_retinaclusters::decode_retinaclusters(
  decode_retinaclusters::Parameters parameters,
  const VeloGeometry* dev_velo_geometry)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const uint selected_event_number = parameters.dev_event_list[event_number];

  const char* raw_input = parameters.dev_velo_retina_raw_input + parameters.dev_velo_retina_raw_input_offsets[selected_event_number];
  const uint* sensor_cluster_start =
    parameters.dev_offsets_each_sensor_size + event_number * Velo::Constants::n_modules * Velo::Constants::n_sensors_per_module;
  uint* module_pair_cluster_num = parameters.dev_module_pair_cluster_num + event_number * Velo::Constants::n_module_pairs;
  uint* offsets_pair_module_size = parameters.dev_offsets_module_pair_cluster + 1 + event_number * Velo::Constants::n_module_pairs;
  uint* first_number_of_dev_offsets_module_pair_cluster = parameters.dev_offsets_module_pair_cluster;
  first_number_of_dev_offsets_module_pair_cluster[0] = 0;

  const uint* offsets_each_sensor_size = parameters.dev_offsets_each_sensor_size + event_number * Velo::Constants::n_modules * Velo::Constants::n_sensors_per_module;

  // Local pointers to parameters.dev_velo_cluster_container
  const uint estimated_number_of_clusters =
    parameters.dev_offsets_each_sensor_size[Velo::Constants::n_module_pairs * number_of_events * 8];
  auto velo_cluster_container = Velo::Clusters {parameters.dev_velo_cluster_container, estimated_number_of_clusters};

  // Load Velo geometry (assume it is the same for all events)
  const VeloGeometry& g = *dev_velo_geometry;


  // Read raw event
  const auto raw_event = VeloRawEvent(raw_input);

  for (uint raw_bank_number = threadIdx.x; raw_bank_number < raw_event.number_of_raw_banks;
       raw_bank_number += blockDim.x) {
    const auto module_pair_number = raw_bank_number / 8;
    const auto offsets_sensor_behind = (module_pair_number + 1) * 8;
    const auto offsets_sensor_in_front = module_pair_number * 8;
    const uint cluster_start = sensor_cluster_start[raw_bank_number];
    offsets_pair_module_size[module_pair_number] = offsets_each_sensor_size[offsets_sensor_behind];
    module_pair_cluster_num[module_pair_number] = offsets_each_sensor_size[offsets_sensor_behind] - offsets_each_sensor_size[offsets_sensor_in_front];

    // Read raw bank
    const auto raw_bank = VeloRawBank(raw_event.payload + raw_event.raw_bank_offset[raw_bank_number]);
    put_retinaclusters_into_container(
      velo_cluster_container,
      g,
      cluster_start,
      raw_bank);
  }
 
}

__global__ void decode_retinaclusters::decode_retinaclusters_mep(
  decode_retinaclusters::Parameters parameters,
  const VeloGeometry* dev_velo_geometry)
{
  const uint number_of_events = gridDim.x;
  const uint event_number = blockIdx.x;
  const uint selected_event_number = parameters.dev_event_list[event_number];

  const uint* sensor_cluster_start =
    parameters.dev_offsets_each_sensor_size + event_number * Velo::Constants::n_modules * Velo::Constants::n_sensors_per_module;
  uint* module_pair_cluster_num = parameters.dev_module_pair_cluster_num + event_number * Velo::Constants::n_module_pairs;
  uint* offsets_pair_module_size = parameters.dev_offsets_module_pair_cluster + 1 + event_number * Velo::Constants::n_module_pairs;
  uint* first_number_of_dev_offsets_module_pair_cluster = parameters.dev_offsets_module_pair_cluster;
  first_number_of_dev_offsets_module_pair_cluster[0] = 0;

  const uint* offsets_each_sensor_size = parameters.dev_offsets_each_sensor_size + event_number * Velo::Constants::n_modules * Velo::Constants::n_sensors_per_module;

  // Local pointers to parameters.dev_velo_cluster_container
  const uint estimated_number_of_clusters =
    parameters.dev_offsets_each_sensor_size[Velo::Constants::n_module_pairs * number_of_events * 8];
  auto velo_cluster_container = Velo::Clusters {parameters.dev_velo_cluster_container, estimated_number_of_clusters};


  // Load Velo geometry (assume it is the same for all events)
  const VeloGeometry& g = *dev_velo_geometry;

  // Read raw event
  auto const number_of_raw_banks = parameters.dev_velo_retina_raw_input_offsets[0];

  for (uint raw_bank_number = threadIdx.x; raw_bank_number < number_of_raw_banks; raw_bank_number += blockDim.x) {
    const auto module_pair_number = raw_bank_number / 8;
    const auto offsets_sensor_behind = (module_pair_number + 1) * 8;
    const auto offsets_sensor_in_front = module_pair_number * 8;
    const uint cluster_start = sensor_cluster_start[raw_bank_number];
    offsets_pair_module_size[module_pair_number] = offsets_each_sensor_size[offsets_sensor_behind];
    module_pair_cluster_num[module_pair_number] = offsets_each_sensor_size[offsets_sensor_behind] - offsets_each_sensor_size[offsets_sensor_in_front];

    // Read raw bank
    const auto raw_bank = MEP::raw_bank<VeloRawBank>(
      parameters.dev_velo_retina_raw_input, parameters.dev_velo_retina_raw_input_offsets, selected_event_number, raw_bank_number);
    put_retinaclusters_into_container(
      velo_cluster_container,
      g,
      cluster_start,
      raw_bank);
  }
 
}
