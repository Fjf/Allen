#include "VeloKalmanFilter.cuh"

void velo_kalman_filter::velo_kalman_filter_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_velo_kalman_beamline_states_t>(
    arguments,
    first<host_number_of_reconstructed_velo_tracks_t>(arguments) * Velo::Consolidated::kalman_states_number_of_arrays *
      sizeof(uint32_t));
}

void velo_kalman_filter::velo_kalman_filter_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  HostBuffers& host_buffers,
  cudaStream_t& cuda_stream,
  cudaEvent_t&) const
{
  global_function(velo_kalman_filter)(
    dim3(first<host_number_of_selected_events_t>(arguments)), property<block_dim_t>(), cuda_stream)(arguments);

  if (runtime_options.do_check) {
    cudaCheck(cudaMemcpyAsync(
      host_buffers.host_kalmanvelo_states,
      data<dev_velo_kalman_beamline_states_t>(arguments),
      size<dev_velo_kalman_beamline_states_t>(arguments),
      cudaMemcpyDeviceToHost,
      cuda_stream));
  }
}

/**
 * @brief Helper function to filter one hit
 */
__device__ void velo_kalman_filter::velo_kalman_filter_step(
  const float z,
  const float zhit,
  const float xhit,
  const float whit,
  float& x,
  float& tx,
  float& covXX,
  float& covXTx,
  float& covTxTx)
{
  // compute the prediction
  const float dz = zhit - z;
  const float predx = x + dz * tx;

  const float dz_t_covTxTx = dz * covTxTx;
  const float predcovXTx = covXTx + dz_t_covTxTx;
  const float dx_t_covXTx = dz * covXTx;

  const float predcovXX = covXX + 2 * dx_t_covXTx + dz * dz_t_covTxTx;
  const float predcovTxTx = covTxTx;
  // compute the gain matrix
  const float R = 1.0f / ((1.0f / whit) + predcovXX);
  const float Kx = predcovXX * R;
  const float KTx = predcovXTx * R;
  // update the state vector
  const float r = xhit - predx;
  x = predx + Kx * r;
  tx = tx + KTx * r;
  // update the covariance matrix. we can write it in many ways ...
  covXX /*= predcovXX  - Kx * predcovXX */ = (1 - Kx) * predcovXX;
  covXTx /*= predcovXTx - predcovXX * predcovXTx / R */ = (1 - Kx) * predcovXTx;
  covTxTx = predcovTxTx - KTx * predcovXTx;
  // not needed by any other algorithm
  // const float chi2 = r * r * R;
}

__global__ void velo_kalman_filter::velo_kalman_filter(velo_kalman_filter::Parameters parameters)
{
  const unsigned number_of_events = gridDim.x;
  const unsigned event_number = blockIdx.x;

  // Consolidated datatypes
  const Velo::Consolidated::Tracks velo_tracks {
    parameters.dev_offsets_all_velo_tracks,
    parameters.dev_offsets_velo_track_hit_number,
    event_number,
    number_of_events};

  Velo::Consolidated::ConstStates velo_states {parameters.dev_velo_states, velo_tracks.total_number_of_tracks()};

  Velo::Consolidated::KalmanStates kalmanvelo_states {
    parameters.dev_velo_kalman_beamline_states, velo_tracks.total_number_of_tracks()};

  const unsigned number_of_tracks_event = velo_tracks.number_of_tracks(event_number);
  const unsigned event_tracks_offset = velo_tracks.tracks_offset(event_number);

  for (unsigned i = threadIdx.x; i < number_of_tracks_event; i += blockDim.x) {

    Velo::Consolidated::ConstHits consolidated_hits = velo_tracks.get_hits(parameters.dev_velo_track_hits.get(), i);
    const unsigned n_hits = velo_tracks.number_of_hits(i);

    MiniState stateAtBeamline = velo_states.getMiniState(event_tracks_offset + i);

    KalmanVeloState kalmanbeam_state = simplified_fit<true>(consolidated_hits, stateAtBeamline, n_hits);

    kalmanvelo_states.set(event_tracks_offset + i, kalmanbeam_state);
  }
}