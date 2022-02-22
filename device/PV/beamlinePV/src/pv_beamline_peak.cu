/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "pv_beamline_peak.cuh"

INSTANTIATE_ALGORITHM(pv_beamline_peak::pv_beamline_peak_t)

void pv_beamline_peak::pv_beamline_peak_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_zpeaks_t>(arguments, first<host_number_of_events_t>(arguments) * PV::max_number_vertices);
  set_size<dev_number_of_zpeaks_t>(arguments, first<host_number_of_events_t>(arguments));
}

void pv_beamline_peak::pv_beamline_peak_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  HostBuffers&,
  const Allen::Context& context) const
{
  global_function(pv_beamline_peak)(dim3(size<dev_event_list_t>(arguments)), warp_size, context)(
    arguments, size<dev_event_list_t>(arguments));
}

__global__ void pv_beamline_peak::pv_beamline_peak(
  pv_beamline_peak::Parameters parameters,
  const unsigned event_list_size)
{
  auto event_index = blockIdx.x;
  const unsigned event_number = parameters.dev_event_list[event_index];

  const float* zhisto = parameters.dev_zhisto + BeamlinePVConstants::Common::Nbins * event_number;
  float* zpeaks = parameters.dev_zpeaks + PV::max_number_vertices * event_number;

  __shared__ Cluster clusters[PV::max_number_of_clusters]; // 200
  unsigned number_of_clusters = 0;
  using BinIndex = unsigned short;
  __shared__ BinIndex clusteredges[PV::max_number_clusteredges]; // 200
  unsigned number_of_clusteredges = 0;

#if defined(TARGET_DEVICE_CUDA)
  uint32_t prev_mask = -1u;
  uint32_t lanemask_eq = 1 << threadIdx.x;
  uint32_t lanemask_lt = lanemask_eq - 1;
  // Use a warp (32 threads) to process the 3200 bins in 100 iterations:
  for (auto i = threadIdx.x; i < BeamlinePVConstants::Common::Nbins; i += blockDim.x) {
    const float zBin = BeamlinePVConstants::Common::zmin + i * BeamlinePVConstants::Common::dz;
    const float Z0Err = zBin < BeamlinePVConstants::Common::SMOG2_pp_separation ?
                          BeamlinePVConstants::Common::SMOG2_maxTrackZ0Err :
                          BeamlinePVConstants::Common::pp_maxTrackZ0Err;
    const float inv_maxTrackZ0Err = 1.f / (10.f * Z0Err);
    const float threshold =
      BeamlinePVConstants::Common::dz * inv_maxTrackZ0Err; // need something sensible that depends on binsize
    bool empty = zhisto[i] < threshold;

    uint32_t loop_mask = -1u; // relies on NBins being a multiple of 32 (warp size)
    uint32_t cur_mask = __ballot_sync(loop_mask, empty);
    uint32_t edge_mask = cur_mask ^ __funnelshift_l(prev_mask, cur_mask, 1);
    prev_mask = cur_mask;
    int index = number_of_clusteredges + __popc(edge_mask & lanemask_lt);
    bool edge = edge_mask & lanemask_eq;
    if (edge) {
      clusteredges[index] = i;
    }
    number_of_clusteredges += __popc(edge_mask);
  }
  __syncthreads();

  // Filter protoclusters with small integrals
  int outIdx = 0;
  for (int i = 0; i < number_of_clusteredges; i += 2) {
    const BinIndex ibegin = clusteredges[i];
    const BinIndex iend = clusteredges[i + 1];

    float integral = 0;
    for (int j = ibegin + threadIdx.x; j < iend; j += blockDim.x) {
      integral += zhisto[j];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
      integral += __shfl_xor_sync(-1u, integral, offset);
    }

    const float zBin = BeamlinePVConstants::Common::zmin + iend * BeamlinePVConstants::Common::dz;
    const float minInSeed = zBin < BeamlinePVConstants::Common::SMOG2_pp_separation ?
                              BeamlinePVConstants::Peak::SMOG2_minTracksInSeed :
                              BeamlinePVConstants::Peak::pp_minTracksInSeed;
    if (integral > minInSeed) {
      if (threadIdx.x == 0) {
        clusteredges[outIdx] = ibegin;
        clusteredges[outIdx + 1] = iend;
      }
      outIdx += 2;
    }
  }
  number_of_clusteredges = outIdx;
  __syncthreads();

#else
  // Fallback on non-vectorized version for targets that does not support warp level intrinsics
  bool prevempty = true;
  float integral = zhisto[0];
  for (BinIndex i = 1; i < BeamlinePVConstants::Common::Nbins; ++i) {
    const float zBin = BeamlinePVConstants::Common::zmin + i * BeamlinePVConstants::Common::dz;
    const float Z0Err = zBin < BeamlinePVConstants::Common::SMOG2_pp_separation ?
                          BeamlinePVConstants::Common::SMOG2_maxTrackZ0Err :
                          BeamlinePVConstants::Common::pp_maxTrackZ0Err;
    const float inv_maxTrackZ0Err = 1.f / (10.f * Z0Err);
    const float threshold =
      BeamlinePVConstants::Common::dz * inv_maxTrackZ0Err; // need something sensible that depends on binsize
    integral += zhisto[i];
    bool empty = zhisto[i] < threshold;
    if (empty != prevempty) {
      const float minInSeed = zBin < BeamlinePVConstants::Common::SMOG2_pp_separation ?
                                BeamlinePVConstants::Peak::SMOG2_minTracksInSeed :
                                BeamlinePVConstants::Peak::pp_minTracksInSeed;
      if (prevempty || integral > minInSeed) {
        clusteredges[number_of_clusteredges] = i;
        number_of_clusteredges++;
      }
      else
        number_of_clusteredges--;
      prevempty = empty;
      integral = 0;
    }
  }
#endif

  // Step B: turn these into clusters. There can be more than one cluster per proto-cluster.
  const size_t Nproto = number_of_clusteredges / 2;
  for (unsigned short i = 0; i < Nproto; ++i) {
    const BinIndex ibegin = clusteredges[i * 2];
    const BinIndex iend = clusteredges[i * 2 + 1];
    // find the extrema
    const float mindip =
      BeamlinePVConstants::Peak::minDipDensity * BeamlinePVConstants::Common::dz; // need to invent something
    const float minpeak = BeamlinePVConstants::Peak::minDensity * BeamlinePVConstants::Common::dz;

    Extremum extrema[PV::max_number_vertices];
    int number_of_extrema = 0;

    bool rising = true;
    float integral = zhisto[ibegin];
    extrema[number_of_extrema] = Extremum(ibegin, zhisto[ibegin], integral);
    number_of_extrema++;
    for (unsigned short i = ibegin; i < iend; ++i) {
      const auto value = zhisto[i];
      bool stillrising = zhisto[i + 1] > value;
      if (rising && !stillrising && value >= minpeak) {
        const auto n = number_of_extrema;
        if (n >= 2) {
          // check that the previous mimimum was significant. we
          // can still simplify this logic a bit.
          const auto dv1 = extrema[n - 2].value - extrema[n - 1].value;
          // const auto di1 = extrema[n-1].index - extrema[n-2].index ;
          const auto dv2 = value - extrema[n - 1].value;
          if (dv1 > mindip && dv2 > mindip) {
            extrema[number_of_extrema] = Extremum(i, value, integral + 0.5f * value);
            number_of_extrema++;
          }
          else if (dv1 > dv2) {
            number_of_extrema--;
          }
          else {
            number_of_extrema -= 2;
            extrema[number_of_extrema] = Extremum(i, value, integral + 0.5f * value);
            number_of_extrema++;
          }
        }
        else {
          extrema[number_of_extrema] = Extremum(i, value, integral + 0.5f * value);
          number_of_extrema++;
        }
      }
      else if (rising != stillrising) {
        extrema[number_of_extrema] = Extremum(i, value, integral + 0.5f * value);
        number_of_extrema++;
      }
      rising = stillrising;
      integral += value;
    }
    assert(rising == false);
    extrema[number_of_extrema] = Extremum(iend, zhisto[iend], integral);
    number_of_extrema++;

    // now partition on  extrema
    const auto N = number_of_extrema;
    Cluster subclusters[PV::max_number_subclusters];
    unsigned number_of_subclusters = 0;
    if (N > 3) {
      for (int i = 1; i < (N / 2) + 1; ++i) {
        const float z_extrema =
          BeamlinePVConstants::Common::zmin + extrema[2 * i].index * BeamlinePVConstants::Common::dz;
        const float minInSeed = z_extrema < BeamlinePVConstants::Common::SMOG2_pp_separation ?
                                  BeamlinePVConstants::Peak::SMOG2_minTracksInSeed :
                                  BeamlinePVConstants::Peak::pp_minTracksInSeed;
        if (extrema[2 * i].integral - extrema[2 * i - 2].integral > minInSeed) {
          subclusters[number_of_subclusters] =
            Cluster(extrema[2 * i - 2].index, extrema[2 * i].index, extrema[2 * i - 1].index);
          number_of_subclusters++;
        }
      }
    }
    if (number_of_subclusters == 0) {
      // FIXME: still need to get the largest maximum!
      if (extrema[1].value >= minpeak) {
        clusters[number_of_clusters] =
          Cluster(extrema[0].index, extrema[number_of_extrema - 1].index, extrema[1].index);
        number_of_clusters++;
      }
    }
    else {
      // adjust the limit of the first and last to extend to the entire protocluster
      subclusters[0].izfirst = ibegin;
      subclusters[number_of_subclusters].izlast = iend;
      for (unsigned i = 0; i < number_of_subclusters; i++) {
        Cluster subcluster = subclusters[i];
        clusters[number_of_clusters] = subcluster;
        number_of_clusters++;
      }
    }
  }

  auto zClusterMean = [&zhisto](auto izmax) -> float {
    const float* b = zhisto + izmax;
    float d1 = *b - *(b - 1);
    float d2 = *b - *(b + 1);
    float idz = d1 + d2 > 0 ? 0.5f * (d1 - d2) / (d1 + d2) : 0.0f;
    return BeamlinePVConstants::Common::zmin + BeamlinePVConstants::Common::dz * (izmax + idz + 0.5f);
  };

  for (unsigned i = threadIdx.x; i < number_of_clusters; i += blockDim.x) {
    zpeaks[i] = zClusterMean(clusters[i].izmax);
    // printf("%d %f\n", i, zpeaks[i]);
  }

  if (threadIdx.x == 0) parameters.dev_number_of_zpeaks[event_number] = number_of_clusters;
}
