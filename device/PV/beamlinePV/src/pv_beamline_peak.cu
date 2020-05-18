#include "pv_beamline_peak.cuh"

void pv_beamline_peak::pv_beamline_peak_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_zpeaks_t>(arguments, first<host_number_of_selected_events_t>(arguments) * PV::max_number_vertices);
  set_size<dev_number_of_zpeaks_t>(arguments, first<host_number_of_selected_events_t>(arguments));
}

void pv_beamline_peak::pv_beamline_peak_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  HostBuffers&,
  cudaStream_t& cuda_stream,
  cudaEvent_t&) const
{
  const auto grid_dim = dim3(
    (first<host_number_of_selected_events_t>(arguments) + PV::num_threads_pv_beamline_peak_t - 1) /
    PV::num_threads_pv_beamline_peak_t);

  device_function(pv_beamline_peak)(grid_dim, PV::num_threads_pv_beamline_peak_t, cuda_stream)(
    arguments, first<host_number_of_selected_events_t>(arguments));
}

__global__ void pv_beamline_peak::pv_beamline_peak(pv_beamline_peak::Parameters parameters, const uint number_of_events)
{
  // At least parallelize over events, even if it's
  // one event on each thread
  for (auto event_number = blockIdx.x * blockDim.x + threadIdx.x; event_number < number_of_events;
       event_number += blockDim.x * gridDim.x) {
    const float* zhisto = parameters.dev_zhisto + BeamlinePVConstants::Common::Nbins * event_number;
    float* zpeaks = parameters.dev_zpeaks + PV::max_number_vertices * event_number;
    uint number_of_peaks = 0;

    Cluster clusters[PV::max_number_of_clusters];
    uint number_of_clusters = 0;
    using BinIndex = unsigned short;
    BinIndex clusteredges[PV::max_number_clusteredges];
    uint number_of_clusteredges = 0;

    {
      const float inv_maxTrackZ0Err = 1.f / (10.f * BeamlinePVConstants::Common::maxTrackZ0Err);
      const float threshold =
        BeamlinePVConstants::Common::dz * inv_maxTrackZ0Err; // need something sensible that depends on binsize
      bool prevempty = true;
      float integral = zhisto[0];
      for (BinIndex i = 1; i < BeamlinePVConstants::Common::Nbins; ++i) {
        integral += zhisto[i];
        bool empty = zhisto[i] < threshold;
        if (empty != prevempty) {
          if (prevempty || integral > BeamlinePVConstants::Peak::minTracksInSeed) {
            clusteredges[number_of_clusteredges] = i;
            number_of_clusteredges++;
          }
          else
            number_of_clusteredges--;
          prevempty = empty;
          integral = 0;
        }
      }

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
        {
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
                  if (number_of_extrema < 0) number_of_extrema = 0;
                }
                else {
                  number_of_extrema--;
                  number_of_extrema--;
                  if (number_of_extrema < 0) number_of_extrema = 0;
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
        }
        // now partition on  extrema
        const auto N = number_of_extrema;
        Cluster subclusters[PV::max_number_subclusters];
        uint number_of_subclusters = 0;
        if (N > 3) {
          for (int i = 1; i < (N / 2) + 1; ++i) {
            if (extrema[2 * i].integral - extrema[2 * i - 2].integral > BeamlinePVConstants::Peak::minTracksInSeed) {
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
          for (uint i = 0; i < number_of_subclusters; i++) {
            Cluster subcluster = subclusters[i];
            clusters[number_of_clusters] = subcluster;
            number_of_clusters++;
          }
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

    for (uint i = 0; i < number_of_clusters; ++i) {
      zpeaks[number_of_peaks] = zClusterMean(clusters[i].izmax);
      number_of_peaks++;
    }

    parameters.dev_number_of_zpeaks[event_number] = number_of_peaks;
  }
}