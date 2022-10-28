/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
\*****************************************************************************/
#include <math.h>
#include "TwoCaloClustersStandalone.cuh"
#include <ROOTHeaders.h>
#include "CaloConstants.cuh"
#include "CaloCluster.cuh"

// Explicit instantiation
INSTANTIATE_LINE(two_calo_clusters_standalone_line::two_calo_clusters_standalone_line_t, two_calo_clusters_standalone_line::Parameters)

void two_calo_clusters_standalone_line::two_calo_clusters_standalone_line_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const HostBuffers& host_buffers) const
{
  // must set_size of all output variables
  static_cast<Line const*>(this)->set_arguments_size(arguments, runtime_options, constants, host_buffers);
  set_size<typename Parameters::dev_histogram_pi0_mass_t>(arguments, 100u);


  set_size<host_ecal_twoclusters_t>(arguments, size<dev_ecal_twoclusters_t>(arguments));

  set_size<host_local_decisions_t>(arguments, get_decisions_size(arguments));

  set_size<dev_local_decisions_t>(arguments, get_decisions_size(arguments));

  set_size<host_ecal_twocluster_offsets_t>(arguments, size<dev_ecal_twocluster_offsets_t>(arguments));
}

__device__ bool two_calo_clusters_standalone_line::two_calo_clusters_standalone_line_t::select(
  const Parameters& parameters,
  std::tuple<const TwoCaloCluster&> input)
{
  const auto& dicluster = std::get<0>(input);
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];

  const float transverse_distance = sqrtf((dicluster.x1-dicluster.x2)*(dicluster.x1-dicluster.x2) + (dicluster.y1-dicluster.y2)*(dicluster.y1-dicluster.y2));

  bool decision = (dicluster.Mass > parameters.minMass) && (dicluster.Mass < parameters.maxMass) &&
                  (dicluster.Et > parameters.minEt) &&
                  (dicluster.et1 > parameters.minEt_clusters && dicluster.et2 > parameters.minEt_clusters) &&
                  (dicluster.et1 + dicluster.et2 > parameters.minSumEt_clusters) &&
                  (dicluster.CaloNeutralE19_1 > parameters.minE19_clusters &&
                   dicluster.CaloNeutralE19_2 > parameters.minE19_clusters) &&
    (transverse_distance > parameters.minTransverseDistance);

  return decision;
}

void two_calo_clusters_standalone_line::two_calo_clusters_standalone_line_t::init_monitor(
  const ArgumentReferences<Parameters>& arguments,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_local_decisions_t>(arguments, false, context);
  Allen::memset_async<dev_histogram_pi0_mass_t>(arguments, 0, context);
}

__device__ void two_calo_clusters_standalone_line::two_calo_clusters_standalone_line_t::monitor(
  const Parameters& parameters,
  std::tuple<const TwoCaloCluster> input,
  unsigned index,
  bool sel)
{
  parameters.dev_local_decisions[index] = sel;
  const auto twocalocluster = std::get<0>(input);
  if (sel) {
    const float m = twocalocluster.Mass;
    if (m > parameters.histogram_pi0_mass_min && m < parameters.histogram_pi0_mass_max) {
      const unsigned int bin = static_cast<unsigned int>(
        (m - parameters.histogram_pi0_mass_min) * parameters.histogram_pi0_mass_nbins /
        (parameters.histogram_pi0_mass_max - parameters.histogram_pi0_mass_min));
      ++parameters.dev_histogram_pi0_mass[bin];
    }
  }
}

//void two_calo_clusters_standalone_line::two_calo_clusters_standalone_line_t::output_monitor(
//  [[maybe_unused]] const ArgumentReferences<Parameters>& arguments,
//  [[maybe_unused]] const RuntimeOptions& runtime_options,
//  [[maybe_unused]] const Allen::Context& context) const
//{
//#ifdef WITH_ROOT
//  auto handler = runtime_options.root_service->handle(name());
//  // Distributions per dicluster
//  auto tree_twoclusters = handler.tree("monitor_tree_twoclusters");
//  if (tree_twoclusters == nullptr) return;
//
//  // Distributions per event
//  auto tree_evts = handler.tree("monitor_tree_evts");
//  if (tree_evts == nullptr) return;
//
//  Allen::copy<host_ecal_twocluster_offsets_t, dev_ecal_twocluster_offsets_t>(arguments, context);
//  Allen::copy<host_ecal_twoclusters_t, dev_ecal_twoclusters_t>(arguments, context);
//  Allen::copy<host_local_decisions_t, dev_local_decisions_t>(arguments, context);
//  Allen::synchronize(context);
//
//  float Mass = 0.f;
//  float Et = 0.f;
//  float Distance = 0.f;
//  float x1 = 0.f;
//  float x2 = 0.f;
//  float y1 = 0.f;
//  float y2 = 0.f;
//  float et1 = 0.f;
//  float et2 = 0.f;
//  float e19_1 = 0.f;
//  float e19_2 = 0.f;
//  unsigned num_twoclusters = 0u;
//  unsigned event_number = 0u;
//
//  handler.branch(tree_twoclusters, "Mass", Mass);
//  handler.branch(tree_twoclusters, "Et", Et);
//  handler.branch(tree_twoclusters, "Distance", Distance);
//  handler.branch(tree_twoclusters, "x1", x1);
//  handler.branch(tree_twoclusters, "x2", x2);
//  handler.branch(tree_twoclusters, "y1", y1);
//  handler.branch(tree_twoclusters, "y2", y2);
//  handler.branch(tree_twoclusters, "et1", et1);
//  handler.branch(tree_twoclusters, "et2", et2);
//  handler.branch(tree_twoclusters, "e19_1", e19_1);
//  handler.branch(tree_twoclusters, "e19_2", e19_2);
//  handler.branch(tree_twoclusters, "num_twoclusters", num_twoclusters);
//  handler.branch(tree_twoclusters, "event_number", event_number);
//
//  handler.branch(tree_evts, "num_twoclusters", num_twoclusters);
//
//  const unsigned n_events = size<host_ecal_twocluster_offsets_t>(arguments) - 1;
//
//  for (unsigned event_index = 0; event_index < n_events; event_index++) {
//    const unsigned& twoclusters_offset = (data<host_ecal_twocluster_offsets_t>(arguments) + event_index)[0];
//    num_twoclusters = (data<host_ecal_twocluster_offsets_t>(arguments) + event_index + 1)[0] - twoclusters_offset;
//    event_number = event_index;
//    tree_evts->Fill();
//
//    for (unsigned twocluster_index = 0; twocluster_index < num_twoclusters; twocluster_index++) {
//      const bool& decision = (data<host_local_decisions_t>(arguments) + twoclusters_offset + twocluster_index)[0];
//      if (decision) {
//        const auto& dicluster = (data<host_ecal_twoclusters_t>(arguments) + twoclusters_offset + twocluster_index)[0];
//
//        Mass = dicluster.Mass;
//        Distance = dicluster.Distance;
//        Et = dicluster.Et;
//        x1 = dicluster.x1;
//        x2 = dicluster.x2;
//        y1 = dicluster.y1;
//        y2 = dicluster.y2;
//        et1 = dicluster.et1;
//        et2 = dicluster.et2;
//        e19_1 = dicluster.CaloNeutralE19_1;
//        e19_2 = dicluster.CaloNeutralE19_2;
//
//        tree_twoclusters->Fill();
//      }
//    }
//  }
//#endif
//}
