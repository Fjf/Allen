#include "TwoCaloClustersStandalone.cuh"

#ifndef ALLEN_STANDALONE
#include "Gaudi/Accumulators/Histogram.h"

template<int I>
using gaudi_histo_t = Gaudi::Accumulators::Histogram<I, Gaudi::Accumulators::atomicity::full, double>;
#endif

void two_calo_clusters_standalone_line::two_calo_clusters_standalone_line_t::init()
{
  Line<two_calo_clusters_standalone_line::two_calo_clusters_standalone_line_t, two_calo_clusters_standalone_line::Parameters>::init();
#ifndef ALLEN_STANDALONE
  histogram_pi0_mass = (char*) new gaudi_histo_t<1>(
      this,
      "pi0_mass",
      "m(gg)",
      Gaudi::Accumulators::Axis<double> {property<histogram_pi0_mass_nbins_t>(),
                                        property<histogram_pi0_mass_min_t>(),
                                        property<histogram_pi0_mass_max_t>()});
#endif
}

void two_calo_clusters_standalone_line::two_calo_clusters_standalone_line_t::output_monitor(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Allen::Context& context) const
{
  const auto dev_histogram_pi0_mass = make_host_buffer<Parameters::dev_histogram_pi0_mass_t>(arguments, context);

#ifndef ALLEN_STANDALONE
  Allen::synchronize(context);
  float binWidth = (property<histogram_pi0_mass_max_t>() - property<histogram_pi0_mass_min_t>()) /
                   property<histogram_pi0_mass_nbins_t>();
  auto mass_buffer = reinterpret_cast<gaudi_histo_t<1>*>(histogram_pi0_mass)->buffer();
  for (auto i = 0u; i < property<histogram_pi0_mass_nbins_t>(); ++i) {
    mass_buffer[property<histogram_pi0_mass_min_t>() + i * binWidth] += *(dev_histogram_pi0_mass.data() + i);
  }
#endif
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
}
