#include "HandleConsolidateTracks.cuh"
#include "../../../cuda/velo/consolidate_tracks/include/ConsolidateTracks.cuh"

void ConsolidateTracks::operator()() {
  consolidate_tracks<mc_check_enabled> <<<num_blocks, num_threads, 0, *stream>>>(
    dev_atomics_storage,
    dev_tracks,
    dev_output_tracks,
    dev_velo_cluster_container,
    dev_module_cluster_start,
    dev_module_cluster_num,
    dev_velo_states
  );
}
