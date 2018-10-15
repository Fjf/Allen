#include "../../velo/common/include/VeloDefinitions.cuh"
#include "patPV_Definitions.cuh"


__global__ void getSeeds(
    VeloState* dev_velo_states,
  int * dev_atomics_storage,
  XYZPoint * dev_seeds,
  uint * dev_number_seeds);

 __device__ int findClusters(vtxCluster * vclus, double * zclusters, int number_of_clusters);