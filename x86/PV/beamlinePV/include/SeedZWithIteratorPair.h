#include <algorithm>

struct SeedZWithIteratorPair {
  using iterator = std::vector<PVTrack>::iterator;
  float z;
  iterator begin;
  iterator end;
  __host__ __device__ SeedZWithIteratorPair(float _z, iterator _begin, iterator _end) :
    z {_z}, begin {_begin}, end {_end} {};
  __host__ __device__ SeedZWithIteratorPair() {};

  PVTrack* get_array() const
  {
    std::vector<PVTrack> track_vec(begin, end);
    return track_vec.data();
  };

  uint get_size() const
  {
    std::vector<PVTrack> track_vec(begin, end);
    return track_vec.size();
  };
};