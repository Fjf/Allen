#ifndef  INDEXEDHITCONTAINER_H
#define  INDEXEDHITCONTAINER_H 1

#include <assert.h>
#include <vector>
#include <algorithm>

template <size_t NStations, size_t NLayersPerStation, size_t NRegionsPerLayer, typename Hit>
class IndexedHitContainer {

  typedef unsigned short offset_t;

  std::vector<Hit>  m_hits;
  offset_t          m_r_offsets[NStations][NLayersPerStation][NRegionsPerLayer+1];
  offset_t          m_l_offsets[NStations][NLayersPerStation+1];
  offset_t          m_s_offsets[NStations+1];

  inline std::pair<size_t,size_t> offset( unsigned station ) const {
    return { m_s_offsets[station], m_s_offsets[station+1] };
  }
  inline std::pair<size_t,size_t> offset( unsigned station, unsigned layer ) const {
    auto i = m_s_offsets[station] ;
    return { i + m_l_offsets[station][layer], i + m_l_offsets[station][layer+1] };
  }
  inline std::pair<size_t,size_t> offset( unsigned station, unsigned layer, unsigned region ) const {
    auto i = m_s_offsets[station] + m_l_offsets[station][layer];
    return { i + m_r_offsets[station][layer][region], i + m_r_offsets[station][layer][region+1] };
  }
  inline void clearOffsets() {
    std::fill( std::begin(m_s_offsets),       std::end(m_s_offsets),                                   0u );
    std::fill( std::begin(m_l_offsets[0]),    std::end(m_l_offsets[NStations-1]),                      0u );
    std::fill( std::begin(m_r_offsets[0][0]), std::end(m_r_offsets[NStations-1][NLayersPerStation-1]), 0u );
  }

public:
  IndexedHitContainer( size_t reserve = 0 )
  {
    if (reserve!=0) m_hits.reserve( reserve );
    clearOffsets();
  }

  void reserve( size_t capacity )
  {
      m_hits.reserve(capacity);
  }

public:

  typedef typename std::vector<Hit>::iterator iterator;
  typedef typename std::vector<Hit>::const_iterator const_iterator;
  typedef typename std::vector<Hit>::const_reverse_iterator const_reverse_iterator;

  typedef const Hit& const_reference;
  typedef Hit value_type;

  typedef Gaudi::Range_<IndexedHitContainer<NStations,
                                            NLayersPerStation,
                                            NRegionsPerLayer,
                                            Hit> > HitRange;
  void clear() {
    m_hits.clear();
    clearOffsets();
  }

  std::pair<iterator,iterator> range_( unsigned station, unsigned layer, unsigned region ) {
    auto rng = offset( station, layer, region );
    return { next( begin(m_hits), rng.first  ),
             next( begin(m_hits), rng.second ) };
  }

  std::pair<iterator,iterator> range_( unsigned station, unsigned layer ) {
    auto rng = offset( station, layer );
    return { next( begin(m_hits), rng.first  ),
             next( begin(m_hits), rng.second ) };
  }

  HitRange range( unsigned station, unsigned layer, unsigned region ) const {
    auto rng = offset( station, layer, region );
    return { next( begin(m_hits), rng.first  ),
             next( begin(m_hits), rng.second ) };
  }
  HitRange range( unsigned station, unsigned layer ) const {
    auto rng = offset( station, layer );
    return { next( begin(m_hits), rng.first  ),
             next( begin(m_hits), rng.second ) };
  }
  HitRange range( unsigned station  ) const {
    auto rng = offset( station );
    return { next( begin(m_hits), rng.first  ),
             next( begin(m_hits), rng.second ) };
  }
  HitRange range( ) const {
    return { begin(m_hits), std::end(m_hits) };
  }

  size_t size( unsigned station )                                  const { return m_s_offsets[station+1] - m_s_offsets[station]; }
  size_t size( unsigned station, unsigned layer )                  const { return m_l_offsets[station][layer+1]-m_l_offsets[station][layer]; }
  size_t size( unsigned station, unsigned layer, unsigned region ) const { return m_r_offsets[station][layer][region+1] - m_r_offsets[station][layer][region] ; }

  bool empty( unsigned station )                                const { return  m_s_offsets[station+1] == m_s_offsets[station]; }
  bool empty( unsigned station, unsigned layer )                const { return  m_l_offsets[station][layer+1] == m_l_offsets[station][layer] ; }
  bool empty( unsigned station, unsigned layer, size_t region ) const { return  m_r_offsets[station][layer][region+1] == m_r_offsets[station][layer][region] ; }

  // empirically, in Brunel ALL inserts happen at the 'end' of m_hits...
  template <typename I>
  iterator insert( unsigned station, unsigned layer, unsigned region, I&& b, I&& e ) {
    auto n = std::distance(b,e);
    assert( m_hits.size() + n < std::numeric_limits<offset_t>::max()  );
    auto it = m_hits.insert( next( begin(m_hits),
                                   m_s_offsets[station] + m_l_offsets[station][layer] + m_r_offsets[station][layer][region]) ,
                             std::forward<I>(b), std::forward<I>(e) );
    // adjust everything 'behind' the insertion point...
    auto add_n = [n](offset_t& offset) { offset+=n; };
    // subsequent stations
    std::for_each( std::next(m_s_offsets,station+1), std::end(m_s_offsets), add_n );
    // subsequent layers in current station
    std::for_each( std::next(m_l_offsets[station],layer+1), std::end(m_l_offsets[station]), add_n );
    // subsequent regions in current layer
    std::for_each( std::next(m_r_offsets[station][layer],region+1), std::end(m_r_offsets[station][layer]), add_n );
    return it;
  }

  template <typename Range>
  void insert( unsigned station, unsigned layer, unsigned region, const Range& range) {
      insert(station,layer,region,begin(range),end(range));
  }

  // -- This is if we don't want to split in regions as well (the Run II PatReco does not use the regions)
  template <typename I>
  void insert( unsigned station, unsigned layer, I&& b, I&& e ) {
    auto n = std::distance(b,e);
    m_hits.insert( std::next( begin(m_hits),
                              m_s_offsets[station] + m_l_offsets[station][layer]) ,
                   std::forward<I>(b), std::forward<I>(e) );
    // adjust everything 'behind' the insertion point...
    auto add_n = [n](offset_t& offset) { offset+=n; };
    // subsequent stations
    std::for_each( std::next(m_s_offsets,station+1), std::end(m_s_offsets), add_n );
    // subsequent layers in current station
    std::for_each( std::next(m_l_offsets[station],layer+1), std::end(m_l_offsets[station]), add_n );
  }


};
#endif
