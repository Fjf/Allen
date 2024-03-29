/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <stdint.h>
#include <cassert>
#include "States.cuh"
#include "VeloEventModel.cuh"
#include "ConsolidatedTypes.cuh"
#include "MultiEventContainer.cuh"
#include "BackendCommon.h"
#include "KinUtils.cuh"

namespace Allen {
  namespace Views {
    namespace Velo {
      namespace Consolidated {
        struct Hit {
        private:
          constexpr static unsigned offset_coordinates = sizeof(unsigned) / sizeof(half_t);

          const half_t* m_base_pointer = nullptr;
          unsigned m_index = 0;
          unsigned m_total_number_of_hits = 0;

        public:
          Hit() = default;

          __host__ __device__
          Hit(const half_t* base_pointer, const unsigned index, const unsigned total_number_of_hits) :
            m_base_pointer(base_pointer),
            m_index(index), m_total_number_of_hits(total_number_of_hits)
          {}

          __host__ __device__ unsigned id() const { return reinterpret_cast<const unsigned*>(m_base_pointer)[m_index]; }

          __host__ __device__ float x() const
          {
            return static_cast<float>(m_base_pointer[offset_coordinates * m_total_number_of_hits + 3 * m_index]);
          }

          __host__ __device__ float y() const
          {
            return static_cast<float>(m_base_pointer[offset_coordinates * m_total_number_of_hits + 3 * m_index + 1]);
          }

          __host__ __device__ float z() const
          {
            return static_cast<float>(m_base_pointer[offset_coordinates * m_total_number_of_hits + 3 * m_index + 2]);
          }

          __host__ __device__ operator ::Velo::HitBase() const { return ::Velo::HitBase {x(), y(), z()}; }

          __host__ __device__ operator ::Velo::Hit() const { return ::Velo::Hit {x(), y(), z(), id()}; }
        };

        struct Hits {
        private:
          const half_t* m_base_pointer = nullptr;
          unsigned m_offset = 0;
          unsigned m_size = 0;
          unsigned m_total_number_of_hits = 0;

        public:
          Hits() = default;

          __host__ __device__ Hits(
            const char* base_pointer,
            const unsigned* offset_tracks,
            const unsigned* offset_track_hit_number,
            const unsigned event_number,
            const unsigned number_of_events) :
            m_base_pointer(reinterpret_cast<const half_t*>(base_pointer)),
            m_offset(offset_track_hit_number[offset_tracks[event_number]]),
            m_size(
              offset_track_hit_number[offset_tracks[event_number + 1]] -
              offset_track_hit_number[offset_tracks[event_number]]),
            m_total_number_of_hits(offset_track_hit_number[offset_tracks[number_of_events]])
          {}

          __host__ __device__ unsigned size() const { return m_size; }

          __host__ __device__ Hit hit(const unsigned index) const
          {
            assert(index < m_size);
            return Hit {m_base_pointer, m_offset + index, m_total_number_of_hits};
          }

          /**
           * @brief This offset indicates the relative position of the
           *        hits in the container for the current event.
           */
          __host__ __device__ unsigned offset() const { return m_offset; }
        };

        // TODO: Just replace this with Allen::Views::Physics::KalmanState.
        struct State {
        private:
          constexpr static unsigned nb_elements_state = 6;
          constexpr static unsigned nb_elements_cov = 8;

          const float* m_base_pointer = nullptr;
          unsigned m_index = 0;
          unsigned m_total_number_of_tracks = 0;

        public:
          State() = default;

          __host__ __device__
          State(const char* base_pointer, const unsigned index, const unsigned total_number_of_tracks) :
            m_base_pointer(reinterpret_cast<const float*>(base_pointer)),
            m_index(index), m_total_number_of_tracks(total_number_of_tracks)
          {}

          __host__ __device__ float x() const { return m_base_pointer[nb_elements_state * m_index]; }

          __host__ __device__ float y() const { return m_base_pointer[nb_elements_state * m_index + 1]; }

          __host__ __device__ float z() const { return m_base_pointer[nb_elements_state * m_index + 2]; }

          __host__ __device__ float tx() const { return m_base_pointer[nb_elements_state * m_index + 3]; }

          __host__ __device__ float ty() const { return m_base_pointer[nb_elements_state * m_index + 4]; }

          __host__ __device__ float c00() const
          {
            return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * m_index];
          }

          __host__ __device__ float c20() const
          {
            return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * m_index + 1];
          }

          __host__ __device__ float c22() const
          {
            return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * m_index + 2];
          }

          __host__ __device__ float c11() const
          {
            return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * m_index + 3];
          }

          __host__ __device__ float c31() const
          {
            return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * m_index + 4];
          }

          __host__ __device__ float c33() const
          {
            return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * m_index + 5];
          }

          __host__ __device__ operator MiniState() const { return MiniState {x(), y(), z(), tx(), ty()}; }

          __host__ __device__ operator KalmanVeloState() const
          {
            return KalmanVeloState {x(), y(), z(), tx(), ty(), c00(), c20(), c22(), c11(), c31(), c33()};
          }
        };

        struct States {
        private:
          const char* m_base_pointer = nullptr;
          unsigned m_offset = 0;
          unsigned m_size = 0;
          unsigned m_total_number_of_tracks = 0;

        public:
          States() = default;

          __host__ __device__ States(
            const char* base_pointer,
            const unsigned* offset_tracks,
            const unsigned event_number,
            const unsigned number_of_events) :
            m_base_pointer(base_pointer),
            m_offset(offset_tracks[event_number]),
            m_size(offset_tracks[event_number + 1] - offset_tracks[event_number]),
            m_total_number_of_tracks(offset_tracks[number_of_events])
          {}

          __host__ __device__ unsigned size() const { return m_size; }

          __host__ __device__ State state(const unsigned track_index) const
          {
            assert(track_index < m_size);
            return State {m_base_pointer, m_offset + track_index, m_total_number_of_tracks};
          }

          /**
           * @brief This offset indicates the relative position of the
           *        states in the container for the current event.
           */
          __host__ __device__ unsigned offset() const { return m_offset; }
        };

        struct Track : Allen::ILHCbIDSequence<Track> {
          friend Allen::ILHCbIDSequence<Track>;

        private:
          const Hits* m_hits = nullptr;
          unsigned m_track_index = 0;
          unsigned m_track_container_offset = 0;
          unsigned m_offset = 0;
          unsigned m_number_of_hits = 0;

          __host__ __device__ unsigned number_of_ids_impl() const { return m_number_of_hits; }

          __host__ __device__ unsigned id_impl(const unsigned index) const { return hit(index).id(); }

        public:
          Track() = default;

          __host__ __device__ Track(
            const Hits* hits,
            const unsigned* offset_tracks,
            const unsigned* offset_track_hit_number,
            const unsigned track_index,
            const unsigned event_number) :
            m_hits(hits + event_number),
            m_track_index(track_index)
          {
            m_track_container_offset = offset_tracks[event_number];
            const auto offset_event = offset_track_hit_number + m_track_container_offset;
            m_offset = offset_event[track_index] - offset_event[0];
            m_number_of_hits = offset_event[track_index + 1] - offset_event[track_index];
          }

          __host__ __device__ unsigned track_index() const { return m_track_index; }

          __host__ __device__ unsigned track_container_offset() const { return m_track_container_offset; }

          __host__ __device__ unsigned number_of_hits() const { return m_number_of_hits; }

          __host__ __device__ Hit hit(const unsigned index) const
          {
            assert(m_hits != nullptr);
            assert(index < m_number_of_hits);
            return m_hits->hit(m_offset + index);
          }

          __host__ __device__ KalmanVeloState state(const Allen::Views::Physics::KalmanStates& states_view) const
          {
            return static_cast<KalmanVeloState>(states_view.state(m_track_index));
          }

          __host__ __device__ float eta(const Allen::Views::Physics::KalmanStates& states_view, bool backward) const
          {
            const auto tx = state(states_view).tx;
            const auto ty = state(states_view).ty;
            return eta_from_rho_z(std::sqrt(tx * tx + ty * ty), backward ? -1.f : 1.f);
          }
        };

        struct Tracks : Allen::ILHCbIDContainer<Tracks> {
          friend Allen::ILHCbIDContainer<Tracks>;
          constexpr static auto TypeID = Allen::TypeIDs::VeloTracks;

        private:
          const Track* m_track;
          unsigned m_size = 0;
          unsigned m_offset = 0;

          __host__ __device__ unsigned number_of_id_sequences_impl() const { return m_size; }

          __host__ __device__ const Track& id_sequence_impl(const unsigned index) const
          {
            assert(index < number_of_id_sequences_impl());
            return m_track[index];
          }

        public:
          Tracks() = default;

          __host__ __device__ Tracks(const Track* track, const unsigned* offset_tracks, const unsigned event_number) :
            m_track(track + offset_tracks[event_number]),
            m_size(offset_tracks[event_number + 1] - offset_tracks[event_number]), m_offset(offset_tracks[event_number])
          {}

          __host__ __device__ unsigned size() const { return m_size; }

          __host__ __device__ const Track& track(const unsigned index) const
          {
            assert(index < size());
            return m_track[index];
          }

          /**
           * @brief This offset indicates the relative position of the
           *        tracks in the container for the current event.
           */
          __host__ __device__ unsigned offset() const { return m_offset; }
        };

        using MultiEventTracks = Allen::MultiEventContainer<Tracks>;
      } // namespace Consolidated
    }   // namespace Velo
  }     // namespace Views
} // namespace Allen

// Deprecated
namespace Velo {
  namespace Consolidated {
    /**
     * @brief Structure to access VELO hits.
     */
    template<typename T>
    struct Hits_t : Velo::Clusters_t<T> {
      using Velo::Clusters_t<T>::m_base_pointer;
      using Velo::Clusters_t<T>::m_total_number_of_hits;
      using Velo::Clusters_t<T>::m_offset;

      __host__ __device__ Hits_t(T* base_pointer, const unsigned offset, const unsigned total_number_of_hits) :
        Velo::Clusters_t<T>(base_pointer, total_number_of_hits, offset)
      {}

      __host__ __device__ Hits_t(const Hits_t<T>& hits) : Velo::Clusters_t<T>(hits) {}

      __host__ __device__ void set(const unsigned index, const ::Velo::Hit& hit)
      {
        assert(m_offset + index < m_total_number_of_hits);
        this->set_x(index, hit.x);
        this->set_y(index, hit.y);
        this->set_z(index, hit.z);
        this->set_id(index, hit.LHCbID);
      }

      __host__ __device__ ::Velo::Hit get(const unsigned index) const
      {
        assert(m_offset + index < m_total_number_of_hits);
        return ::Velo::Hit {this->x(index), this->y(index), this->z(index), this->id(index)};
      }
    };

    typedef const Hits_t<const char> ConstHits;
    typedef Hits_t<char> Hits;

    struct Tracks : public ::Consolidated::Tracks {
      __host__ __device__ Tracks(
        const unsigned* atomics_base_pointer,
        const unsigned* track_hit_number_base_pointer,
        const unsigned current_event_number,
        const unsigned number_of_events) :
        ::Consolidated::Tracks(
          atomics_base_pointer,
          track_hit_number_base_pointer,
          current_event_number,
          number_of_events)
      {}

      __host__ __device__ ConstHits get_hits(const char* hits_base_pointer, const unsigned track_number) const
      {
        return ConstHits {hits_base_pointer, track_offset(track_number), m_total_number_of_hits};
      }

      __host__ __device__ Hits get_hits(char* hits_base_pointer, const unsigned track_number)
      {
        return Hits {hits_base_pointer, track_offset(track_number), m_total_number_of_hits};
      }

      __host__ std::vector<unsigned> get_lhcbids_for_track(const char* hits_base_pointer, const unsigned track_number)
        const
      {
        std::vector<unsigned> ids;
        const auto hits = ConstHits {hits_base_pointer, track_offset(track_number), m_total_number_of_hits};
        for (unsigned i = 0; i < number_of_hits(track_number); ++i) {
          ids.push_back(hits.id(i));
        }
        return ids;
      }
    };

    typedef const Tracks ConstTracks;

    // TODO: Use the same states for Kalman filter and VELO.
    /**
     * @brief States data structure.
     * @detail An SOA of two AOS is used:
     *         SOA{AOS{x, y, z, tx, ty, qop}, AOS{c00, c20, c22, c11, c31, c33, chi2, ndof}}
     *
     * @tparam T
     */
    template<typename T>
    struct States_t {
    private:
      Allen::forward_type_t<T, float>* m_base_pointer;
      const unsigned m_total_number_of_tracks;
      const unsigned m_offset;

    public:
      constexpr static unsigned nb_elements_state = 6;
      constexpr static unsigned nb_elements_cov = 8;
      constexpr static unsigned size = (nb_elements_state + nb_elements_cov) * sizeof(uint32_t);

      __host__ __device__ States_t(T* base_pointer, const unsigned total_number_of_tracks, const unsigned offset = 0) :
        m_base_pointer(reinterpret_cast<Allen::forward_type_t<T, float>*>(base_pointer)),
        m_total_number_of_tracks(total_number_of_tracks), m_offset(offset)
      {}

      __host__ __device__ States_t(const States_t<T>& states) :
        m_base_pointer(states.m_base_pointer), m_total_number_of_tracks(states.m_total_number_of_tracks)
      {}

      // Accessors and lvalue references for all types
      __host__ __device__ float x(const unsigned index) const
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * (m_offset + index)];
      }

      __host__ __device__ float& x(const unsigned index)
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * (m_offset + index)];
      }

      __host__ __device__ float y(const unsigned index) const
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * (m_offset + index) + 1];
      }

      __host__ __device__ float& y(const unsigned index)
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * (m_offset + index) + 1];
      }

      __host__ __device__ float z(const unsigned index) const
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * (m_offset + index) + 2];
      }

      __host__ __device__ float& z(const unsigned index)
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * (m_offset + index) + 2];
      }

      __host__ __device__ float tx(const unsigned index) const
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * (m_offset + index) + 3];
      }

      __host__ __device__ float& tx(const unsigned index)
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * (m_offset + index) + 3];
      }

      __host__ __device__ float ty(const unsigned index) const
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * (m_offset + index) + 4];
      }

      __host__ __device__ float& ty(const unsigned index)
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * (m_offset + index) + 4];
      }

      __host__ __device__ float qop(const unsigned index) const
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * (m_offset + index) + 5];
      }

      __host__ __device__ float& qop(const unsigned index)
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * (m_offset + index) + 5];
      }

      __host__ __device__ float c00(const unsigned index) const
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * (m_offset + index)];
      }

      __host__ __device__ float& c00(const unsigned index)
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * (m_offset + index)];
      }

      __host__ __device__ float c20(const unsigned index) const
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * (m_offset + index) + 1];
      }

      __host__ __device__ float& c20(const unsigned index)
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * (m_offset + index) + 1];
      }

      __host__ __device__ float c22(const unsigned index) const
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * (m_offset + index) + 2];
      }

      __host__ __device__ float& c22(const unsigned index)
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * (m_offset + index) + 2];
      }

      __host__ __device__ float c11(const unsigned index) const
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * (m_offset + index) + 3];
      }

      __host__ __device__ float& c11(const unsigned index)
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * (m_offset + index) + 3];
      }

      __host__ __device__ float c31(const unsigned index) const
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * (m_offset + index) + 4];
      }

      __host__ __device__ float& c31(const unsigned index)
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * (m_offset + index) + 4];
      }

      __host__ __device__ float c33(const unsigned index) const
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * (m_offset + index) + 5];
      }

      __host__ __device__ float& c33(const unsigned index)
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * (m_offset + index) + 5];
      }

      __host__ __device__ float chi2(const unsigned index) const
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * (m_offset + index) + 6];
      }

      __host__ __device__ float& chi2(const unsigned index)
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return m_base_pointer[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * (m_offset + index) + 6];
      }

      __host__ __device__ unsigned ndof(const unsigned index) const
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return reinterpret_cast<const unsigned*>(
          m_base_pointer)[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * (m_offset + index) + 7];
      }

      __host__ __device__ unsigned& ndof(const unsigned index)
      {
        assert(m_offset + index < m_total_number_of_tracks);
        return reinterpret_cast<unsigned*>(
          m_base_pointer)[nb_elements_state * m_total_number_of_tracks + nb_elements_cov * (m_offset + index) + 7];
      }

      __host__ __device__ void set(const unsigned track_number, const KalmanVeloState& state)
      {
        assert(track_number < m_total_number_of_tracks);

        x(track_number) = state.x;
        y(track_number) = state.y;
        z(track_number) = state.z;
        tx(track_number) = state.tx;
        ty(track_number) = state.ty;

        c00(track_number) = state.c00;
        c20(track_number) = state.c20;
        c22(track_number) = state.c22;
        c11(track_number) = state.c11;
        c31(track_number) = state.c31;
        c33(track_number) = state.c33;
      }

      __host__ __device__ void set(const unsigned track_number, const MiniState& state)
      {
        assert(track_number < m_total_number_of_tracks);

        x(track_number) = state.x;
        y(track_number) = state.y;
        z(track_number) = state.z;
        tx(track_number) = state.tx;
        ty(track_number) = state.ty;

        c00(track_number) = 0.f;
        c20(track_number) = 0.f;
        c22(track_number) = 0.f;
        c11(track_number) = 0.f;
        c31(track_number) = 0.f;
        c33(track_number) = 0.f;
      }

      __host__ __device__ MiniState get(const uint track_number) const
      {
        return MiniState {x(track_number), y(track_number), z(track_number), tx(track_number), ty(track_number)};
      }

      __host__ __device__ KalmanVeloState get_kalman_state(const unsigned track_number) const
      {
        assert(track_number < m_total_number_of_tracks);

        KalmanVeloState state;

        state.x = x(track_number);
        state.y = y(track_number);
        state.z = z(track_number);
        state.tx = tx(track_number);
        state.ty = ty(track_number);

        state.c00 = c00(track_number);
        state.c20 = c20(track_number);
        state.c22 = c22(track_number);
        state.c11 = c11(track_number);
        state.c31 = c31(track_number);
        state.c33 = c33(track_number);

        return state;
      }
    };

    typedef const States_t<const char> ConstStates;
    typedef States_t<char> States;
  } // namespace Consolidated
} // namespace Velo
