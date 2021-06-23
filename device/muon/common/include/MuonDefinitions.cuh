/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "SystemOfUnits.h"
#include "States.cuh"
#include "MuonEventModel.cuh"

namespace Muon {
  namespace Constants {
    /* Detector description
       There are four stations with number of regions in it
       in current implementation regions are ignored
    */
    static constexpr unsigned n_stations = 4;
    static constexpr unsigned n_regions = 4;
    static constexpr unsigned n_quarters = 4;
    /* Cut-offs */
    static constexpr unsigned max_numhits_per_event = 600 * n_stations;

    static constexpr float SQRT3 = 1.7320508075688772;
    static constexpr float INVSQRT3 = 0.5773502691896258;
    // Multiple scattering factor 13.6 / (sqrt(6 * 17.58))
    static constexpr float MSFACTOR = 1.324200805763835;

    /*Muon Catboost model uses 5 features for each station: Delta time, Time, Crossed, X residual, Y residual*/
    static constexpr unsigned n_catboost_features = 5 * n_stations;

    // Safe margin to account for hit crossings
    static constexpr unsigned compact_hit_allocate_factor = 2;

    // Number of layouts
    static constexpr unsigned n_layouts = 2;

    /* IsMuon constants */
    namespace FoiParams {
      static constexpr unsigned n_parameters = 3;
      static constexpr unsigned a = 0;
      static constexpr unsigned b = 1;
      static constexpr unsigned c = 2;

      static constexpr unsigned n_coordinates = 2;
      static constexpr unsigned x = 0;
      static constexpr unsigned y = 1;
    } // namespace FoiParams

    static constexpr float momentum_cuts[] = {3 * Gaudi::Units::GeV, 6 * Gaudi::Units::GeV, 10 * Gaudi::Units::GeV};
    struct FieldOfInterest {
    private:
      /* FOI_x = a_x + b_x * exp(-c_x * p)
       *  FOI_y = a_y + b_y * exp(-c_y * p)
       */
      float m_factor = 1.2f;
      float m_params[Constants::n_stations * FoiParams::n_parameters * FoiParams::n_coordinates * Constants::n_regions];

    public:
      __host__ __device__ float factor() const { return m_factor; }

      __host__ __device__ void set_factor(const float factor) { m_factor = factor; }

      __host__ __device__ float
      param(const unsigned param, const unsigned coord, const unsigned station, const unsigned region) const
      {
        return m_params
          [station * FoiParams::n_parameters * FoiParams::n_coordinates * Constants::n_regions +
           param * FoiParams::n_coordinates * Constants::n_regions + coord * Constants::n_regions + region];
      }

      __host__ __device__ void set_param(
        const unsigned param,
        const unsigned coord,
        const unsigned station,
        const unsigned region,
        const float value)
      {
        m_params
          [station * FoiParams::n_parameters * FoiParams::n_coordinates * Constants::n_regions +
           param * FoiParams::n_coordinates * Constants::n_regions + coord * Constants::n_regions + region] = value;
      }

      __host__ __device__ float* params_begin() { return reinterpret_cast<float*>(m_params); }

      __host__ __device__ const float* params_begin_const() const { return reinterpret_cast<const float*>(m_params); }
    };
  } // namespace Constants
} // namespace Muon

namespace MatchUpstreamMuon {
  static constexpr float kickOffset = 338.92f * Gaudi::Units::MeV; // KickOffset
  static constexpr float kickScale = 1218.62f * Gaudi::Units::MeV; // KickScale
  static constexpr float za = 5.331f * Gaudi::Units::m;            // MagnetPlaneParA
  static constexpr float zb = -0.958f * Gaudi::Units::m;           // MagnetPlaneParB
  static constexpr float ca = 25.17f * Gaudi::Units::mm;           // MagnetCorrParA
  static constexpr float cb = -701.5f * Gaudi::Units::mm;          // MagnetCorrParB

  static constexpr float maxChi2DoF = 20.f;
  // static constexpr bool fitY = false;
  // static constexpr bool setQOverP = false;

  static constexpr int M2 {0}, M3 {1}, M4 {2}, M5 {3};
  static constexpr int VeryLowP {0}, LowP {1}, MediumP {2}, HighP {3};

  struct Hit {
    /// Build a hit from a MuonID hit
    __device__ Hit(Muon::ConstHits& hits, const unsigned& i_muonhit) :
      x(hits.x(i_muonhit)), dx2(hits.dx(i_muonhit) * hits.dx(i_muonhit) / 12), y(hits.y(i_muonhit)),
      dy2(hits.dy(i_muonhit) * hits.dy(i_muonhit) / 12), z(hits.z(i_muonhit))
    {}
    /// Build a hit extrapolating the values from a state to the given point.
    __device__ Hit(const KalmanVeloState& state, const float& pz)
    {
      const float dz = pz - state.z;

      const float dz2 = dz * dz;

      x = state.x + dz * state.tx;

      dx2 = state.c00 + dz2 * state.c22;

      y = state.y + dz * state.ty;

      dy2 = state.c11 + dz2 * state.c33;

      z = pz;
    };
    __device__ Hit() : x {0.f}, dx2 {0.f}, y {0.f}, dy2 {0.f}, z {0.f} {};
    float x, dx2, y, dy2, z;
  };

  struct MuonChambers {
    int first[4] {M3, M4, M5, M5};
    int firstOffsets[4] {0, 1, 3, 4};

    int afterKick[6] {M2, M3, M2, M4, M3, M2};
    int afterKickOffsets[4] {0, 1, 3, 6};
  };

  struct SearchWindows {
    float Windows[8] {500.f * Gaudi::Units::mm, // M2
                      400.f * Gaudi::Units::mm,

                      600.f * Gaudi::Units::mm, // M3
                      500.f * Gaudi::Units::mm,

                      700.f * Gaudi::Units::mm, // M4
                      600.f * Gaudi::Units::mm,

                      800.f * Gaudi::Units::mm, // M5
                      700.f * Gaudi::Units::mm

    };
  };

} // namespace MatchUpstreamMuon
