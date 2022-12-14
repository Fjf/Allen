/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include "Common.h"
#include "BackendCommon.h"
#include <fstream>
#include <stdio.h>

namespace ParKalmanFilter {

  template<int Degx1, int Degx2, int Degy1, int Degy2>
  struct KalmanParametrizationsCoef {
    float coefs[2 * Degx2 + 4 * Degx1 + 2 * Degy2 + 4 * Degy1];
    int degx1 = Degx1, degx2 = Degx2, degy1 = Degy1, degy2 = Degy2;

    // Read parameters.
    //__host__ int Read(std::istream &inFile, int degx1, int degx2, int degy1, int degy2);
    __host__ int Read(std::istream& inFile)
    {
      if (degx1 > 10 || degx2 > 10 || degy1 > 10 || degy2 > 10)
        throw StrException(
          "You have to increase the size of the internal arrays of the KalmanParametrizationsCoef class");

      // I guess we'll try a lambda?
      auto GetPar = [](std::istream& ss) -> float {
        std::string s;
        ss >> s;
        return std::stof(s);
      };

      for (int i = 0; i < degx2; i++) {
        this->x00(i) = GetPar(inFile);
      }
      for (int i = 0; i < degx2; i++) {
        this->tx00(i) = 1e-3f * GetPar(inFile);
      }
      for (int i = 0; i < degx1; i++) {
        this->x10(i) = GetPar(inFile);
      }
      for (int i = 0; i < degx1; i++) {
        this->x01(i) = GetPar(inFile);
      }
      for (int i = 0; i < degx1; i++) {
        this->tx10(i) = 1e-3f * GetPar(inFile);
      }
      for (int i = 0; i < degx1; i++) {
        this->tx01(i) = 1e-3f * GetPar(inFile);
      }

      for (int i = 0; i < degy2; i++) {
        this->y00(i) = GetPar(inFile);
      }
      for (int i = 0; i < degy2; i++) {
        this->ty00(i) = 1e-3f * GetPar(inFile);
      }
      for (int i = 0; i < degy1; i++) {
        this->y10(i) = GetPar(inFile);
      }
      for (int i = 0; i < degy1; i++) {
        this->y01(i) = GetPar(inFile);
      }
      for (int i = 0; i < degy1; i++) {
        this->ty10(i) = 1e-3f * GetPar(inFile);
      }
      for (int i = 0; i < degy1; i++) {
        this->ty01(i) = 1e-3f * GetPar(inFile);
      }
      // if(feof(inFile)) return 0;
      return 1;
    }

    __device__ __host__ inline float& x00(int idx) { return coefs[idx]; }
    __device__ __host__ inline float& x10(int idx) { return coefs[Degx2 + idx]; }
    __device__ __host__ inline float& x01(int idx) { return coefs[Degx1 + Degx2 + idx]; }
    __device__ __host__ inline float& tx00(int idx) { return coefs[2 * Degx1 + Degx2 + idx]; }
    __device__ __host__ inline float& tx10(int idx) { return coefs[2 * Degx1 + 2 * Degx2 + idx]; }
    __device__ __host__ inline float& tx01(int idx) { return coefs[3 * Degx1 + 2 * Degx2 + idx]; }
    __device__ __host__ inline float& y00(int idx) { return coefs[4 * Degx1 + 2 * Degx2 + idx]; }
    __device__ __host__ inline float& y10(int idx) { return coefs[4 * Degx1 + 2 * Degx2 + Degy2 + idx]; }
    __device__ __host__ inline float& y01(int idx) { return coefs[4 * Degx1 + 2 * Degx2 + Degy1 + Degy2 + idx]; }
    __device__ __host__ inline float& ty00(int idx) { return coefs[4 * Degx1 + 2 * Degx2 + 2 * Degy1 + Degy2 + idx]; }
    __device__ __host__ inline float& ty10(int idx)
    {
      return coefs[4 * Degx1 + 2 * Degx2 + 2 * Degy1 + 2 * Degy2 + idx];
    }
    __device__ __host__ inline float& ty01(int idx)
    {
      return coefs[4 * Degx1 + 2 * Degx2 + 3 * Degy1 + 2 * Degy2 + idx];
    }
  };

  // Operators for Kalman coefficients.
  typedef KalmanParametrizationsCoef<7, 9, 5, 7> StandardCoefs;

  __device__ __host__ inline StandardCoefs operator+(const StandardCoefs& a, const StandardCoefs& b);
  __device__ __host__ inline StandardCoefs operator-(const StandardCoefs& a, const StandardCoefs& b);
  __device__ __host__ inline StandardCoefs operator*(const StandardCoefs& a, const float p);

} // namespace ParKalmanFilter

namespace ParKalmanFilter {

  __device__ __host__ StandardCoefs operator+(const StandardCoefs& a, const StandardCoefs& b)
  {
    StandardCoefs c;
    int nMax = 4 * a.degx1 + 2 * a.degx2 + 4 * a.degy1 + 2 * a.degy2;
    for (int i = 0; i < nMax; i++) {
      c.coefs[i] = a.coefs[i] + b.coefs[i];
    }
    return c;
  }

  __device__ __host__ StandardCoefs operator-(const StandardCoefs& a, const StandardCoefs& b)
  {
    StandardCoefs c;
    int nMax = 4 * a.degx1 + 2 * a.degx2 + 4 * a.degy1 + 2 * a.degy2;
    for (int i = 0; i < nMax; i++) {
      c.coefs[i] = a.coefs[i] - b.coefs[i];
    }
    return c;
  }

  __device__ __host__ StandardCoefs operator*(const StandardCoefs& a, const float p)
  {
    StandardCoefs c;
    int nMax = 4 * a.degx1 + 2 * a.degx2 + 4 * a.degy1 + 2 * a.degy2;
    for (int i = 0; i < nMax; i++) {
      c.coefs[i] = p * a.coefs[i];
    }
    return c;
  }

} // namespace ParKalmanFilter
