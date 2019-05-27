#include "MuonRawToHits.cuh"
#include <cstdio>

namespace Muon {
  __device__ Hit::Hit(HitsSoA* hitsSoA, uint index) {
    tile = hitsSoA->tile[index];
    x = hitsSoA->x[index];
    dx = hitsSoA->dx[index];
    y = hitsSoA->y[index];
    dy = hitsSoA->dy[index];
    z = hitsSoA->z[index];
    dz = hitsSoA->dz[index];
    uncrossed = hitsSoA->uncrossed[index];
    time = hitsSoA->time[index];
    delta_time = hitsSoA->delta_time[index];
    cluster_size = hitsSoA->cluster_size[index];
    region = hitsSoA->region_id[index];
  }

  __device__ void setAtIndex(HitsSoA* hitsSoA, uint index, Hit* hit) {
    hitsSoA->tile[index] = hit->tile;
    hitsSoA->x[index] = hit->x;
    hitsSoA->dx[index] = hit->dx;
    hitsSoA->y[index] = hit->y;
    hitsSoA->dy[index] = hit->dy;
    hitsSoA->z[index] = hit->z;
    hitsSoA->dz[index] = hit->dz;
    hitsSoA->uncrossed[index] = hit->uncrossed;
    hitsSoA->time[index] = hit->time;
    hitsSoA->delta_time[index] = hit->delta_time;
    hitsSoA->cluster_size[index] = hit->cluster_size;
    hitsSoA->region_id[index] = hit->region;
  }

  __device__ void setAtIndex(HitsSoA* hitsSoA, uint index, int tile, float x, float dx, float y, float dy,
      float z, float dz, int uncrossed, unsigned int time, int delta_time, int cluster_size, int region) {
    hitsSoA->tile[index] = tile;
    hitsSoA->x[index] = x;
    hitsSoA->dx[index] = dx;
    hitsSoA->y[index] = y;
    hitsSoA->dy[index] = dy;
    hitsSoA->z[index] = z;
    hitsSoA->dz[index] = dz;
    hitsSoA->uncrossed[index] = uncrossed;
    hitsSoA->time[index] = time;
    hitsSoA->delta_time[index] = delta_time;
    hitsSoA->cluster_size[index] = cluster_size;
    hitsSoA->region_id[index] = region;
  }

  __device__ uint regionAndQuarter(const Digit& i) {
    return i.tile.region() * Constants::n_quarters + i.tile.quarter();
  }

  __device__ void MuonRawToHits::makeStripLayouts(const unsigned int station, const unsigned int region,
      MuonLayout* layouts) const {
    const unsigned int x1 = getLayoutX(muonTables, MuonTables::stripXTableNumber, station, region);
    const unsigned int y1 = getLayoutY(muonTables, MuonTables::stripXTableNumber, station, region);
    const unsigned int x2 = getLayoutX(muonTables, MuonTables::stripYTableNumber, station, region);
    const unsigned int y2 = getLayoutY(muonTables, MuonTables::stripYTableNumber, station, region);

    if (x1 > x2) {
      layouts[0] = MuonLayout(x1, y1);
      layouts[1] = MuonLayout(x2, y2);
    } else {
      layouts[0] = MuonLayout(x2, y2);
      layouts[1] = MuonLayout(x1, y1);
    }
  }

  __device__ void MuonRawToHits::addCoordsCrossingMap(unsigned int* tileIds, unsigned int* tdcValues, bool* used,
      uint startIndex, uint endIndex, HitsSoA* hitsSoA, uint& currentHitIndex) const {
    if (startIndex == endIndex) {
      return;
    }

    MuonLayout layouts[2];
    makeStripLayouts(MuonTileID::station(tileIds[startIndex]), MuonTileID::region(tileIds[startIndex]), layouts);
    const MuonLayout& layoutOne = layouts[0];
    const MuonLayout& layoutTwo = layouts[1];
    uint midIndex = startIndex;
    unsigned int tmpTileId;
    for (uint i = startIndex; i < endIndex; i++) {
      if (MuonTileID::layout(tileIds[i]) == layoutOne) {
        if (midIndex != i) {
          tmpTileId = tileIds[i];
          tileIds[i] = tileIds[midIndex];
          tileIds[midIndex] = tmpTileId;

          tmpTileId = tdcValues[i];
          tdcValues[i] = tdcValues[midIndex];
          tdcValues[midIndex] = tmpTileId;
        }
        midIndex++;
      }
    }

    const int thisGridX = layoutOne.xGrid();
    const int thisGridY = layoutOne.yGrid();
    const int otherGridX = layoutTwo.xGrid();
    const int otherGridY = layoutTwo.yGrid();
    for (uint digitsOneIndex = startIndex; digitsOneIndex < midIndex; digitsOneIndex++) {
      const unsigned int keyX = MuonTileID::nX(tileIds[digitsOneIndex]) * otherGridX / thisGridX;
      const unsigned int keyY = MuonTileID::nY(tileIds[digitsOneIndex]);

      for (uint digitsTwoIndex = midIndex; digitsTwoIndex < endIndex; digitsTwoIndex++) {
        const unsigned int candidateX = MuonTileID::nX(tileIds[digitsTwoIndex]);
        const unsigned int candidateY = MuonTileID::nY(tileIds[digitsTwoIndex]) * thisGridY / otherGridY;

        if (keyX == candidateX && keyY == candidateY) {
          MuonTileID padTile(tileIds[digitsOneIndex]);
          padTile.setY(MuonTileID::nY(tileIds[digitsTwoIndex]));
          padTile.setLayout(MuonLayout(thisGridX, otherGridY));
          float x = 0., dx = 0., y = 0., dy = 0., z = 0., dz = 0.;
          calcTilePos(muonTables, padTile, x, dx, y, dy, z);
          const unsigned int uncrossed = 0;
          const int clusterSize = 0;
          const int region = padTile.region();
          const int localCurrentHitIndex = atomicAdd(&currentHitIndex, 1);
          setAtIndex(hitsSoA, localCurrentHitIndex, padTile.id(), x, dx, y, dy, z, dz, uncrossed,
                     tdcValues[digitsOneIndex], tdcValues[digitsOneIndex] - tdcValues[digitsTwoIndex],
                     clusterSize, region);
          used[digitsOneIndex] = used[digitsTwoIndex] = true;
        }
      }
    }
    // 
    // for (int j=startIndex; j<endIndex; ++j) {
    //   float x = 0.f, dx = 0.f, y = 0.f, dy = 0.f, z = 0.f, dz = 0.f;
    //   const auto tile = Muon::MuonTileID(tileIds[j]);
    //   const auto region = tile.region();
    //   if (tile.station() > (Muon::Constants::n_stations - 3) && region == 0) {
    //     calcTilePos(muonTables, tile, x, dx, y, dy, z);
    //   } else {
    //     const auto station = Muon::MuonTileID::station(tileIds[j]);
    //     const auto region = Muon::MuonTileID::region(tileIds[j]);

    //     const auto x1 = getLayoutX(muonTables, Muon::MuonTables::stripXTableNumber, station, region);
    //     const auto x2 = getLayoutX(muonTables, Muon::MuonTables::stripYTableNumber, station, region);



    //     int dxi;
    //     if (Muon::MuonTileID::layout(tileIds[j]) == layoutOne) {
    //       dxi = calcStripXPos(muonTables, tile, x, dx, y, dy, z);
    //     } else {
    //       dxi = calcStripYPos(muonTables, tile, x, dx, y, dy, z);
    //     }

    //     if (dxi > 10000 || dxi < 0) {
    //       printf("x1>x2, j %i, midIndex %i, dxi %i, x1 %i, x2 %i\n", j, 0, dxi, x1, x2);
    //     }

    //     // if (x1 > x2) {
    //     //   // if (j > midIndex) {
    //     //   //   printf("Wrong >=: %i, %i\n", j, midIndex);
    //     //   // }
    //     //   const auto dxi = calcStripXPos(muonTables, tile, x, dx, y, dy, z);

    //     //   if (dxi > 10000 || dxi < 0) {
    //     //     printf("x1>x2, j %i, midIndex %i, dxi %i, x1 %i, x2 %i\n", j, midIndex, dxi, x1, x2);
    //     //   }
    //     // } else {
    //     //   // if (j <= midIndex) {
    //     //   //   printf("Wrong <: %i, %i\n", j, midIndex);
    //     //   // }
    //     //   const auto dxi = calcStripYPos(muonTables, tile, x, dx, y, dy, z);

    //     //   if (dxi > 10000 || dxi < 0) {
    //     //     printf("x1<=x2, j %i, midIndex %i, dxi %i, x1 %i, x2 %i\n", j, midIndex, dxi, x1, x2);
    //     //   }
    //     // }
    //   }

    //   const unsigned int uncrossed = 1;
    //   const int clusterSize = 0;
    //   const int localCurrentHitIndex = atomicAdd(&currentHitIndex, 1);
    //   setAtIndex(hitsSoA, localCurrentHitIndex, tile.id(), x, dx, y, dy, z, dz, uncrossed,
    //              tdcValues[j], tdcValues[j], clusterSize, region);
    // }

    const uint startIndices[] = {startIndex, midIndex};
    const uint endIndices[] = {midIndex, endIndex};
    for (uint currentDigitsIndex = 0; currentDigitsIndex < 2; currentDigitsIndex++) {
      for (uint currentDigitIndex = startIndices[currentDigitsIndex];
           currentDigitIndex < endIndices[currentDigitsIndex];
           currentDigitIndex++) {
        if (!used[currentDigitIndex]) {
          float x = 0., dx = 0., y = 0., dy = 0., z = 0., dz = 0.;
          MuonTileID tile = MuonTileID(tileIds[currentDigitIndex]);
          const int region = tile.region();
          if (tile.station() > (Constants::n_stations - 3) && region == 0) {
            calcTilePos(muonTables, tile, x, dx, y, dy, z);
          } else {
            if (currentDigitsIndex == 0) {
              calcStripXPos(muonTables, tile, x, dx, y, dy, z);
            } else {
              calcStripYPos(muonTables, tile, x, dx, y, dy, z);
            }
          }
          const unsigned int uncrossed = 1;
          const int clusterSize = 0;
          const int localCurrentHitIndex = atomicAdd(&currentHitIndex, 1);
          setAtIndex(hitsSoA, localCurrentHitIndex, tile.id(), x, dx, y, dy, z, dz, uncrossed,
                     tdcValues[currentDigitIndex], tdcValues[currentDigitIndex], clusterSize, region);
        }
      }
    }
  }
};
