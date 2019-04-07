#include "MuonTable.h"

void MuonTableReader::read(const char* raw_input, MuonTable* pad, MuonTable* stripX, MuonTable* stripY) {
  MuonTable* muonTables[3] = {pad, stripX, stripY};
  for (MuonTable* muonTable : muonTables) {
    size_t gridXSize;
    std::copy_n((size_t*) raw_input, 1, &gridXSize);
    raw_input += sizeof(size_t);
    std::copy_n((int*) raw_input, gridXSize, muonTable -> gridX);
    raw_input += sizeof(int) * gridXSize;

    size_t gridYSize;
    std::copy_n((size_t*) raw_input, 1, &gridYSize);
    raw_input += sizeof(size_t);
    std::copy_n((int*) raw_input, gridYSize, muonTable -> gridY);
    raw_input += sizeof(int) * gridYSize;

    size_t sizeXSize;
    std::copy_n((size_t *) raw_input, 1, &sizeXSize);
    raw_input += sizeof(size_t);
    std::copy_n((float*) raw_input, sizeXSize, muonTable -> sizeX);
    raw_input += sizeof(float) * sizeXSize;

    size_t sizeYSize;
    std::copy_n((size_t*) raw_input, 1, &sizeYSize);
    raw_input += sizeof(size_t);
    std::copy_n((float*) raw_input, sizeYSize, muonTable -> sizeY);
    raw_input += sizeof(float) * sizeYSize;

    size_t offsetSize;
    std::copy_n((size_t*) raw_input, 1, &offsetSize);
    raw_input += sizeof(size_t);
    std::copy_n((unsigned int*) raw_input, offsetSize, muonTable -> offset);
    raw_input += sizeof(unsigned int) * offsetSize;

    size_t tableSize;
    std::copy_n((size_t*) raw_input, 1, &tableSize);
    raw_input += sizeof(size_t);
    for (int i = 0; i < tableSize; i++) {
      size_t stationTableSize;
      std::copy_n((size_t*) raw_input, 1, &stationTableSize);
      raw_input += sizeof(size_t);
      (muonTable -> points)[i].resize(stationTableSize);
      for (int j = 0; j < stationTableSize; j++) {
        float point[3];
        std::copy_n((float *) raw_input, 3, point);
        raw_input += sizeof(float) * 3;
        for (int k = 0; k < 3; k++) {
          (muonTable->points)[i][j][k] = point[k];
        }
      }
    }
  }
}


void calcTilePos(MuonTable* pad, LHCb::MuonTileID& tile, double& x, double& deltax, double& y, double& deltay,
                 double& z) {
  int          station    = tile.station();
  int          region     = tile.region();
  int          quarter    = tile.quarter();
  int          perQuarter = 3 * padGridX[station] * padGridY[station];
  unsigned int xpad       = tile.nX();
  unsigned int ypad       = tile.nY();
  auto index              = static_cast<unsigned int>((region * 4 + quarter ) * perQuarter);

  if ( ypad < padGridY[station] ) {
    index = index + padGridX[station] * ypad + xpad - padGridX[station];
  } else {
    index = index + padGridX[station] * padGridY[station] +
            2 * padGridX[station] * ( ypad - padGridY[station] ) + xpad;
  }

  auto& p = (pad -> points)[station][index];
  x                        = p[0];
  y                        = p[1];
  z                        = p[2];
  deltax                   = (pad -> sizeX)[station * 4 + region];
  deltay                   = (pad -> sizeY)[station * 4 + region];
}

void calcStripXPos(MuonTable* stripX, LHCb::MuonTileID& tile, double& x, double& deltax, double& y, double& deltay,
                   double& z) {
  int station        = tile.station();
  int region         = tile.region();
  int quarter        = tile.quarter();
  int perQuarter     = 3 * stripXGridX[station * 4 + region] * stripXGridY[station * 4 + region];
  unsigned int xpad  = tile.nX();
  unsigned int ypad  = tile.nY();
  unsigned int index = (stripX -> offset)[station * 4 + region] + quarter * perQuarter;

  if ( ypad < stripXGridY[station * 4 + region] ) {
    index = index + stripXGridX[station * 4 + region] * ypad + xpad -
            stripXGridX[station * 4 + region];
  } else {
    index = index + stripXGridX[station * 4 + region] * stripXGridY[station * 4 + region] +
            2 * stripXGridX[station * 4 + region] * ( ypad - stripXGridY[station * 4 + region] ) +
            xpad;
  }

  auto& p = (stripX -> points)[station][index];
  x                        = p[0];
  y                        = p[1];
  z                        = p[2];
  deltax                   = (stripX -> sizeX)[station * 4 + region];
  deltay                   = (stripX -> sizeY)[station * 4 + region];
}

void calcStripYPos(MuonTable* stripY, LHCb::MuonTileID& tile, double& x, double& deltax, double& y, double& deltay,
                   double& z) {
  int station        = tile.station();
  int region         = tile.region();
  int quarter        = tile.quarter();
  int perQuarter     = 3 * stripYGridX[station * 4 + region] * stripYGridY[station * 4 + region];
  unsigned int xpad  = tile.nX();
  unsigned int ypad  = tile.nY();
  unsigned int index = (stripY -> offset)[station * 4 + region] + quarter * perQuarter;

  if ( ypad < stripYGridY[station * 4 + region] ) {
    index = index + stripYGridX[station * 4 + region] * ypad + xpad -
            stripYGridX[station * 4 + region];
  } else {
    index = index + stripYGridX[station * 4 + region] * stripYGridY[station * 4 + region] +
            2 * stripYGridX[station * 4 + region] * ( ypad - stripYGridY[station * 4 + region] ) +
            xpad;
  }

  auto& p = (stripY -> points)[station][index];
  x                        = p[0];
  y                        = p[1];
  z                        = p[2];
  deltax                   = (stripY -> sizeX)[station * 4 + region];
  deltay                   = (stripY -> sizeY)[station * 4 + region];
}

int transform_for_uncrossed_hits(LHCb::MuonTileID& tile, MuonTable* pad, MuonTable* stripX, MuonTable* stripY,
                                 double& x, double& dx, double& y, double& dy, double& z) {
  unsigned int x1 = getLayoutX(0, tile.station(), tile.region());
  unsigned int y1 = getLayoutY(0, tile.station(), tile.region());
  MuonLayout layoutOne = MuonLayout(x1, y1);
  if (tile.station() > 1 && tile.region() == 0) {
    calcTilePos(pad, tile, x, dx, y, dy, z);
    return 1;
  } else {
    if (tile.layout() == layoutOne) {
      calcStripXPos(stripX, tile, x, dx, y, dy, z);
      return 2;
    } else {
      calcStripYPos(stripY, tile, x, dx, y, dy, z);
      return 3;
    }
  }
}
