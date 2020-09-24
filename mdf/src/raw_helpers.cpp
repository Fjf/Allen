/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include <cstring>
#include <arpa/inet.h>

#include "compression.hpp"
#include "raw_helpers.hpp"

/// one-at-time hash function
unsigned int LHCb::hash32Checksum(const void* ptr, size_t len)
{
  unsigned int hash = 0;
  const char* k = (const char*) ptr;
  for (size_t i = 0; i < len; ++i, ++k) {
    hash += *k;
    hash += (hash << 10);
    hash ^= (hash >> 6);
  }
  hash += (hash << 3);
  hash ^= (hash >> 11);
  hash += (hash << 15);
  return hash;
}

/* ========================================================================= */
unsigned int LHCb::adler32Checksum(unsigned int adler, const char* buf, size_t len)
{
#define DO1(buf, i)               \
  {                               \
    s1 += (unsigned char) buf[i]; \
    s2 += s1;                     \
  }
#define DO2(buf, i) \
  DO1(buf, i);      \
  DO1(buf, i + 1);
#define DO4(buf, i) \
  DO2(buf, i);      \
  DO2(buf, i + 2);
#define DO8(buf, i) \
  DO4(buf, i);      \
  DO4(buf, i + 4);
#define DO16(buf) \
  DO8(buf, 0);    \
  DO8(buf, 8);

  static const unsigned int BASE = 65521; /* largest prime smaller than 65536 */
  /* NMAX is the largest n such that 255n(n+1)/2 + (n+1)(BASE-1) <= 2^32-1 */
  static const unsigned int NMAX = 5550;
  unsigned int s1 = adler & 0xffff;
  unsigned int s2 = (adler >> 16) & 0xffff;

  if (buf == NULL) return 1;

  while (len > 0) {
    int k = len < NMAX ? (int) len : NMAX;
    len -= k;
    while (k >= 16) {
      DO16(buf);
      buf += 16;
      k -= 16;
    }
    if (k != 0) do {
        s1 += (unsigned char) *buf++;
        s2 += s1;
      } while (--k);
    s1 %= BASE;
    s2 %= BASE;
  }
  unsigned int result = (s2 << 16) | s1;
  return result;
}
/* ========================================================================= */

static unsigned int xorChecksum(const int* ptr, size_t len)
{
  unsigned int checksum = 0;
  len = len / sizeof(int) + ((len % sizeof(int)) ? 1 : 0);
  for (const int *p = ptr, *end = p + len; p < end; ++p) {
    checksum ^= *p;
  }
  return checksum;
}

#define QUOTIENT 0x04c11db7
class CRC32Table {
public:
  unsigned int m_data[256];
  CRC32Table()
  {
    for (int i = 0; i < 256; i++) {
      unsigned int crc = i << 24;
      for (int j = 0; j < 8; j++) {
        if (crc & 0x80000000)
          crc = (crc << 1) ^ QUOTIENT;
        else
          crc = crc << 1;
      }
      m_data[i] = htonl(crc);
    }
  }
  const unsigned int* data() const { return m_data; }
};

// Only works for word aligned data and assumes that the data is an exact number of words
// Copyright  1993 Richard Black. All rights are reserved.
static unsigned int crc32Checksum(const char* data, size_t len)
{
  static CRC32Table table;
  const unsigned int* crctab = table.data();
  const unsigned int* p = (const unsigned int*) data;
  const unsigned int* e = (const unsigned int*) (data + len);
  if (len < 4 || (size_t(data) % sizeof(unsigned int)) != 0) return ~0x0;
  unsigned int result = ~*p++;
  while (p < e) {
#if defined(LITTLE_ENDIAN)
    result = crctab[result & 0xff] ^ result >> 8;
    result = crctab[result & 0xff] ^ result >> 8;
    result = crctab[result & 0xff] ^ result >> 8;
    result = crctab[result & 0xff] ^ result >> 8;
    result ^= *p++;
#else
    result = crctab[result >> 24] ^ result << 8;
    result = crctab[result >> 24] ^ result << 8;
    result = crctab[result >> 24] ^ result << 8;
    result = crctab[result >> 24] ^ result << 8;
    result ^= *p++;
#endif
  }

  return ~result;
}

static unsigned short crc16Checksum(const char* data, size_t len)
{
  static const unsigned short wCRCTable[] = {
    0X0000, 0XC0C1, 0XC181, 0X0140, 0XC301, 0X03C0, 0X0280, 0XC241, 0XC601, 0X06C0, 0X0780, 0XC741, 0X0500, 0XC5C1,
    0XC481, 0X0440, 0XCC01, 0X0CC0, 0X0D80, 0XCD41, 0X0F00, 0XCFC1, 0XCE81, 0X0E40, 0X0A00, 0XCAC1, 0XCB81, 0X0B40,
    0XC901, 0X09C0, 0X0880, 0XC841, 0XD801, 0X18C0, 0X1980, 0XD941, 0X1B00, 0XDBC1, 0XDA81, 0X1A40, 0X1E00, 0XDEC1,
    0XDF81, 0X1F40, 0XDD01, 0X1DC0, 0X1C80, 0XDC41, 0X1400, 0XD4C1, 0XD581, 0X1540, 0XD701, 0X17C0, 0X1680, 0XD641,
    0XD201, 0X12C0, 0X1380, 0XD341, 0X1100, 0XD1C1, 0XD081, 0X1040, 0XF001, 0X30C0, 0X3180, 0XF141, 0X3300, 0XF3C1,
    0XF281, 0X3240, 0X3600, 0XF6C1, 0XF781, 0X3740, 0XF501, 0X35C0, 0X3480, 0XF441, 0X3C00, 0XFCC1, 0XFD81, 0X3D40,
    0XFF01, 0X3FC0, 0X3E80, 0XFE41, 0XFA01, 0X3AC0, 0X3B80, 0XFB41, 0X3900, 0XF9C1, 0XF881, 0X3840, 0X2800, 0XE8C1,
    0XE981, 0X2940, 0XEB01, 0X2BC0, 0X2A80, 0XEA41, 0XEE01, 0X2EC0, 0X2F80, 0XEF41, 0X2D00, 0XEDC1, 0XEC81, 0X2C40,
    0XE401, 0X24C0, 0X2580, 0XE541, 0X2700, 0XE7C1, 0XE681, 0X2640, 0X2200, 0XE2C1, 0XE381, 0X2340, 0XE101, 0X21C0,
    0X2080, 0XE041, 0XA001, 0X60C0, 0X6180, 0XA141, 0X6300, 0XA3C1, 0XA281, 0X6240, 0X6600, 0XA6C1, 0XA781, 0X6740,
    0XA501, 0X65C0, 0X6480, 0XA441, 0X6C00, 0XACC1, 0XAD81, 0X6D40, 0XAF01, 0X6FC0, 0X6E80, 0XAE41, 0XAA01, 0X6AC0,
    0X6B80, 0XAB41, 0X6900, 0XA9C1, 0XA881, 0X6840, 0X7800, 0XB8C1, 0XB981, 0X7940, 0XBB01, 0X7BC0, 0X7A80, 0XBA41,
    0XBE01, 0X7EC0, 0X7F80, 0XBF41, 0X7D00, 0XBDC1, 0XBC81, 0X7C40, 0XB401, 0X74C0, 0X7580, 0XB541, 0X7700, 0XB7C1,
    0XB681, 0X7640, 0X7200, 0XB2C1, 0XB381, 0X7340, 0XB101, 0X71C0, 0X7080, 0XB041, 0X5000, 0X90C1, 0X9181, 0X5140,
    0X9301, 0X53C0, 0X5280, 0X9241, 0X9601, 0X56C0, 0X5780, 0X9741, 0X5500, 0X95C1, 0X9481, 0X5440, 0X9C01, 0X5CC0,
    0X5D80, 0X9D41, 0X5F00, 0X9FC1, 0X9E81, 0X5E40, 0X5A00, 0X9AC1, 0X9B81, 0X5B40, 0X9901, 0X59C0, 0X5880, 0X9841,
    0X8801, 0X48C0, 0X4980, 0X8941, 0X4B00, 0X8BC1, 0X8A81, 0X4A40, 0X4E00, 0X8EC1, 0X8F81, 0X4F40, 0X8D01, 0X4DC0,
    0X4C80, 0X8C41, 0X4400, 0X84C1, 0X8581, 0X4540, 0X8701, 0X47C0, 0X4680, 0X8641, 0X8201, 0X42C0, 0X4380, 0X8341,
    0X4100, 0X81C1, 0X8081, 0X4040};

  unsigned short wCRCWord = 0xFFFF;
  while (len--) {
    unsigned int nTemp = *data++ ^ wCRCWord;
    wCRCWord >>= 8;
    wCRCWord = (unsigned short) (wCRCWord ^ wCRCTable[nTemp]);
  }
  return wCRCWord;
}

static char crc8Checksum(const char* data, int len)
{
  static unsigned char crc8_table[] = {
    0,   94,  188, 226, 97,  63,  221, 131, 194, 156, 126, 32,  163, 253, 31,  65,  157, 195, 33,  127, 252, 162,
    64,  30,  95,  1,   227, 189, 62,  96,  130, 220, 35,  125, 159, 193, 66,  28,  254, 160, 225, 191, 93,  3,
    128, 222, 60,  98,  190, 224, 2,   92,  223, 129, 99,  61,  124, 34,  192, 158, 29,  67,  161, 255, 70,  24,
    250, 164, 39,  121, 155, 197, 132, 218, 56,  102, 229, 187, 89,  7,   219, 133, 103, 57,  186, 228, 6,   88,
    25,  71,  165, 251, 120, 38,  196, 154, 101, 59,  217, 135, 4,   90,  184, 230, 167, 249, 27,  69,  198, 152,
    122, 36,  248, 166, 68,  26,  153, 199, 37,  123, 58,  100, 134, 216, 91,  5,   231, 185, 140, 210, 48,  110,
    237, 179, 81,  15,  78,  16,  242, 172, 47,  113, 147, 205, 17,  79,  173, 243, 112, 46,  204, 146, 211, 141,
    111, 49,  178, 236, 14,  80,  175, 241, 19,  77,  206, 144, 114, 44,  109, 51,  209, 143, 12,  82,  176, 238,
    50,  108, 142, 208, 83,  13,  239, 177, 240, 174, 76,  18,  145, 207, 45,  115, 202, 148, 118, 40,  171, 245,
    23,  73,  8,   86,  180, 234, 105, 55,  213, 139, 87,  9,   235, 181, 54,  104, 138, 212, 149, 203, 41,  119,
    244, 170, 72,  22,  233, 183, 85,  11,  136, 214, 52,  106, 43,  117, 151, 201, 74,  20,  246, 168, 116, 42,
    200, 150, 21,  75,  169, 247, 182, 232, 10,  84,  215, 137, 107, 53};
  const char* s = data;
  char c = 0;
  while (len--)
    c = crc8_table[c ^ *s++];
  return c;
}

/// Generate XOR Checksum
unsigned int LHCb::genChecksum(int flag, const void* ptr, size_t len)
{
  switch (flag) {
  case 0: return xorChecksum((const int*) ptr, len);
  case 1: return hash32Checksum(ptr, len);
  case 2: len = (len / sizeof(int)) * sizeof(int); return crc32Checksum((const char*) ptr, len);
  case 3: len = (len / sizeof(short)) * sizeof(short); return crc16Checksum((const char*) ptr, len);
  case 4: return crc8Checksum((const char*) ptr, len);
  case 5: len = (len / sizeof(int)) * sizeof(int); return adler32Checksum(1, (const char*) ptr, len);
  case 22: // Old CRC32 (fixed by now)
    return crc32Checksum((const char*) ptr, len);
  default: return ~0x0;
  }
}

/// Decompress opaque data buffer
bool LHCb::decompressBuffer(
  int algtype,
  unsigned char* tar,
  size_t tar_len,
  unsigned char* src,
  size_t src_len,
  size_t& new_len)
{
  int in_len, out_len, res_len = 0;
  switch (algtype) {
  case 0:
    if (tar != src && tar_len >= src_len) {
      new_len = src_len;
      ::memcpy(tar, src, src_len);
      return true;
    }
    break;
  case 1:
  case 2:
  case 3:
  case 4:
  case 5:
  case 6:
  case 7:
  case 8:
  case 9:
    in_len = src_len;
    out_len = tar_len;
    Compression::unzip(&in_len, src, &out_len, tar, &res_len);
    if (res_len > 0) {
      new_len = res_len;
      return true;
    }
    break;
  default: break;
  }
  return false;
}
