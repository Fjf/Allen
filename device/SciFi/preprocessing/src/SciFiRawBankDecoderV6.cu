/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#include "SciFiRawBankDecoderV6.cuh"
#include <MEPTools.h>
#include "assert.h"

INSTANTIATE_ALGORITHM(scifi_raw_bank_decoder_v6::scifi_raw_bank_decoder_v6_t)

// Merge of PrStoreFTHit and RawBankDecoder.
__device__ void make_cluster_v6(
  const int hit_index,
  const SciFi::SciFiGeometry& geom,
  uint32_t chan,
  uint8_t fraction,
  uint8_t pseudoSize,
  SciFi::Hits& hits)
{
  const SciFi::SciFiChannelID id {chan};

  // Offset to save space in geometry structure, see DumpFTGeometry.cpp
  const uint32_t mat = id.uniqueMat() - 512;
  const uint32_t planeCode = id.uniqueLayer() - 4;
  const float dxdy = geom.dxdy[mat];
  const float dzdy = geom.dzdy[mat];
  float uFromChannel = geom.uBegin[mat] + (2 * id.channel() + 1 + fraction) * geom.halfChannelPitch[mat];
  if (id.die()) uFromChannel += geom.dieGap[mat];
  uFromChannel += id.sipm() * geom.sipmPitch[mat];
  const float endPointX = geom.mirrorPointX[mat] + geom.ddxX[mat] * uFromChannel;
  const float endPointY = geom.mirrorPointY[mat] + geom.ddxY[mat] * uFromChannel;
  const float endPointZ = geom.mirrorPointZ[mat] + geom.ddxZ[mat] * uFromChannel;
  const float x0 = endPointX - dxdy * endPointY;
  const float z0 = endPointZ - dzdy * endPointY;

  assert(pseudoSize < 9 && "Pseudosize of cluster is > 8. Out of range.");

  // Apparently the unique* methods are not designed to start at 0, therefore -16
  const uint32_t uniqueZone = ((id.uniqueQuarter() - 16) >> 1);

  const unsigned plane_code = 2 * planeCode + (uniqueZone % 2);
  hits.x0(hit_index) = x0;
  hits.z0(hit_index) = z0;
  hits.channel(hit_index) = chan;
  hits.endPointY(hit_index) = endPointY;
  assert(fraction <= 0x1 && plane_code <= 0x1f && pseudoSize <= 0xf && mat <= 0x7ff);
  hits.assembled_datatype(hit_index) = fraction << 20 | plane_code << 15 | pseudoSize << 11 | mat;
}

template<bool mep_layout>
__global__ void scifi_raw_bank_decoder_v6_kernel(
  scifi_raw_bank_decoder_v6::Parameters parameters,
  const char* scifi_geometry)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  const unsigned number_of_events = parameters.dev_number_of_events[0];

  const SciFi::SciFiGeometry geom {scifi_geometry};
  const auto scifi_raw_event =
    SciFi::RawEvent<mep_layout>(parameters.dev_scifi_raw_input, parameters.dev_scifi_raw_input_offsets, event_number);

  SciFi::Hits hits {parameters.dev_scifi_hits,
                    parameters.dev_scifi_hit_offsets[number_of_events * SciFi::Constants::n_mat_groups_and_mats]};
  SciFi::ConstHitCount hit_count {parameters.dev_scifi_hit_offsets, event_number};
  const unsigned number_of_hits_in_event = hit_count.event_number_of_hits();

  for (unsigned i = threadIdx.x; i < number_of_hits_in_event; i += blockDim.x) {
    const uint32_t cluster_reference = parameters.dev_cluster_references[hit_count.event_offset() + i];
    const int raw_bank_number = (cluster_reference >> 24) & 0xFF;
    const int it_number = (cluster_reference >> 16) & 0xFF;
    const int condition = (cluster_reference >> 13) & 0x07;
    const int delta_parameter = cluster_reference & 0xFF;

    const auto rawbank = scifi_raw_event.raw_bank(raw_bank_number);
    const uint16_t* it = rawbank.data + 2;
    it += it_number;

    const uint16_t c = *it;
    const uint32_t ch = geom.bank_first_channel[rawbank.sourceID] + SciFi::channelInBank(c);

    // Call parameters for make_cluster
    uint32_t cluster_chan = ch;
    uint8_t cluster_fraction = SciFi::fraction(c);
    uint8_t pseudoSize = 4;

    assert(condition != 0x00 && "Invalid cluster condition. Usually empty slot due to counting/decoding mismatch.");

    if (condition == 0x02) {
      pseudoSize = 0;
    }
    else if (condition > 0x02) {
      const auto c2 = *(it + 1);
      const auto widthClus = (SciFi::cell(c2) - SciFi::cell(c) + 2);

      if (condition == 0x03) {
        pseudoSize = 0;
        cluster_fraction = 1;
        cluster_chan += delta_parameter;
      }
      else if (condition == 0x04) {
        pseudoSize = 0;
        cluster_fraction = (widthClus - 1) % 2;
        cluster_chan += delta_parameter + (widthClus - delta_parameter - 1) / 2 - 1;
      }
      else if (condition == 0x05) {
        pseudoSize = widthClus;
        cluster_fraction = (widthClus - 1) % 2;
        cluster_chan += (widthClus - 1) / 2 - 1;
      }
    }

    make_cluster_v6(hit_count.event_offset() + i, geom, cluster_chan, cluster_fraction, pseudoSize, hits);
  }
}

void scifi_raw_bank_decoder_v6::scifi_raw_bank_decoder_v6_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&,
  const HostBuffers&) const
{
  set_size<dev_scifi_hits_t>(
    arguments,
    first<host_accumulated_number_of_scifi_hits_t>(arguments) * SciFi::Hits::number_of_arrays * sizeof(uint32_t));
}

void scifi_raw_bank_decoder_v6::scifi_raw_bank_decoder_v6_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  HostBuffers&,
  const Allen::Context& context) const
{
  global_function(
    runtime_options.mep_layout ? scifi_raw_bank_decoder_v6_kernel<true> : scifi_raw_bank_decoder_v6_kernel<false>)(
    dim3(size<dev_event_list_t>(arguments)), property<block_dim_t>(), context)(arguments, constants.dev_scifi_geometry);
}
