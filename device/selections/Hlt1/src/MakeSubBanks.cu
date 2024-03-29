/*****************************************************************************\
* (c) Copyright 2022 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "MakeSubBanks.cuh"

INSTANTIATE_ALGORITHM(make_subbanks::make_subbanks_t)

void make_subbanks::make_subbanks_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_rb_substr_t>(arguments, first<host_substr_bank_size_t>(arguments));
  set_size<dev_rb_hits_t>(arguments, first<host_hits_bank_size_t>(arguments));
  set_size<dev_rb_stdinfo_t>(arguments, first<host_stdinfo_bank_size_t>(arguments));
  set_size<dev_rb_objtyp_t>(arguments, first<host_objtyp_bank_size_t>(arguments));
}

void make_subbanks::make_subbanks_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_rb_substr_t>(arguments, 0, context);
  Allen::memset_async<dev_rb_hits_t>(arguments, 0, context);
  Allen::memset_async<dev_rb_stdinfo_t>(arguments, 0, context);
  Allen::memset_async<dev_rb_objtyp_t>(arguments, 0, context);
  global_function(make_rb_substr)(dim3(first<host_number_of_events_t>(arguments)), property<block_dim_t>(), context)(
    arguments, first<host_number_of_events_t>(arguments));

  global_function(make_rb_hits)(dim3(first<host_number_of_events_t>(arguments)), property<block_dim_t>(), context)(
    arguments);
}

__global__ void make_subbanks::make_rb_substr(make_subbanks::Parameters parameters, const unsigned number_of_events)
{

  for (unsigned event_number = blockIdx.x * blockDim.x + threadIdx.x; event_number < number_of_events;
       event_number += blockDim.x * gridDim.x) {

    const unsigned n_sels = parameters.dev_sel_count[event_number];
    if (n_sels == 0) continue;

    unsigned* event_rb_substr = parameters.dev_rb_substr + parameters.dev_rb_substr_offsets[event_number];
    const unsigned event_rb_substr_size =
      parameters.dev_rb_substr_offsets[event_number + 1] - parameters.dev_rb_substr_offsets[event_number];
    const unsigned n_lines = parameters.dev_number_of_active_lines[0];
    const unsigned* line_object_offsets = parameters.dev_max_objects_offsets + n_lines * event_number;
    const unsigned n_children = parameters.max_children_per_object;
    const unsigned selected_object_offset = n_children * line_object_offsets[0];
    const unsigned n_tracks = parameters.dev_unique_track_count[event_number];
    const unsigned n_svs = parameters.dev_unique_sv_count[event_number];

    const unsigned sels_start_short = 2;
    const unsigned svs_start_short = sels_start_short + parameters.dev_substr_sel_size[event_number];
    const unsigned tracks_start_short = svs_start_short + parameters.dev_substr_sv_size[event_number];

    const auto event_track_ptrs = parameters.dev_basic_particle_ptrs + selected_object_offset;
    const auto event_sv_ptrs = parameters.dev_composite_particle_ptrs + selected_object_offset;
    const unsigned* event_unique_track_list = parameters.dev_unique_track_list + selected_object_offset;
    const unsigned* event_unique_sv_list = parameters.dev_unique_sv_list + selected_object_offset;

    // Add the track substructures.
    // Each track substructure has one pointer to a sequence of LHCbIDs.
    unsigned track_struct = ((1 & 0xFFFF) << 1) | 1;
    for (unsigned i_track = 0; i_track < n_tracks; i_track++) {
      const unsigned i_short = tracks_start_short + 2 * i_track;
      const unsigned i_word = i_short / 2;
      const unsigned i_part = i_short % 2;
      const unsigned mask = 0xFFFFL;
      const unsigned bits = 16;

      if (i_part == 0) {
        event_rb_substr[i_word] = (event_rb_substr[i_word] & ~mask) | track_struct;
        event_rb_substr[i_word] = (event_rb_substr[i_word] & ~(mask << bits)) | (i_track << bits);
      }
      else {
        event_rb_substr[i_word] = (event_rb_substr[i_word] & ~(mask << bits)) | (track_struct << bits);
        event_rb_substr[i_word + 1] = (event_rb_substr[i_word + 1] & ~mask) | i_track;
      }
    }

    // Add the SV substructures.
    // Each SV substructure has a pointer to each of its constituent particles.
    unsigned i_short = svs_start_short;
    for (unsigned i_sv = 0; i_sv < n_svs; i_sv++) {
      unsigned i_word = i_short / 2;
      unsigned i_part = i_short % 2;
      const unsigned mask = 0xFFFFL;
      const unsigned bits = 16;
      const unsigned sv_index = event_unique_sv_list[i_sv];
      const Allen::Views::Physics::CompositeParticle* sv =
        static_cast<const Allen::Views::Physics::CompositeParticle*>(event_sv_ptrs[sv_index]);

      const unsigned n_substr = sv->number_of_children();
      const unsigned sv_struct = ((n_substr & 0xFFFF) << 1) | 0;
      if (i_part == 0) {
        event_rb_substr[i_word] = (event_rb_substr[i_word] & ~mask) | sv_struct;
      }
      else {
        event_rb_substr[i_word] = (event_rb_substr[i_word] & ~(mask << bits)) | (sv_struct << bits);
      }
      for (unsigned i_substr = 0; i_substr < n_substr; i_substr++) {
        // Find the location of the substructure in the bank.
        const auto substr = sv->child(i_substr);
        const auto basic_substr = Allen::dyn_cast<const Allen::Views::Physics::BasicParticle*>(substr);
        unsigned substr_loc;
        if (basic_substr) {
          for (unsigned i_track = 0; i_track < n_tracks; i_track++) {
            const unsigned track_index = event_unique_track_list[i_track];
            if (basic_substr == event_track_ptrs[track_index]) {
              substr_loc = n_sels + n_svs + i_track;
              break;
            }
          }
        }
        else {
          const auto composite_substr = static_cast<const Allen::Views::Physics::CompositeParticle*>(substr);
          for (unsigned i_sv = 0; i_sv < n_svs; i_sv++) {
            const unsigned sv_index = event_unique_sv_list[i_sv];
            if (composite_substr == event_sv_ptrs[sv_index]) {
              substr_loc = n_sels + i_sv;
              break;
            }
          }
        }

        i_short++;
        i_word = i_short / 2;
        i_part = i_short % 2;
        if (i_part == 0) {
          event_rb_substr[i_word] = (event_rb_substr[i_word] & ~mask) | substr_loc;
        }
        else {
          event_rb_substr[i_word] = (event_rb_substr[i_word] & ~(mask << bits)) | (substr_loc << bits);
        }
      }
      i_short++;
    }

    // Set the banks size.
    event_rb_substr[0] = (event_rb_substr[0] & ~0xFFFFL) | (unsigned) (n_sels + n_svs + n_tracks);
    event_rb_substr[0] = (event_rb_substr[0] & ~(0xFFFL << 16)) | (unsigned) (event_rb_substr_size << 16);

    const unsigned* event_candidate_offsets =
      parameters.dev_candidate_offsets + event_number * parameters.dev_number_of_active_lines[0];
    const unsigned* event_sel_list = parameters.dev_sel_list + event_number * parameters.dev_number_of_active_lines[0];

    unsigned insert_short = sels_start_short;
    for (unsigned i_line = 0; i_line < n_sels; i_line += 1) {
      unsigned line_id = event_sel_list[i_line];
      const auto mec = parameters.dev_multi_event_particle_containers[line_id];

      // Handle lines that select BasicParticles.
      if (Allen::dyn_cast<const Allen::Views::Physics::MultiEventBasicParticles*>(mec)) {
        const unsigned* line_candidate_indices =
          parameters.dev_sel_track_indices + n_children * line_object_offsets[line_id];
        unsigned n_cand = event_candidate_offsets[line_id + 1] - event_candidate_offsets[line_id];
        unsigned i_word = insert_short / 2;
        unsigned i_part = insert_short % 2;
        unsigned bits = 16 * i_part;
        unsigned mask = 0xFFFFL << bits;
        unsigned sel_struct = ((n_cand & 0xFFFF) << 1) | 0;
        event_rb_substr[i_word] = (event_rb_substr[i_word] & ~mask) | (sel_struct << bits);
        insert_short++;
        for (unsigned i_cand = 0; i_cand < n_cand; i_cand++) {
          const unsigned i_track = line_candidate_indices[i_cand];
          const unsigned track_index = parameters.dev_track_duplicate_map[selected_object_offset + i_track] >= 0 ?
                                         parameters.dev_track_duplicate_map[selected_object_offset + i_track] :
                                         i_track;
          unsigned obj_index = 0;
          // if (track_index < 0) track_index = i_track;
          for (unsigned j_track = 0; j_track < n_tracks; j_track++) {
            const unsigned test_index = parameters.dev_unique_track_list[selected_object_offset + j_track];
            if (track_index == test_index) {
              obj_index = n_sels + n_svs + j_track;
              break;
            }
          }
          unsigned i_word = insert_short / 2;
          unsigned i_part = insert_short % 2;
          unsigned bits = 16 * i_part;
          unsigned mask = 0xFFFFL << bits;
          event_rb_substr[i_word] = (event_rb_substr[i_word] & ~mask) | (obj_index << bits);
          insert_short++;
        }
      }
      // Handle lines that select CompositeParticles.
      else if (Allen::dyn_cast<const Allen::Views::Physics::MultiEventCompositeParticles*>(mec)) {
        const unsigned* line_candidate_indices =
          parameters.dev_sel_sv_indices + n_children * line_object_offsets[line_id];
        unsigned n_cand = event_candidate_offsets[line_id + 1] - event_candidate_offsets[line_id];
        unsigned i_word = insert_short / 2;
        unsigned i_part = insert_short % 2;
        unsigned bits = 16 * i_part;
        unsigned mask = 0xFFFFL << bits;
        unsigned sel_struct = ((n_cand & 0xFFFF) << 1) | 0;
        event_rb_substr[i_word] = (event_rb_substr[i_word] & ~mask) | (sel_struct << bits);
        insert_short++;
        for (unsigned i_cand = 0; i_cand < n_cand; i_cand++) {
          const unsigned i_sv = line_candidate_indices[i_cand];
          const unsigned sv_index = parameters.dev_sv_duplicate_map[selected_object_offset + i_sv] >= 0 ?
                                      parameters.dev_sv_duplicate_map[selected_object_offset + i_sv] :
                                      i_sv;
          unsigned obj_index = 0;
          // if (sv_index < 0) sv_index = i_sv;
          for (unsigned j_sv = 0; j_sv < n_svs; j_sv++) {
            const unsigned test_index = event_unique_sv_list[j_sv];
            if (sv_index == test_index) {
              obj_index = n_sels + j_sv;
              break;
            }
          }
          unsigned i_word = insert_short / 2;
          unsigned i_part = insert_short % 2;
          unsigned bits = 16 * i_part;
          unsigned mask = 0xFFFFL << bits;
          event_rb_substr[i_word] = (event_rb_substr[i_word] & ~mask) | (obj_index << bits);
          insert_short++;
        }
      }
      // If the line does not select particles, it contains 0 pointers to
      // object-type substructures.
      else {
        unsigned i_word = insert_short / 2;
        unsigned i_part = insert_short % 2;
        unsigned bits = 16 * i_part;
        unsigned mask = 0xFFFFL << bits;
        event_rb_substr[i_word] = (event_rb_substr[i_word] & ~mask) | (0 << bits);
        insert_short++;
      }
    }

    // Create the ObjTyp subbank.
    const unsigned objtyp_offset = parameters.dev_rb_objtyp_offsets[event_number];
    const unsigned objtyp_size = parameters.dev_rb_objtyp_offsets[event_number + 1] - objtyp_offset;
    const unsigned n_objtyps = objtyp_size - 1;
    unsigned* event_rb_objtyp = parameters.dev_rb_objtyp + objtyp_offset;
    const unsigned mask = 0xFFFFL;
    const unsigned bits = 16;
    unsigned i_obj = 1;
    // Fill the bank size.
    event_rb_objtyp[0] = (event_rb_objtyp[0] & ~mask) | n_objtyps;
    event_rb_objtyp[0] = (event_rb_objtyp[0] & ~(mask << bits)) | (objtyp_size << bits);
    // Selections.
    if (n_sels != 0) {
      unsigned short CLID = 1;
      event_rb_objtyp[i_obj] = (event_rb_objtyp[i_obj] & ~mask) | n_sels;
      event_rb_objtyp[i_obj] = (event_rb_objtyp[i_obj] & ~(mask << bits)) | (CLID << bits);
      i_obj++;
    }
    // SVs.
    if (n_svs != 0) {
      unsigned short CLID = 10030;
      event_rb_objtyp[i_obj] = (event_rb_objtyp[i_obj] & ~mask) | (n_sels + n_svs);
      event_rb_objtyp[i_obj] = (event_rb_objtyp[i_obj] & ~(mask << bits)) | (CLID << bits);
      i_obj++;
    }
    // Tracks.
    if (n_tracks != 0) {
      unsigned short CLID = 10010;
      event_rb_objtyp[i_obj] = (event_rb_objtyp[i_obj] & ~mask) | (n_sels + n_svs + n_tracks);
      event_rb_objtyp[i_obj] = (event_rb_objtyp[i_obj] & ~(mask << bits)) | (CLID << bits);
    }

    // Create the StdInfo bank.
    unsigned* event_rb_stdinfo = parameters.dev_rb_stdinfo + parameters.dev_rb_stdinfo_offsets[event_number];
    const unsigned stdinfo_size =
      parameters.dev_rb_stdinfo_offsets[event_number + 1] - parameters.dev_rb_stdinfo_offsets[event_number];
    const unsigned sels_start_word = 1 + (3 + n_tracks + n_svs + n_sels) / 4;

    // Skip events with an empty StdInfo bank.
    if (stdinfo_size == 0) continue;

    // Number of objects stored in the less significant short.
    event_rb_stdinfo[0] = (event_rb_stdinfo[0] & ~0xFFFFu) | ((unsigned) (n_tracks + n_svs + n_sels));
    // Bank size in words in the more significant short.
    event_rb_stdinfo[0] = (event_rb_stdinfo[0] & ~(0xFFFFu << 16)) | ((unsigned) (stdinfo_size << 16));

    for (unsigned i_sel = 0; i_sel < n_sels; i_sel++) {
      unsigned i_word = 1 + i_sel / 4;
      unsigned i_part = i_sel % 4;
      unsigned bits = 8 * i_part;
      unsigned mask = 0xFFL << bits;
      unsigned n_info = 1;
      event_rb_stdinfo[i_word] = (event_rb_stdinfo[i_word] & ~mask) | (n_info << bits);

      // Selection IDs must be stored as floats
      i_word = sels_start_word + i_sel;
      float* float_info = reinterpret_cast<float*>(event_rb_stdinfo);
      float_info[i_word] = static_cast<float>(event_sel_list[i_sel] + 1);
    }

    // Add SV information to the beginning of the bank.
    const auto svs_start_word = sels_start_word + n_sels;
    for (unsigned i_sv = 0; i_sv < n_svs; i_sv++) {
      unsigned i_obj = n_sels + i_sv;
      unsigned i_word = 1 + i_obj / 4;
      unsigned i_part = i_obj % 4;
      unsigned bits = 8 * i_part;
      unsigned mask = 0xFFL << bits;
      unsigned n_info = 4;
      event_rb_stdinfo[i_word] = (event_rb_stdinfo[i_word] & ~mask) | (n_info << bits);

      unsigned sv_index = event_unique_sv_list[i_sv];
      const auto sv_ptr = event_sv_ptrs[sv_index];
      // Store pt, (dipion) mass, FD, FD chi2
      i_word = svs_start_word + i_sv;
      float* float_info = reinterpret_cast<float*>(event_rb_stdinfo);
      if (sv_ptr->has_vertex()) {
        float_info[i_word] = sv_ptr->vertex().x();
        float_info[i_word + 1] = sv_ptr->vertex().y();
        float_info[i_word + 2] = sv_ptr->vertex().z();
        float_info[i_word + 3] = sv_ptr->vertex().chi2();
      }
    }

    // Add track information to the beginning of the bank.
    const auto tracks_start_word = svs_start_word + 4 * n_svs;
    for (unsigned i_track = 0; i_track < n_tracks; i_track++) {
      unsigned i_obj = n_sels + n_svs + i_track;
      unsigned i_word = 1 + i_obj / 4;
      unsigned i_part = i_obj % 4;
      unsigned bits = 8 * i_part;
      unsigned mask = 0xFFL << bits;
      unsigned n_info = 8;
      event_rb_stdinfo[i_word] = (event_rb_stdinfo[i_word] & ~mask) | (n_info << bits);

      unsigned track_index = event_unique_track_list[i_track];
      const auto track_ptr = event_track_ptrs[track_index];
      // Store pt, tx, ty, IP, IP chi2, muon ID, electron ID
      i_word = tracks_start_word + i_track;
      float* float_info = reinterpret_cast<float*>(event_rb_stdinfo);
      float_info[i_word] = track_ptr->state().z();
      float_info[i_word + 1] = track_ptr->state().x();
      float_info[i_word + 2] = track_ptr->state().y();
      float_info[i_word + 3] = track_ptr->state().tx();
      float_info[i_word + 4] = track_ptr->state().ty();
      float_info[i_word + 5] = track_ptr->state().qop();
      float_info[i_word + 6] = track_ptr->state().chi2() / track_ptr->state().ndof();
      float_info[i_word + 7] = static_cast<float>(track_ptr->state().ndof());
    }
  }
}

__global__ void make_subbanks::make_rb_hits(make_subbanks::Parameters parameters)
{
  const unsigned event_number = blockIdx.x;

  // Hit "sequence" here refers to the hits associated to a single track. See
  // https://gitlab.cern.ch/lhcb/LHCb/-/blob/master/Hlt/HltDAQ/HltDAQ/HltSelRepRBHits.h
  const unsigned n_hit_sequences = parameters.dev_unique_track_count[event_number];
  unsigned* event_rb_hits = parameters.dev_rb_hits + parameters.dev_rb_hits_offsets[event_number];
  const unsigned bank_info_size = 1 + (n_hit_sequences / 2);
  const unsigned n_lines = parameters.dev_number_of_active_lines[0];
  const unsigned track_offset =
    parameters.max_children_per_object * parameters.dev_max_objects_offsets[event_number * n_lines];

  // Run sequentially over tracks and in parallel over hits. There will usually
  // only be ~1 selected track anyway.
  unsigned seq_begin = bank_info_size;
  for (unsigned i_seq = 0; i_seq < n_hit_sequences; i_seq++) {
    const unsigned track_index = parameters.dev_unique_track_list[track_offset + i_seq];
    const Allen::Views::Physics::BasicParticle* track = parameters.dev_basic_particle_ptrs[track_offset + track_index];
    const unsigned n_hits = track->number_of_ids();
    unsigned* hits_insert_pointer = event_rb_hits + seq_begin;

    for (unsigned i_hit = threadIdx.x; i_hit < n_hits; i_hit += blockDim.x) {
      hits_insert_pointer[i_hit] = track->id(i_hit);
    }

    if (threadIdx.x == 0) {
      const unsigned seq_end = seq_begin + n_hits;
      unsigned i_word = (i_seq + 1) / 2;
      unsigned i_part = (i_seq + 1) % 2;
      unsigned bits = i_part * 16;
      unsigned mask = 0xFFFFL << bits;
      event_rb_hits[i_word] = (event_rb_hits[i_word] & ~mask) | (seq_end << bits);
    }

    seq_begin += n_hits;
  }

  if (threadIdx.x == 0) {
    event_rb_hits[0] = (event_rb_hits[0] & ~0xFFFFL) | n_hit_sequences;
  }
}