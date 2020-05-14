#include "PrepareRawBanks.cuh"

__global__ void prepare_raw_banks::prepare_raw_banks(
  prepare_raw_banks::Parameters parameters,
  const uint selected_number_of_events,
  const uint total_number_of_events,
  const uint event_start)
{
  // Handle special lines for events that don't pass the GEC.
  for (uint selected_event_number = selected_number_of_events + blockIdx.x * blockDim.x + threadIdx.x;
       selected_event_number < total_number_of_events;
       selected_event_number += blockDim.x * gridDim.x) {
    const uint event_number = parameters.dev_event_list[selected_event_number] - event_start;
    const int n_hlt1_lines = std::tuple_size<configured_lines_t>::value;
    uint32_t dec_mask = HltDecReport::decReportMasks::decisionMask;

    // Set the DecReport.
    uint32_t* event_dec_reports = parameters.dev_dec_reports + (2 + n_hlt1_lines) * event_number;
    // Continue if event has not passed any selections.
    bool pass = false;
    for (int i_line = 0; i_line < n_hlt1_lines; i_line++) {
      pass = pass || ((event_dec_reports[2 + i_line] & dec_mask) == (1 & dec_mask));
      if (pass) {
        break;
      }
    }
    if (!pass) continue;

    // Set the rest of the DecReport.
    event_dec_reports[0] = Hlt1::TCK;
    event_dec_reports[1] = Hlt1::taskID;
    uint n_decisions = 0;
    const auto lambda_fn = [&](const unsigned long i_line) {
      HltDecReport dr;
      dr.setDecision(false);
      dr.setErrorBits(0);
      dr.setNumberOfCandidates(0);
      dr.setIntDecisionID(i_line);
      event_dec_reports[2 + i_line] |= dr.getDecReport();
      if (event_dec_reports[2 + i_line] & dec_mask) {
        n_decisions++;
      }
    };
    Hlt1::TraverseLines<configured_lines_t, Hlt1::Line, decltype(lambda_fn)>::traverse(lambda_fn);

    // TODO: Handle SelReports.
    const uint event_sel_rb_stdinfo_offset = event_number * Hlt1::maxStdInfoEvent;
    uint* event_sel_rb_stdinfo = parameters.dev_sel_rb_stdinfo + event_sel_rb_stdinfo_offset;
    const uint event_sel_rb_objtyp_offset = event_number * (Hlt1::nObjTyp + 1);
    uint* event_sel_rb_objtyp = parameters.dev_sel_rb_objtyp + event_sel_rb_objtyp_offset;
    const uint event_sel_rb_substr_offset = event_number * Hlt1::subStrDefaultAllocationSize;
    uint* event_sel_rb_substr = parameters.dev_sel_rb_substr + event_sel_rb_substr_offset;

    // Populate dev_passing_event_list
    parameters.dev_passing_event_list[event_number] = true;

    // Create the substructure sub-bank.
    // Use default allocation size.
    HltSelRepRBSubstr substr_bank(0, event_sel_rb_substr);

    // Create the standard info sub-bank.
    uint nAllInfo = Hlt1::nStdInfoDecision * n_decisions;
    uint nObj = n_decisions;
    bool writeStdInfo = nAllInfo < Hlt1::maxStdInfoEvent;
    HltSelRepRBStdInfo stdinfo_bank(nObj, nAllInfo, event_sel_rb_stdinfo);

    // Create the object type sub-bank.
    HltSelRepRBObjTyp objtyp_bank(Hlt1::nObjTyp, event_sel_rb_objtyp);
    objtyp_bank.addObj(Hlt1::selectionCLID, n_decisions);
    // TODO: Check if these are actually necessary.
    objtyp_bank.addObj(Hlt1::trackCLID, 0);
    objtyp_bank.addObj(Hlt1::svCLID, 0);

    // Add special decisions to the substr bank.
    const auto lambda_special_fn = [&](const unsigned int i_line) {
      if (event_dec_reports[2 + i_line] & dec_mask) {
        // Substructure is pointers, but there are no candidates.
        stdinfo_bank.addObj(Hlt1::nStdInfoDecision);
        stdinfo_bank.addInfo(i_line);
        substr_bank.addSubstr(0, 0);
      }
    };
    Hlt1::TraverseLines<configured_lines_t, Hlt1::SpecialLine, decltype(lambda_special_fn)>::traverse(lambda_special_fn);

    // Set the sizes of the banks.
    objtyp_bank.saveSize();
    substr_bank.saveSize();
    stdinfo_bank.saveSize();
    uint selrep_size = HltSelRepRawBank::Header::kHeaderSize + objtyp_bank.size() + substr_bank.size() +
                       writeStdInfo * stdinfo_bank.size();
    parameters.dev_sel_rep_sizes[event_number] = selrep_size;
  }

  // Handle all lines for events that pass the GEC.
  for (auto selected_event_number = blockIdx.x * blockDim.x + threadIdx.x;
       selected_event_number < selected_number_of_events;
       selected_event_number += blockDim.x * gridDim.x) {
    const uint event_number = parameters.dev_event_list[selected_event_number] - event_start;

    // Create velo tracks.
    Velo::Consolidated::ConstTracks velo_tracks {parameters.dev_atomics_velo,
                                                 parameters.dev_velo_track_hit_number,
                                                 selected_event_number,
                                                 selected_number_of_events};

    // Create UT tracks.
    UT::Consolidated::ConstExtendedTracks ut_tracks {parameters.dev_atomics_ut,
                                                     parameters.dev_ut_track_hit_number,
                                                     parameters.dev_ut_qop,
                                                     parameters.dev_ut_track_velo_indices,
                                                     selected_event_number,
                                                     selected_number_of_events};

    // Create SciFi tracks.
    SciFi::Consolidated::ConstTracks scifi_tracks {parameters.dev_offsets_forward_tracks,
                                                   parameters.dev_scifi_track_hit_number,
                                                   parameters.dev_scifi_qop,
                                                   parameters.dev_scifi_states,
                                                   parameters.dev_scifi_track_ut_indices,
                                                   selected_event_number,
                                                   selected_number_of_events};

    // Tracks.
    const int* event_save_track = parameters.dev_save_track + scifi_tracks.tracks_offset(selected_event_number);
    const uint* event_saved_tracks_list =
      parameters.dev_saved_tracks_list + scifi_tracks.tracks_offset(selected_event_number);
    const ParKalmanFilter::FittedTrack* event_kf_tracks =
      parameters.dev_kf_tracks + scifi_tracks.tracks_offset(selected_event_number);

    // Vertices.
    const int* event_save_sv = parameters.dev_save_sv + parameters.dev_sv_offsets[selected_event_number];
    const uint* event_saved_svs_list = parameters.dev_saved_svs_list + parameters.dev_sv_offsets[selected_event_number];
    const VertexFit::TrackMVAVertex* event_svs =
      parameters.dev_consolidated_svs + parameters.dev_sv_offsets[selected_event_number];

    // Dec reports.
    const int n_hlt1_lines = std::tuple_size<configured_lines_t>::value;
    uint32_t* event_dec_reports = parameters.dev_dec_reports + (2 + n_hlt1_lines) * event_number;

    // Sel reports.
    const uint event_sel_rb_hits_offset =
      scifi_tracks.tracks_offset(selected_event_number) * ParKalmanFilter::nMaxMeasurements + 3 * selected_event_number;
    uint* event_sel_rb_hits = parameters.dev_sel_rb_hits + event_sel_rb_hits_offset;
    const uint event_sel_rb_stdinfo_offset = event_number * Hlt1::maxStdInfoEvent;
    uint* event_sel_rb_stdinfo = parameters.dev_sel_rb_stdinfo + event_sel_rb_stdinfo_offset;
    const uint event_sel_rb_objtyp_offset = event_number * (Hlt1::nObjTyp + 1);
    uint* event_sel_rb_objtyp = parameters.dev_sel_rb_objtyp + event_sel_rb_objtyp_offset;
    const uint event_sel_rb_substr_offset = event_number * Hlt1::subStrDefaultAllocationSize;
    uint* event_sel_rb_substr = parameters.dev_sel_rb_substr + event_sel_rb_substr_offset;

    // If any line is passed, add to selected events and create the rest of the DecReport.
    uint32_t dec_mask = HltDecReport::decReportMasks::decisionMask;

    // Continue if event has not passed any selections.
    bool pass = false;
    for (int i_line = 0; i_line < n_hlt1_lines; i_line++) {
      pass = pass || ((event_dec_reports[2 + i_line] & dec_mask) == (1 & dec_mask));
      if (pass) {
        break;
      }
    }
    if (!pass) continue;

    // Populate dev_passing_event_list
    parameters.dev_passing_event_list[event_number] = true;

    // Create the rest of the dec report.
    event_dec_reports[0] = Hlt1::TCK;
    event_dec_reports[1] = Hlt1::taskID;
    uint n_decisions = 0;
    for (int i_line = 0; i_line < n_hlt1_lines; i_line++) {
      HltDecReport dec_report;
      dec_report.setDecision(false);
      // TODO: These are all placeholder values for now.
      dec_report.setErrorBits(0);
      dec_report.setNumberOfCandidates(1);
      dec_report.setIntDecisionID(i_line);
      dec_report.setExecutionStage(1);
      event_dec_reports[2 + i_line] |= dec_report.getDecReport();
      if (event_dec_reports[2 + i_line] & dec_mask) {
        n_decisions++;
      }
    }

    // Create the hits sub-bank.
    HltSelRepRBHits hits_bank(
      parameters.dev_n_tracks_saved[event_number], parameters.dev_n_hits_saved[event_number], event_sel_rb_hits);

    // Create the substructure sub-bank.
    // Use default allocation size.
    HltSelRepRBSubstr substr_bank(0, event_sel_rb_substr);

    // Create the standard info sub-bank.
    uint nAllInfo = Hlt1::nStdInfoDecision * n_decisions +
                    Hlt1::nStdInfoTrack * (parameters.dev_n_tracks_saved[event_number]) +
                    Hlt1::nStdInfoSV * (parameters.dev_n_svs_saved[event_number]);
    uint nObj = n_decisions + parameters.dev_n_tracks_saved[event_number] + parameters.dev_n_svs_saved[event_number];
    bool writeStdInfo = nAllInfo < Hlt1::maxStdInfoEvent;
    HltSelRepRBStdInfo stdinfo_bank(nObj, nAllInfo, event_sel_rb_stdinfo);

    // Create the object type sub-bank.
    HltSelRepRBObjTyp objtyp_bank(Hlt1::nObjTyp, event_sel_rb_objtyp);
    objtyp_bank.addObj(Hlt1::selectionCLID, n_decisions);
    objtyp_bank.addObj(Hlt1::trackCLID, parameters.dev_n_tracks_saved[event_number]);
    objtyp_bank.addObj(Hlt1::svCLID, parameters.dev_n_svs_saved[event_number]);

    // Note: This was moved because it needs to be in the same order
    // as the lines in the substr
    // Add decision summaries to the StdInfo subbank. CLID = 1.
    // if (writeStdInfo) {
    //   for (uint i_line = 0; i_line < n_hlt1_lines; i_line++) {
    //     if (event_dec_reports[2 + i_line] & dec_mask) {
    //       stdinfo_bank.addObj(Hlt1::nStdInfoDecision);
    //       stdinfo_bank.addInfo(i_line);
    //     }
    //   }
    // }

    // Add one-track decisions to the substr and stdinfo.
    const auto lambda_onetrack_fn = [&](const unsigned int i_line) {
      const uint* candidate_counts = parameters.dev_candidate_counts + i_line * total_number_of_events + event_number;
      const uint* candidate_list =
        parameters.dev_candidate_lists + (i_line * total_number_of_events + event_number) * Hlt1::maxCandidates;
      // Substructure is pointers to candidates.
      if (event_dec_reports[2 + i_line] & dec_mask) {
        stdinfo_bank.addObj(Hlt1::nStdInfoDecision);
        stdinfo_bank.addInfo(i_line);
        substr_bank.addSubstr(candidate_counts[0], 0);
        for (uint i_sub = 0; i_sub < candidate_counts[0]; i_sub++) {
          substr_bank.addPtr(n_decisions + event_save_track[candidate_list[i_sub]]);
        }
      }
    };
    Hlt1::TraverseLines<configured_lines_t, Hlt1::OneTrackLine, decltype(lambda_onetrack_fn)>::traverse(lambda_onetrack_fn);

    // Add two-track decisions to the substr and stdinfo.
    const auto lambda_twotrack_fn = [&](const unsigned int i_line) {
      const uint* candidate_counts = parameters.dev_candidate_counts + i_line * total_number_of_events + event_number;
      const uint* candidate_list =
        parameters.dev_candidate_lists + (i_line * total_number_of_events + event_number) * Hlt1::maxCandidates;
      if (event_dec_reports[2 + i_line] & dec_mask) {
        stdinfo_bank.addObj(Hlt1::nStdInfoDecision);
        stdinfo_bank.addInfo(i_line);
        substr_bank.addSubstr(candidate_counts[0], 0);
        for (uint i_sub = 0; i_sub < candidate_counts[0]; i_sub++) {
          substr_bank.addPtr(n_decisions + event_save_sv[candidate_list[i_sub]]);
        }
      }
    };
    Hlt1::TraverseLines<configured_lines_t, Hlt1::TwoTrackLine, decltype(lambda_twotrack_fn)>::traverse(lambda_twotrack_fn);

    // Add special decisions to substr and stdinfo.
    const auto lambda_special_fn = [&](const unsigned int i_line) {
      if (event_dec_reports[2 + i_line] & dec_mask) {
        stdinfo_bank.addObj(Hlt1::nStdInfoDecision);
        stdinfo_bank.addInfo(i_line);
        substr_bank.addSubstr(0, 0);
      }
    };
    // Can use this lambda for both the VELO and special lines.
    Hlt1::TraverseLines<configured_lines_t, Hlt1::VeloLine, decltype(lambda_special_fn)>::traverse(lambda_special_fn);
    Hlt1::TraverseLines<configured_lines_t, Hlt1::SpecialLine, decltype(lambda_special_fn)>::traverse(lambda_special_fn);

    // Add tracks to the hits subbank and to the StdInfo. CLID = 10010.
    // TODO: dev_n_tracks_saved was 0s at the beginning! ./Allen -m3
    for (uint i_saved_track = 0; i_saved_track < parameters.dev_n_tracks_saved[event_number]; i_saved_track++) {
      uint i_track = event_saved_tracks_list[i_saved_track];
      // Add track parameters to StdInfo.
      if (writeStdInfo) {
        stdinfo_bank.addObj(Hlt1::nStdInfoTrack);
        stdinfo_bank.addInfo(event_kf_tracks[i_track].state[0]);
        stdinfo_bank.addInfo(event_kf_tracks[i_track].state[1]);
        stdinfo_bank.addInfo(event_kf_tracks[i_track].z);
        stdinfo_bank.addInfo(event_kf_tracks[i_track].state[2]);
        stdinfo_bank.addInfo(event_kf_tracks[i_track].state[3]);
        stdinfo_bank.addInfo(event_kf_tracks[i_track].state[4]);
      }

      // Add to Substr.
      // No substructure. Ptr will be to hits.
      substr_bank.addSubstr(0, 1);
      substr_bank.addPtr(i_saved_track);

      // Create the tracks for saving.
      const int i_ut_track = scifi_tracks.ut_track(i_track);
      const int i_velo_track = ut_tracks.velo_track(i_ut_track);
      const uint n_hits = scifi_tracks.number_of_hits(i_track) + ut_tracks.number_of_hits(i_ut_track) +
                          velo_tracks.number_of_hits(i_velo_track);
      uint begin = hits_bank.addSeq(n_hits);
      SciFi::Consolidated::ConstHits scifi_hits = scifi_tracks.get_hits(parameters.dev_scifi_track_hits, i_track);
      UT::Consolidated::ConstHits ut_hits = ut_tracks.get_hits(parameters.dev_ut_track_hits, i_ut_track);
      Velo::Consolidated::ConstHits velo_hits = velo_tracks.get_hits(parameters.dev_velo_track_hits, i_velo_track);

      // Add the velo hits.
      // NB: these are stored in backwards order.
      uint i_hit = 0;
      for (uint i_velo_hit = 0; i_velo_hit < velo_tracks.number_of_hits(i_velo_track); i_velo_hit++) {
        hits_bank.m_location[begin + i_hit] = velo_hits.id(velo_tracks.number_of_hits(i_velo_track) - 1 - i_velo_hit);
        i_hit++;
      }
      // Add UT hits.
      for (uint i_ut_hit = 0; i_ut_hit < ut_tracks.number_of_hits(i_ut_track); i_ut_hit++) {
        hits_bank.m_location[begin + i_hit] = ut_hits.id(i_ut_hit);
        i_hit++;
      }
      // Add SciFi hits.
      for (uint i_scifi_hit = 0; i_scifi_hit < scifi_tracks.number_of_hits(i_track); i_scifi_hit++) {
        hits_bank.m_location[begin + i_hit] = scifi_hits.id(i_scifi_hit);
        i_hit++;
      }
    }

    // Add secondary vertices to the hits StdInfo. CLID = 10030.
    for (uint i_saved_sv = 0; i_saved_sv < parameters.dev_n_svs_saved[event_number]; i_saved_sv++) {
      uint i_sv = event_saved_svs_list[i_saved_sv];

      // Add to Substr.
      int i_track = event_svs[i_sv].trk1;
      int j_track = event_svs[i_sv].trk2;
      // Two substructures, pointers to objects.
      substr_bank.addSubstr(2, 0);
      substr_bank.addPtr(n_decisions + event_save_track[i_track]);
      substr_bank.addPtr(n_decisions + event_save_track[j_track]);

      // Add to StdInfo.
      if (writeStdInfo) {
        stdinfo_bank.addObj(Hlt1::nStdInfoSV);
        stdinfo_bank.addInfo(event_svs[i_sv].x);
        stdinfo_bank.addInfo(event_svs[i_sv].y);
        stdinfo_bank.addInfo(event_svs[i_sv].z);
      }
    }

    // Get the sizes of the banks.
    objtyp_bank.saveSize();
    substr_bank.saveSize();
    stdinfo_bank.saveSize();
    uint selrep_size = HltSelRepRawBank::Header::kHeaderSize + hits_bank.size() + objtyp_bank.size() +
                       substr_bank.size() + writeStdInfo * stdinfo_bank.size();
    parameters.dev_sel_rep_sizes[event_number] = selrep_size;
  }
}
