from definitions.VeloSequence import VeloSequence
from definitions.PVSequence import PVSequence
from definitions.UTSequence import UTSequence
from definitions.algorithms import compose_sequences

velo_sequence = VeloSequence()

pv_sequence = PVSequence(
  initialize_lists = velo_sequence["initialize_lists"],
  velo_copy_track_hit_number = velo_sequence["velo_copy_track_hit_number"],
  velo_consolidate_tracks = velo_sequence["velo_consolidate_tracks"],
  prefix_sum_offsets_velo_track_hit_number = velo_sequence["prefix_sum_offsets_velo_track_hit_number"])

ut_sequence = UTSequence(
  initialize_lists = velo_sequence["initialize_lists"],
  velo_copy_track_hit_number = velo_sequence["velo_copy_track_hit_number"],
  velo_consolidate_tracks = velo_sequence["velo_consolidate_tracks"],
  prefix_sum_offsets_velo_track_hit_number = velo_sequence["prefix_sum_offsets_velo_track_hit_number"])

compose_sequences(velo_sequence, pv_sequence, ut_sequence).generate()