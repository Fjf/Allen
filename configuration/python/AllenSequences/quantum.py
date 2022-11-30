

from AllenCore.algorithms import quantum_t
from AllenConf.velo_reconstruction import decode_velo, make_velo_tracks
from AllenConf.utils import initialize_number_of_events
from PyConf.control_flow import CompositeNode
from AllenCore.generator import generate, make_algorithm

number_of_events = initialize_number_of_events()
decoded_velo = decode_velo()
velo_tracks = make_velo_tracks(decoded_velo)

print(quantum_t.getDefaultProperties())

saxpy = make_algorithm(
    quantum_t,
    name = "quantum",
    host_number_of_events_t = number_of_events["host_number_of_events"],
    dev_number_of_events_t = number_of_events["dev_number_of_events"],
    dev_offsets_all_velo_tracks_t = velo_tracks["dev_offsets_all_velo_tracks"],
    dev_offsets_velo_track_hit_number_t = velo_tracks["dev_offsets_velo_track_hit_number"])

saxpy_sequence = CompositeNode("Saxpy", [saxpy])
generate(saxpy_sequence)

