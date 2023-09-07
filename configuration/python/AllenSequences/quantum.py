

from AllenCore.algorithms import quantum_t
from AllenConf.validators import mc_data_provider
from AllenConf.velo_reconstruction import decode_velo
from AllenConf.utils import initialize_number_of_events
from PyConf.control_flow import CompositeNode
from AllenCore.generator import generate, make_algorithm

mc_events = mc_data_provider()
number_of_events = initialize_number_of_events()
decoded_velo = decode_velo()


quantum = make_algorithm(
    quantum_t,
    name = "quantum",
    host_number_of_events_t=number_of_events["host_number_of_events"],
    host_mc_events_t=mc_events.host_mc_events_t,
    host_total_number_of_velo_clusters_t=decoded_velo["host_total_number_of_velo_clusters"],
    dev_offsets_estimated_input_size_t=decoded_velo["dev_offsets_estimated_input_size"],
    dev_module_cluster_num_t=decoded_velo["dev_module_cluster_num"],
    dev_velo_cluster_container_t=decoded_velo["dev_sorted_velo_cluster_container"]
)

quantum_sequence = CompositeNode("Quantum", [quantum])
generate(quantum_sequence)

