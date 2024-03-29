###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from AllenConf.utils import initialize_number_of_events, mep_layout, make_gec
from AllenConf.hlt1_reconstruction import hlt1_reconstruction, validator_node
from AllenConf.hlt1_calibration_lines import make_passthrough_line, make_rich_1_line, make_rich_2_line
from AllenConf.hlt1_monitoring_lines import make_beam_line, make_velo_micro_bias_line, make_odin_event_type_line, make_odin_event_and_orbit_line, make_beam_gas_line
from AllenConf.hlt1_heavy_ions_lines import make_heavy_ion_event_line
from AllenConf.calo_reconstruction import decode_calo
from AllenConf.validators import rate_validation
from AllenCore.generator import make_algorithm
from PyConf.control_flow import NodeLogic, CompositeNode
from AllenConf.odin import odin_error_filter
from PyConf.tonic import configurable
from AllenConf.odin import decode_odin
from AllenConf.persistency import make_gather_selections, make_global_decision, make_sel_report_writer
from AllenConf.lumi_reconstruction import lumi_reconstruction
from AllenConf.enum_types import TrackingType, includes_matching


# Helper function to make composite nodes with the gec
def make_line_composite_node_with_gec(line_name,
                                      line_algorithm,
                                      gec_name="gec"):
    return CompositeNode(
        line_name,
        [make_gec(max_scifi_clusters=30000, count_ut=False), line_algorithm],
        NodeLogic.LAZY_AND,
        force_order=True)


@configurable
def line_maker(line_name, line_algorithm, enableGEC=True):
    if (enableGEC):
        node = make_line_composite_node_with_gec(line_name, line_algorithm)
    else:
        node = line_algorithm
    return line_algorithm, node


def default_lines(velo_tracks, forward_tracks, long_track_particles,
                  velo_states, calo_decoding, pvs):
    lines = []
    lines.append(
        line_maker(
            "Hlt1NoBeam",
            make_beam_line(
                beam_crossing_type=0,
                pre_scaler_hash_string="no_beam_line_pre",
                post_scaler_hash_string="no_beam_line_post"),
            enableGEC=False))
    lines.append(
        line_maker(
            "Hlt1BeamOne",
            make_beam_line(
                beam_crossing_type=1,
                pre_scaler_hash_string="beam_one_line_pre",
                post_scaler_hash_string="beam_one_line_post"),
            enableGEC=False))
    lines.append(
        line_maker(
            "Hlt1BeamTwo",
            make_beam_line(
                beam_crossing_type=2,
                pre_scaler_hash_string="beam_two_line_pre",
                post_scaler_hash_string="beam_two_line_post"),
            enableGEC=False))
    lines.append(
        line_maker(
            "Hlt1BothBeams",
            make_beam_line(
                beam_crossing_type=3,
                pre_scaler_hash_string="both_beams_line_pre",
                post_scaler_hash_string="both_beams_line_post"),
            enableGEC=False))
    lines.append(
        line_maker(
            "Hlt1VeloMicroBias",
            make_velo_micro_bias_line(velo_tracks),
            enableGEC=False))
    lines.append(
        line_maker(
            "Hlt1ODINLumi",
            make_odin_event_type_line(
                odin_event_type='Lumi',
                pre_scaler_hash_string="odin_lumi_line_pre",
                post_scaler_hash_string="odin_lumi_line_post"),
            enableGEC=False))
    lines.append(
        line_maker(
            "Hlt1ODINEventAndOrbitLumi",
            make_odin_event_and_orbit_line(
                odin_event_type='Lumi',
                odin_orbit_modulo=30,
                odin_orbit_remainder=1,
                pre_scaler_hash_string="odin_lumi_line_1khz_pre",
                post_scaler_hash_string="odin_lumi_line_1khz_post"),
            enableGEC=False))
    lines.append(
        line_maker(
            "Hlt1ODINNoBias",
            make_odin_event_type_line(
                odin_event_type='NoBias',
                pre_scaler_hash_string="odin_no_bias_pre",
                post_scaler_hash_string="odin_no_bias_post"),
            enableGEC=False))
    lines.append(
        line_maker(
            "Hlt1Passthrough", make_passthrough_line(), enableGEC=False))

    # lines.append(
    #     line_maker(
    #         "Hlt1RICH1Alignment",
    #         make_rich_1_line(forward_tracks, long_track_particles),
    #         enableGEC=True))
    # lines.append(
    #     line_maker(
    #         "HLt1RICH2Alignment",
    #         make_rich_2_line(forward_tracks, long_track_particles),
    #         enableGEC=True))

    lines.append(
        line_maker(
            "Hlt1BeamGas",
            make_beam_gas_line(
                velo_tracks,
                velo_states,
                pre_scaler_hash_string="no_beam_line_pre",
                post_scaler_hash_string="no_beam_line_post",
                beam_crossing_type=1),
            enableGEC=True))

    lines.append(
        line_maker(
            "Hlt1GECPassthrough",
            make_passthrough_line(
                name="Hlt1GECPassthrough",
                pre_scaler_hash_string="passthrough_with_gec_line_pre",
                post_scaler_hash_string="passthrough_with_gec_line_post",
                pre_scaler=1.)))

    lines.append(
        line_maker(
            "Hlt1PbPbMicroBiasVelo",
            make_heavy_ion_event_line(
                name="Hlt1HeavyIonPbPbMicroBias",
                pre_scaler_hash_string="PbPbMicroBias_line_pre",
                post_scaler_hash_string="PbPbMicroBias_line_post",
                velo_tracks=velo_tracks,
                pvs=pvs,
                min_pvs_PbPb=1,
                max_pvs_SMOG=0,
                calo_decoding=calo_decoding,
                pre_scaler=0.1),
            enableGEC=False))

    lines.append(
        line_maker(
            "Hlt1PbPbMBOneTrack",
            make_heavy_ion_event_line(
                name="Hlt1PbPbMBOneTrack",
                pre_scaler_hash_string="PbPbMBOneTrack_line_pre",
                post_scaler_hash_string="PbPbMBOneTrack_line_post",
                velo_tracks=velo_tracks,
                pvs=pvs,
                calo_decoding=calo_decoding,
                min_velo_tracks_PbPb=1,
                max_pvs_PbPb=0,
                pre_scaler=0.01),
            enableGEC=False))

    lines.append(
        line_maker(
            "Hlt1PbSMOGMB",
            make_heavy_ion_event_line(
                name="Hlt1PbSMOGMicroBias",
                pre_scaler_hash_string="PbSMOGMicroBias_line_pre",
                post_scaler_hash_string="PbSMOGMicroBias_line_post",
                velo_tracks=velo_tracks,
                pvs=pvs,
                calo_decoding=calo_decoding,
                max_pvs_PbPb=0,
                min_pvs_SMOG=1),
            enableGEC=False))

    lines.append(
        line_maker(
            "Hlt1PbSMOGMBOneTrack",
            make_heavy_ion_event_line(
                name="Hlt1PbSMOGMBOneTrack",
                pre_scaler_hash_string="PbSMOGOneTrack_line_pre",
                post_scaler_hash_string="PbSMOGOneTrack_line_post",
                velo_tracks=velo_tracks,
                pvs=pvs,
                calo_decoding=calo_decoding,
                min_velo_tracks_SMOG=1,
                max_pvs_PbPb=0,
                pre_scaler=0.01),
            enableGEC=False))

    lines.append(
        line_maker(
            "Hlt1PbPbPeriph",
            make_heavy_ion_event_line(
                name="Hlt1HeavyIonPbPbPeripheral",
                pre_scaler_hash_string="PbPbPeripheral_line_pre",
                post_scaler_hash_string="PbPbPeripheral_line_post",
                velo_tracks=velo_tracks,
                pvs=pvs,
                calo_decoding=calo_decoding,
                max_pvs_SMOG=0,
                min_pvs_PbPb=1,
                min_ecal_e=310000,
                max_ecal_e=14860000),
            enableGEC=False))

    lines.append(
        line_maker(
            "Hlt1PbPbCent",
            make_heavy_ion_event_line(
                name="Hlt1HeavyIonPbPbCentral",
                pre_scaler_hash_string="PbPbCentral_line_pre",
                post_scaler_hash_string="PbPbCentral_line_post",
                velo_tracks=velo_tracks,
                pvs=pvs,
                calo_decoding=calo_decoding,
                max_pvs_SMOG=0,
                min_pvs_PbPb=1,
                min_ecal_e=14860000),
            enableGEC=False))

    lines.append(
        line_maker(
            "Hlt1PbPbUPCMB",
            make_heavy_ion_event_line(
                name="Hlt1HeavyIonPbPbUPCMB",
                pre_scaler_hash_string="PbPbUPCMB_line_pre",
                post_scaler_hash_string="PbPbUPCMB_line_post",
                velo_tracks=velo_tracks,
                pvs=pvs,
                calo_decoding=calo_decoding,
                max_ecal_e=94000,
                max_pvs_PbPb=1,
                max_pvs_SMOG=0,
            ),  #treat it as random number for now as we're still trying to convert ADC to energy
            enableGEC=False))

    return lines


def setup_hlt1_node(withMCChecking=False,
                    EnableGEC=True,
                    with_lumi=True,
                    with_odin_filter=True,
                    tracking_type=TrackingType.FORWARD,
                    with_ut=True):

    hlt1_config = {}

    # Reconstruct objects needed as input for selection lines
    reconstructed_objects = hlt1_reconstruction(
        algorithm_name='pbpb_setup_hlt1_node',
        with_ut=with_ut,
        tracking_type=tracking_type)

    hlt1_config['reconstruction'] = reconstructed_objects

    calo_decoding = decode_calo()

    lines = default_lines(reconstructed_objects["velo_tracks"],
                          reconstructed_objects["long_tracks"],
                          reconstructed_objects["long_track_particles"],
                          reconstructed_objects["velo_states"], calo_decoding,
                          reconstructed_objects["pvs"])

    # list of line algorithms, required for the gather selection and DecReport algorithms
    line_algorithms = [tup[0] for tup in lines]
    # lost of line nodes, required to set up the CompositeNode
    line_nodes = [tup[1] for tup in lines]

    lines = CompositeNode(
        "SetupAllLines", line_nodes, NodeLogic.NONLAZY_OR, force_order=False)

    global_decision = make_global_decision(lines=line_algorithms)

    hlt1_node = CompositeNode(
        "Allen", [
            lines,
            global_decision,
            rate_validation(lines=line_algorithms),
            *make_sel_report_writer(lines=line_algorithms)["algorithms"],
        ],
        NodeLogic.NONLAZY_AND,
        force_order=True)

    hlt1_config['line_nodes'] = line_nodes
    hlt1_config['line_algorithms'] = line_algorithms
    hlt1_config['global_decision'] = global_decision

    if with_lumi:

        odin_err_filter = [odin_error_filter("odin_error_filter")
                           ] if with_odin_filter else []

        gather_selections = make_gather_selections(lines=line_algorithms)
        hlt1_config['gather_selections'] = gather_selections

        lumiline_name = "Hlt1ODINLumi"
        lumilinefull_name = "Hlt1ODINEventAndOrbitLumi"

        lumi_node = CompositeNode(
            "AllenLumiNode",
            lumi_reconstruction(
                gather_selections=gather_selections,
                lines=line_algorithms,
                lumiline_name=lumiline_name,
                lumilinefull_name=lumilinefull_name)["algorithms"],
            NodeLogic.NONLAZY_AND,
            force_order=False)

        lumi_with_prefilter = CompositeNode(
            "LumiWithPrefilter",
            odin_err_filter + [lumi_node],
            NodeLogic.LAZY_AND,
            force_order=True)

        hlt1_node = CompositeNode(
            "AllenWithLumi", [hlt1_node, lumi_with_prefilter],
            NodeLogic.NONLAZY_AND,
            force_order=False)

    if not withMCChecking:
        hlt1_config['control_flow_node'] = hlt1_node
    else:
        validation_node = validator_node(
            reconstructed_objects, line_algorithms,
            includes_matching(tracking_type), with_ut)

        node = CompositeNode(
            "AllenWithValidators", [hlt1_node, validation_node],
            NodeLogic.NONLAZY_AND,
            force_order=False)

        hlt1_config['control_flow_node'] = node

    return hlt1_config
