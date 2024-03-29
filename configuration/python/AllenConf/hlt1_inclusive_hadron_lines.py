###############################################################################
# (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
from AllenCore.algorithms import (
    kstopipi_line_t, track_mva_line_t, two_track_mva_line_t,
    two_track_mva_evaluator_t, two_track_line_ks_t, lambda2ppi_line_t)
from AllenConf.utils import initialize_number_of_events
from AllenCore.generator import make_algorithm
from AllenCore.configuration_options import is_allen_standalone


def make_kstopipi_line(long_tracks,
                       secondary_vertices,
                       pre_scaler_hash_string=None,
                       post_scaler_hash_string=None,
                       name='Hlt1KsToPiPi_{hash}',
                       double_muon_misid=False,
                       enable_monitoring=True):
    number_of_events = initialize_number_of_events()

    return make_algorithm(
        kstopipi_line_t,
        name=name,
        double_muon_misid=double_muon_misid,
        enable_monitoring=is_allen_standalone() and enable_monitoring,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_number_of_svs_t=secondary_vertices["host_number_of_svs"],
        dev_particle_container_t=secondary_vertices[
            "dev_multi_event_composites"],
        pre_scaler_hash_string=pre_scaler_hash_string or name + "_pre",
        post_scaler_hash_string=post_scaler_hash_string or name + "_post")


def make_track_mva_line(long_tracks,
                        long_track_particles,
                        pre_scaler_hash_string=None,
                        post_scaler_hash_string=None,
                        name='Hlt1TrackMVA_{hash}',
                        enable_tupling=False):
    number_of_events = initialize_number_of_events()

    return make_algorithm(
        track_mva_line_t,
        name=name,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_number_of_reconstructed_scifi_tracks_t=long_tracks[
            "host_number_of_reconstructed_scifi_tracks"],
        dev_particle_container_t=long_track_particles[
            "dev_multi_event_basic_particles"],
        pre_scaler_hash_string=pre_scaler_hash_string or name + "_pre",
        post_scaler_hash_string=post_scaler_hash_string or name + "_post",
        enable_tupling=enable_tupling)


def make_two_track_mva_line(long_tracks,
                            secondary_vertices,
                            pre_scaler_hash_string=None,
                            post_scaler_hash_string=None,
                            name='Hlt1TwoTrackMVA_{hash}',
                            enable_tupling=False):
    number_of_events = initialize_number_of_events()

    two_track_mva_evaluator = make_algorithm(
        two_track_mva_evaluator_t,
        name='two_track_mva_evaluator_{hash}',
        dev_consolidated_svs_t=secondary_vertices["dev_consolidated_svs"],
        dev_sv_offsets_t=secondary_vertices["dev_sv_offsets"],
        host_number_of_svs_t=secondary_vertices["host_number_of_svs"])

    return make_algorithm(
        two_track_mva_line_t,
        name=name,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_number_of_svs_t=secondary_vertices["host_number_of_svs"],
        dev_particle_container_t=secondary_vertices[
            "dev_multi_event_composites"],
        pre_scaler_hash_string=pre_scaler_hash_string or name + "_pre",
        post_scaler_hash_string=post_scaler_hash_string or name + "_post",
        dev_two_track_mva_evaluation_t=two_track_mva_evaluator.
        dev_two_track_mva_evaluation_t,
        enable_tupling=enable_tupling)


def make_two_track_line_ks(long_tracks,
                           secondary_vertices,
                           pre_scaler_hash_string=None,
                           post_scaler_hash_string=None,
                           name='Hlt1TwoTrackKs_{hash}',
                           enable_tupling=False):
    number_of_events = initialize_number_of_events()

    return make_algorithm(
        two_track_line_ks_t,
        name=name,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_number_of_svs_t=secondary_vertices["host_number_of_svs"],
        dev_particle_container_t=secondary_vertices[
            "dev_multi_event_composites"],
        pre_scaler_hash_string=pre_scaler_hash_string or name + "_pre",
        post_scaler_hash_string=post_scaler_hash_string or name + "_post",
        enable_tupling=enable_tupling)


def make_lambda2ppi_line(secondary_vertices,
                         name="Hlt1L02PPi",
                         pre_scaler_hash_string=None,
                         post_scaler_hash_string=None,
                         enable_tupling=False):

    number_of_events = initialize_number_of_events()

    return make_algorithm(
        lambda2ppi_line_t,
        name=name,
        host_number_of_events_t=number_of_events["host_number_of_events"],
        host_number_of_svs_t=secondary_vertices["host_number_of_svs"],
        dev_particle_container_t=secondary_vertices[
            "dev_multi_event_composites"],
        pre_scaler_hash_string=pre_scaler_hash_string or name + '_pre',
        post_scaler_hash_string=post_scaler_hash_string or name + '_post',
        enable_tupling=enable_tupling)
