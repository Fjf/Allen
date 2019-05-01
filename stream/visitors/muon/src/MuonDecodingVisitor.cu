#include "SequenceVisitor.cuh"
#include "MuonDecoding.cuh"
#include "MuonRawToHits.cuh"
#include "MuonTable.cuh"
//#include "MuonGeometry.сuh"

template<>
void SequenceVisitor::set_arguments_size<muon_decoding_t>(
    muon_decoding_t::arguments_t arguments,
    const RuntimeOptions& runtime_options,
    const Constants& constants,
    const HostBuffers& host_buffers) {
  arguments.set_size<dev_muon_hits>(runtime_options.number_of_events);
}

template<>
void SequenceVisitor::visit<muon_decoding_t>(
    muon_decoding_t& state,
    const muon_decoding_t::arguments_t& arguments,
    const RuntimeOptions& runtime_options,
    const Constants& constants,
    HostBuffers& host_buffers,
    cudaStream_t& cuda_stream,
    cudaEvent_t& cuda_generic_event) {

  std::string file_name_muon_table = "../../../input/muon/muon_table.bin";
  std::string file_name_muon_geometry = "../../../input/muon/muon_geometry.bin";
  char muon_table_raw_input[1200000];
  memset(muon_table_raw_input, 0, sizeof(muon_table_raw_input));
  std::ifstream muon_table_file(file_name_muon_table, std::ios::binary);
  muon_table_file.read(muon_table_raw_input, sizeof(muon_table_raw_input));
  muon_table_file.close();
  char muon_geometry_raw_input[100000];
  memset(muon_geometry_raw_input, 0, sizeof(muon_geometry_raw_input));
  std::ifstream muon_geometry_file(file_name_muon_geometry, std::ios::binary);
  muon_geometry_file.read(muon_geometry_raw_input, sizeof(muon_geometry_raw_input));
  muon_geometry_file.close();
  MuonTable pad = MuonTable();
  MuonTable stripX = MuonTable();
  MuonTable stripY = MuonTable();
  read_muon_table(muon_table_raw_input, &pad, &stripX, &stripY);
  Muon::MuonGeometry muonGeometry = Muon::MuonGeometry();
  muonGeometry.read_muon_geometry(muon_geometry_raw_input);
  MuonRawToHits muonRawToHits = MuonRawToHits(&pad, &stripX, &stripY, &muonGeometry);
  cudaCheck(cudaMemcpyAsync(
      arguments.offset<dev_muon_raw_to_hits>(),
      &muonRawToHits,
      sizeof(muonRawToHits),
      cudaMemcpyHostToDevice,
      cuda_stream
  ));


  state.set_opts(dim3(1), dim3(1), cuda_stream);
  state.set_arguments(
      runtime_options.host_muon_events,
      runtime_options.host_muon_event_offsets,
      runtime_options.host_muon_events_size,
      arguments.offset<dev_muon_raw_to_hits>(),
      arguments.offset<dev_muon_hits>()
  );
  state.invoke();
}
