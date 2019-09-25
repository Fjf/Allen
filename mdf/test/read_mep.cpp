#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <unordered_set>
#include <map>
#include <cassert>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <Logger.h>

#include <raw_bank.hpp>
#include <read_mdf.hpp>
#include <eb_header.hpp>
#include <read_mep.hpp>

using namespace std;

int main(int argc, char* argv[])
{
  if (argc != 3) {
    cout << "usage: read_mep file.mep n_mep" << endl;
    return -1;
  }

  string filename = {argv[1]};
  size_t n_meps = atol(argv[2]);

  // Some storage for reading the events into
  LHCb::MDFHeader header;
  vector<char> read_buffer(1024 * 1024, '\0');

  bool eof = false, error = false;


  int input = ::open(filename.c_str(), O_RDONLY);
  if (input != -1) {
    info_cout << "Opened " << filename << "\n";
  }
  else {
    cerr << "Failed to open file " << filename << " " << strerror(errno) << "\n";
    return -1;
  }

  vector<char> header_buffer(sizeof(LHCb::MDFHeader));
  LHCb::MDFHeader* mdf_header = reinterpret_cast<LHCb::MDFHeader*>(header_buffer.data());

  vector<char> data;

  size_t i_mep = 0;
  while (!eof && i_mep++ < n_meps) {

    auto [success, mep_header, mep_span] = MEP::read_mep(input, data);

    auto header_size = mep_header.header_size(mep_header.n_blocks);
    auto const* d = mep_span.begin() + header_size;
    size_t i_block = 0;
    while(d != mep_span.end()) {
      EB::BlockHeader const block_header{d};
      char const* block_data = d + block_header.header_size(block_header.n_frag);
      char const* block_end = block_data + block_header.block_size;

      assert(d - (mep_span.begin() + header_size) == mep_header.offsets[i_block]);

      auto lhcb_type = int(block_header.types[0]);

      cout << "fragment"
           << " packing: " << std::setw(4) << block_header.n_frag
           << " event_id: " << std::setw(6) << block_header.event_id
           << " type: " << std::setw(3) << int(block_header.types[0])
           << " source_id " << std::setw(4) << mep_header.source_ids[i_block]
           << " version: " << std::setw(2) << mep_header.versions[i_block]
           << " size: " << std::setw(6) << block_header.block_size
           << "\n";

      ++count[lhcb_type];

      d = block_end;
      ++i_block;
    }
  }

  return 0;
}
