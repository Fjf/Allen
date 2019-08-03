#pragma once

#include <regex>
#include <InputProvider.h>
#include <InputTools.h>
#include <BankTypes.h>

namespace {
  void order_files(std::vector<std::string>& files, EventIDs const& order, std::string const& folder) {
    std::map<std::tuple<int, long long>, std::string> file_ids;
    for (auto file : files) {
      file_ids.emplace(name_to_number(file), file);
    }
    files.clear();
    for (auto event_id : order) {
      auto file_it = file_ids.find(event_id);
      if (file_it != file_ids.end()) {
        files.emplace_back(std::move(file_it->second));
      } else {
        auto [run, event] = file_it->first;
        throw StrException{std::string{"file with event ID "} + std::to_string(run)
                           + " " + std::to_string(event) + " not found in " + folder};
      }
    }
  }
}

template <BankTypes... Banks>
class BinaryProvider final : public InputProvider<BinaryProvider<Banks...>> {
public:
  BinaryProvider(size_t n_slices, size_t events_per_slice, std::optional<size_t> n_events,
                 std::vector<std::string> connections,
                 bool loop = false,
                 std::optional<EventIDs> order = std::optional<EventIDs>{}) :
    InputProvider<BinaryProvider<Banks...>>{n_slices, events_per_slice, n_events},
    m_slice_free(n_slices, true),
    m_loop {loop}, m_event_ids(n_slices)
  {
    for (auto bank_type : this->types()) {
      auto it = std::find_if(connections.begin(), connections.end(),
                             [bank_type] (auto const& folder) {
                               auto bn = bank_name(bank_type);
                               return folder.substr(folder.size() - bn.size(), std::string::npos) == bn;
                             });
      if (it == connections.end()) {
        throw StrException{"Failed to find a folder for " + bank_name(bank_type) + " banks"};
      } else {
        auto ib = to_integral<BankTypes>(bank_type);
        auto contents = list_folder(*it);
        if (order) {
          order_files(contents, *order, *it);
        }
        m_files[ib] = std::tuple{*it, std::move(contents)};
      }
    }

    // Reinitialize to take the possible minimum into account
    events_per_slice = this->events_per_slice();

    auto const& some_files = std::get<1>(*m_files.begin());
    m_all_events.reserve(some_files.size());
    std::regex file_expr {"(\\d+)_(\\d+).*\\.bin"};
    std::smatch result;
    for (auto const& file : some_files) {
      if (std::regex_match(file, result, file_expr)) {
        m_all_events.emplace_back(std::tuple {std::atoi(result[1].str().c_str()), std::atol(result[2].str().c_str())});
      }
    }

    for (auto bank_type : this->types()) {
      auto it = BankSizes.find(bank_type);
      auto ib = to_integral<BankTypes>(bank_type);
      if (it == end(BankSizes)) {
        throw std::out_of_range {std::string {"Bank type "} + std::to_string(ib) + " has no known size"};
      }

      // Fudge with extra 20% memory
      size_t n_bytes = std::lround(it->second * events_per_slice * 1024 * bank_size_fudge_factor);
      auto& slices = m_slices[ib];
      slices.reserve(n_slices);
      for (size_t i = 0; i < n_slices; ++i) {
        char* events_mem = nullptr;
        uint* offsets_mem = nullptr;
        cudaCheck(cudaMallocHost((void**) &events_mem, n_bytes));
        cudaCheck(cudaMallocHost((void**) &offsets_mem, (events_per_slice + 1) * sizeof(uint)));

        offsets_mem[0] = 0;
        slices.emplace_back(gsl::span<char>{events_mem, n_bytes},
                            gsl::span<uint>{offsets_mem, events_per_slice + 1},
                            1);
      }
    }

    for (size_t n = 0; n < n_slices; ++n) {
      m_event_ids[n].reserve(events_per_slice);
    }

    // start prefetch thread
    m_prefetch_thread = std::make_unique<std::thread>([this] { prefetch(); });
  }

  static constexpr const char* name = "Binary";

  std::vector<std::tuple<unsigned int, unsigned long>> const& event_ids(size_t slice_index) const
  {
    return m_event_ids[slice_index];
  }

  // success, timed_out, slice_index, n_filled
  std::tuple<bool, bool, size_t, size_t> get_slice(std::optional<unsigned int> timeout = std::optional<unsigned int>{}) override
  {
    bool timed_out = false;
    size_t slice_index = 0, n_filled = 0;
    std::unique_lock<std::mutex> lock{m_prefetch_mut};
    if (!m_read_error && !m_prefetched.empty()) {
      if (m_prefetched.empty()) {
        if (timeout) {
          timed_out = !m_prefetch_cond.wait_for(lock, std::chrono::milliseconds{*timeout},
                                                [this] { return !m_prefetched.empty() && !m_read_error; });
        } else {
          m_prefetch_cond.wait(lock, [this] { return !m_prefetched.empty() && !m_read_error; });
        }
      }
      slice_index = m_prefetched.front();
      n_filled = std::get<2>((*m_slices.begin())[slice_index]) - 1;
      m_prefetched.pop_front();
    }

    return {!m_read_error && !m_done, timed_out, slice_index, m_read_error ? 0 : n_filled};
  }

  void slice_free(size_t slice_index) override
  {
    bool freed = false;
    {
      std::unique_lock<std::mutex> lock{m_prefetch_mut};
      if (!m_slice_free[slice_index]) {
        m_slice_free[slice_index] = true;
        freed = true;
      }
    }
    if (freed) {
      this->debug_output("freed slice " + std::to_string(slice_index));
      m_prefetch_cond.notify_one();
    }
  }

  BanksAndOffsets banks(BankTypes bank_type, size_t slice_index) const
  {
    auto ib = to_integral<BankTypes>(bank_type);
    auto const& [banks, offsets, offsets_size] = m_slices[ib][slice_index];
    span<char const> b {banks.data(), offsets[offsets_size - 1]};
    span<unsigned int const> o {offsets.data(), offsets_size};
    return BanksAndOffsets {std::move(b), std::move(o)};
  }

private:

  void prefetch() {

    bool eof = false, prefetch_done = false;
    size_t bytes_read = 0;

    auto to_read = this->n_events();
    size_t eps = this->events_per_slice();

    size_t n_files = std::get<1>(m_files.front()).size();

    while(!m_done && (!to_read || *to_read > 0)) {

      auto it = m_slice_free.end();
      {
        std::unique_lock<std::mutex> lock{m_prefetch_mut};
        it = find(m_slice_free.begin(), m_slice_free.end(), true);
        if (it == m_slice_free.end()) {
          this->debug_output("waiting for free slice", 1);
          m_prefetch_cond.wait(lock, [this, &it] {
            it = std::find(m_slice_free.begin(), m_slice_free.end(), true);
            return it != m_slice_free.end() || m_done;
          });
          if (m_done) {
            break;
          }
        }
        *it = false;
      }
      size_t slice_index = distance(m_slice_free.begin(), it);
      auto& slice = m_slices[slice_index];
      this->debug_output("got slice index " + std::to_string(slice_index), 1);

      // "Reset" the slice
      for (auto bank_type : {Banks...}) {
        auto ib = to_integral<BankTypes>(bank_type);
        std::get<1>(m_slices[ib][slice_index])[0] = 0;
        std::get<2>(m_slices[ib][slice_index]) = 1;
      }

      size_t start = m_current;

      while (!m_done && !m_read_error && m_current < start + eps) {
        auto inputs = open_files(m_current % n_files);
        if (m_read_error) break;

        auto full = std::any_of(this->types().begin(), this->types().end(), [this, slice_index, &inputs](auto bank_type) {
            auto ib = to_integral<BankTypes>(bank_type);
            const auto& [slice, offsets, offsets_size] = m_slices[ib][slice_index];
            return (offsets[offsets_size - 1] + std::get<2>(inputs[ib])) > slice.size();
          });
        if (!full) {
          for (auto& [bank_type, input, data_size] : inputs) {
            auto ib = to_integral<BankTypes>(bank_type);
            auto& [slice, offsets, offsets_size] = m_slices[ib][slice_index];
            read_file(input, data_size, slice.data(), offsets.data(), offsets_size - 1);
            ++offsets_size;
          }
          m_event_ids[slice_index].emplace_back(m_all_events[m_current]);
        }

        prefetch_done = m_current == n_files && !m_loop;
        ++m_current;
      }

      this->debug_output("read " + std::to_string(m_current - start) + " events into " + std::to_string(slice_index));

      if (!m_read_error) {
        {
          std::unique_lock<std::mutex> lock{m_prefetch_mut};
          m_prefetched.push_back(slice_index);
        }
        m_done = prefetch_done;
        this->debug_output("notifying one");
        m_prefetch_cond.notify_one();
      }
    }
  }

  std::array<std::tuple<BankTypes, std::ifstream, size_t>, NBankTypes> open_files(size_t n)
  {
    std::array<std::tuple<BankTypes, std::ifstream, size_t>, NBankTypes> result;
    for (auto bank_type : this->types()) {
      auto ib = to_integral<BankTypes>(bank_type);
      auto filename = std::get<0>(m_files[ib]) + "/" + std::get<1>(m_files[ib])[n];
      std::ifstream input(filename, std::ifstream::binary);
      if (!input.is_open() || !input.good()) {
        m_read_error = true;
        break;
      }
      input.seekg(0, std::ios::end);
      auto end = input.tellg();
      input.seekg(0, std::ios::beg);
      auto data_size = end - input.tellg();

      if (data_size == 0) {
        m_read_error = true;
        break;
      }
      result[ib] = std::tuple{bank_type, std::move(input), data_size};
    }
    return result;
  }

  void read_file(std::ifstream& input, size_t data_size, char* events, unsigned int* offsets, size_t n)
  {
    // read content
    size_t const previous_size = offsets[n];
    input.read(events + previous_size, data_size);
    offsets[n + 1] = previous_size + data_size;
  }

  // Pinned memory slices, N per banks types,
  std::array<std::vector<std::tuple<gsl::span<char>, gsl::span<unsigned int>, size_t>>, NBankTypes> m_slices;

  // data members for prefetch thread
  std::mutex m_prefetch_mut;
  std::condition_variable m_prefetch_cond;
  std::deque<size_t> m_prefetched;
  std::unique_ptr<std::thread> m_prefetch_thread;

  // Atomics to flag errors and completion
  std::atomic<bool> m_done = false;
  std::atomic<bool> m_read_error = false;

  // Keep track of what slices are free
  std::vector<bool> m_slice_free;

  // Index of the event file currently being read
  size_t m_current = 0;

  // Loop on available input files
  bool m_loop = false;

  // Run and event numbers of all available events
  std::vector<std::tuple<unsigned int, unsigned long>> m_all_events;

  // Run and events numbers of events in each slice
  std::vector<std::vector<std::tuple<unsigned int, unsigned long>>> m_event_ids;

  // Folder and file names per bank type
  std::array<std::tuple<std::string, std::vector<std::string>>, NBankTypes> m_files;

  using base_class = InputProvider<BinaryProvider<Banks...>>;


};
