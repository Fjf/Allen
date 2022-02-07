/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
\*****************************************************************************/
#pragma once

#include <unordered_set>
#include <map>
#include <vector>
#include <cmath>
#include <mutex>

#include <gsl/gsl>
#include <Logger.h>
#include <BankTypes.h>
#include <Common.h>
#include <AllenUnits.h>


struct IInputProvider {

  enum class Layout {
    Allen,
    MEP
  };

  /// Desctructor
  virtual ~IInputProvider() {};

  /**
   * @brief      Are slices provided in MEP layout or not
   *
   * @return     layout
   */
  virtual Layout layout() const = 0;

  /**
   * @brief      Get the maximum number of events per slice
   *
   * @return     number of events per slice
   */
  virtual size_t events_per_slice() const = 0;

  /**
   * @brief      Get event ids in a given slice
   *
   * @param      slice index
   *
   * @return     event ids
   */
  virtual EventIDs event_ids(size_t slice_index, std::optional<size_t> first = {}, std::optional<size_t> last = {})
    const = 0;

  /**
   * @brief      Indicate a slice is free for filling
   *
   * @param      slice index
   */
  virtual void slice_free(size_t slice_index) = 0;

  /**
   * @brief      Get a slice with n events
   *
   * @param      optional timeout in ms to wait for slice
   *
   * @return     tuple of (success, eof, timed_out, slice_index, n_filled)
   */
  virtual std::tuple<bool, bool, bool, size_t, size_t, uint> get_slice(
    std::optional<unsigned int> timeout = {}) = 0;

  /**
   * @brief      Get banks and offsets of a given type
   *
   * @param      bank type requested
   *
   * @return     spans spanning bank and offset memory
   */
  virtual BanksAndOffsets banks(BankTypes bank_type, size_t slice_index) const = 0;

  virtual int start() = 0;

  virtual int stop() = 0;

  virtual void event_sizes(
    size_t const slice_index,
    gsl::span<unsigned int const> const selected_events,
    std::vector<size_t>& sizes) const = 0;

  virtual void copy_banks(size_t const slice_index, unsigned int const event, gsl::span<char> buffer) const = 0;
};

// InputProvider
template<class Derived>
class InputProvider;

template<template<BankTypes...> typename Derived, BankTypes... Banks>
class InputProvider<Derived<Banks...>> : public IInputProvider {
public:
  explicit InputProvider(size_t n_slices, size_t events_per_slice, Layout layout, std::optional<size_t> n_events) :
    m_layout{layout }, m_nslices {n_slices}, m_events_per_slice {events_per_slice}, m_nevents {n_events}, m_types {banks_set<Banks...>()}
  {}

  /// Descturctor
  virtual ~InputProvider() {};

  /**
   * @brief      Are slices provided in MEP layout or not
   *
   * @return     layout
   */
  Layout layout() const override { return m_layout; }

  /**
   * @brief      Get the bank types filled by this provider
   *
   * @return     unordered set of bank types
   */
  std::unordered_set<BankTypes> const& types() const { return m_types; }

  /**
   * @brief      Get the number of slices
   *
   * @return     number of slices
   */
  size_t n_slices() const { return m_nslices; }

  /**
   * @brief      Get the maximum number of events per slice
   *
   * @return     number of events per slice
   */
  size_t events_per_slice() const override { return m_events_per_slice; }

  std::optional<size_t> const& n_events() const { return m_nevents; }

  /**
   * @brief      Get event ids in a given slice
   *
   * @param      slice index
   *
   * @return     event ids
   */
  EventIDs event_ids(size_t slice_index, std::optional<size_t> first = {}, std::optional<size_t> last = {})
    const override
  {
    return static_cast<Derived<Banks...> const*>(this)->event_ids(slice_index, first, last);
  }

  /**
   * @brief      Get a slice with n events
   *
   * @param      optional timeout in ms to wait for slice
   *
   * @return     tuple of (succes, eof, timed_out, slice_index, n_filled)
   */
  std::tuple<bool, bool, bool, size_t, size_t, uint> get_slice(
    std::optional<unsigned int> timeout = {}) override
  {
    return static_cast<Derived<Banks...>*>(this)->get_slice(timeout);
  }

  /**
   * @brief      Indicate a slice is free for filling
   *
   * @param      slice index
   */
  void slice_free(size_t slice_index) override
  {
    return static_cast<Derived<Banks...>*>(this)->slice_free(slice_index);
  }

  /**
   * @brief      Get banks and offsets of a given type
   *
   * @param      bank type requested
   *
   * @return     spans spanning bank and offset memory
   */
  BanksAndOffsets banks(BankTypes bank_type, size_t slice_index) const override
  {
    return static_cast<const Derived<Banks...>*>(this)->banks(bank_type, slice_index);
  }

  void event_sizes(
    size_t const slice_index,
    gsl::span<unsigned int const> const selected_events,
    std::vector<size_t>& sizes) const override
  {
    return static_cast<const Derived<Banks...>*>(this)->event_sizes(slice_index, selected_events, sizes);
  }

  void copy_banks(size_t const slice_index, unsigned int const event, gsl::span<char> buffer) const override
  {
    return static_cast<const Derived<Banks...>*>(this)->copy_banks(slice_index, event, buffer);
  }

  int start() override { return true; };

  int stop() override { return true; };

protected:
  template<typename MSG>
  void debug_output(const MSG& msg, std::optional<size_t> const thread_id = {}) const
  {
    if (logger::verbosity() >= logger::debug) {
      std::unique_lock<std::mutex> lock {m_output_mut};
      debug_cout << (thread_id ? std::to_string(*thread_id) + " " : std::string {}) << msg << "\n";
    }
  }

private:

  // MEP layout
  const Layout m_layout = Layout::Allen;

  // Number of slices to be provided
  const size_t m_nslices = 0;

  // Number of events per slice
  const size_t m_events_per_slice = 0;

  // Optional total number of events to be provided
  const std::optional<size_t> m_nevents;

  // BankTypes provided by this provider
  const std::unordered_set<BankTypes> m_types;

  // Mutex for ordered debug output
  mutable std::mutex m_output_mut;
};

struct BufferStatus {
  bool writable = true;
  int work_counter = 0;
  std::vector<std::tuple<size_t, size_t>> intervals;
};
