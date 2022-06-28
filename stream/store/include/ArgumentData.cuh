/*****************************************************************************\
* (c) Copyright 2022 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include <string>
#include <unordered_map>

namespace Allen::Store {

  enum class Scope { Host, Device, Invalid };

  /**
   * @brief Contains the data of an argument, namely its name, data pointer and size.
   *
   */
  struct ArgumentData {
  private:
    std::string m_name = "";
    Scope m_scope = Scope::Invalid;
    void* m_pointer = nullptr;
    size_t m_size = 0;
    size_t m_type_size = 0;

  public:
    ArgumentData() = default;
    ArgumentData(const ArgumentData&) = default;
    ArgumentData(const std::string& name) : m_name(name) {}
    ArgumentData(const std::string& name, Scope scope) : m_name(name), m_scope(scope) {}

    virtual void* pointer() const { return m_pointer; }
    virtual size_t size() const { return m_size; }
    virtual size_t sizebytes() const { return m_size * m_type_size; }
    virtual std::string name() const { return m_name; }
    virtual Scope scope() const { return m_scope; }
    virtual void set_pointer(void* pointer) { m_pointer = pointer; }
    virtual void set_size(size_t size) { m_size = size; }
    virtual void set_type_size(size_t type_size) { m_type_size = type_size; }
    virtual void set_name(const std::string& name) { m_name = name; }
    virtual void set_scope(Scope scope) { m_scope = scope; }
    virtual ~ArgumentData() {}
  };

} // namespace Allen::Store