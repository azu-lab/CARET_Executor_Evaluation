// Copyright 2021 Research Institute of Systems Planning, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CARET_ANALYZE_CPP_IMPL__RECORD_HPP_

#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <map>
#include <algorithm>
#include <utility>
#include <iterator>

#include "caret_analyze_cpp_impl/column_manager.hpp"


class Record
{
public:
  Record();
  explicit Record(std::unordered_map<std::string, uint64_t> dict);
  Record(const Record & record);
  ~Record() = default;

  std::unordered_map<std::string, uint64_t> get_data() const;
  std::unordered_set<std::string> get_columns() const;

  void change_dict_key(std::string key_from, std::string key_to);
  bool equals(const Record & other) const;
  void merge(const Record & other);
  uint64_t get(std::string column) const;
  uint64_t get_with_default(std::string column, uint64_t default_value) const;
  void add(std::string column, uint64_t stamp);
  void drop_columns(std::vector<std::string> columns);
  bool has_column(const std::string column);

private:
  std::unordered_map<size_t, uint64_t> data_;
};


#endif  // CARET_ANALYZE_CPP_IMPL__RECORD_HPP_
#define CARET_ANALYZE_CPP_IMPL__RECORD_HPP_
