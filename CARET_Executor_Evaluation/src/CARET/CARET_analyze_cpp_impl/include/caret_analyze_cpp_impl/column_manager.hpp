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

#ifndef CARET_ANALYZE_CPP_IMPL__COLUMN_MANAGER_HPP_

#include <string>
#include <memory>
#include <unordered_map>

class ColumnManager
{
public:
  ColumnManager(const ColumnManager &) = delete;
  ColumnManager & operator=(const ColumnManager &) = delete;
  ColumnManager(ColumnManager &&) = delete;
  ColumnManager & operator=(ColumnManager &&) = delete;

  static ColumnManager & get_instance();

  void register_column(std::string column);
  std::string get_column(size_t hash);
  size_t get_hash(std::string column);

private:
  ColumnManager() = default;
  ~ColumnManager() = default;
  std::unordered_map<size_t, std::string> column_map_;
  std::unordered_map<std::string, size_t> hash_map_;
};

#endif  // CARET_ANALYZE_CPP_IMPL__COLUMN_MANAGER_HPP_
#define CARET_ANALYZE_CPP_IMPL__COLUMN_MANAGER_HPP_
