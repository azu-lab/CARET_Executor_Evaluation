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

#include <iostream>
#include <string>

#include "caret_analyze_cpp_impl/column_manager.hpp"


ColumnManager & ColumnManager::get_instance()
{
  static ColumnManager instance;
  return instance;
}

std::string ColumnManager::get_column(size_t hash)
{
  if (column_map_.count(hash) == 0) {
    std::cerr << "Unknown hash value" << std::endl;
  }
  return column_map_[hash];
}

size_t ColumnManager::get_hash(std::string column)
{
  if (hash_map_.count(column) == 0) {
    register_column(column);
  }
  return hash_map_[column];
}

void ColumnManager::register_column(std::string column)
{
  size_t hash;
  hash = std::hash<std::string>()(column);
  while (column_map_.count(hash) > 0) {
    hash++;
  }
  column_map_[hash] = column;
  hash_map_[column] = hash;
}
