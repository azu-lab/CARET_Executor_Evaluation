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

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <string>
#include <limits>
#include <utility>
#include <functional>

#include "caret_analyze_cpp_impl/column_manager.hpp"
#include "caret_analyze_cpp_impl/record.hpp"

Record::Record()
{
}

Record::Record(std::unordered_map<std::string, uint64_t> init)
{
  for (auto & pair : init) {
    add(pair.first, pair.second);
  }
}

Record::Record(const Record & record)
: data_(record.data_)
{
}

std::unordered_map<std::string, uint64_t> Record::get_data() const
{
  auto & column_manager = ColumnManager::get_instance();

  std::unordered_map<std::string, uint64_t> data;
  for (auto pair : data_) {
    auto column = column_manager.get_column(pair.first);
    data[column] = get(column);
  }
  return data;
}

uint64_t Record::get(std::string column) const
{
  auto & column_manager = ColumnManager::get_instance();
  auto hash = column_manager.get_hash(column);
  return data_.at(hash);
}

uint64_t Record::get_with_default(std::string column, uint64_t default_value) const
{
  auto & column_manager = ColumnManager::get_instance();
  auto hash = column_manager.get_hash(column);

  if (data_.count(hash) > 0) {
    return data_.at(hash);
  }

  return default_value;
}

void Record::change_dict_key(std::string key_from, std::string key_to)
{
  auto & column_manager = ColumnManager::get_instance();
  auto h_from = column_manager.get_hash(key_from);
  auto h_to = column_manager.get_hash(key_to);

  if (data_.count(h_from) == 0) {
    return;
  }
  data_.insert(std::make_pair(h_to, data_[h_from]));
  data_.erase(h_from);
}

void Record::drop_columns(std::vector<std::string> columns)
{
  auto & column_manager = ColumnManager::get_instance();
  for (auto & column : columns) {
    auto hash = column_manager.get_hash(column);
    data_.erase(hash);
  }
}

bool Record::equals(const Record & other) const
{
  return this->data_ == other.data_;
}

void Record::add(std::string column, uint64_t stamp)
{
  auto & column_manager = ColumnManager::get_instance();
  auto hash = column_manager.get_hash(column);
  data_[hash] = stamp;
}

void Record::merge(const Record & other)
{
  auto & column_manager = ColumnManager::get_instance();

  for (auto & pair : other.data_) {
    auto column = column_manager.get_column(pair.first);
    add(column, pair.second);
  }
}

bool Record::has_column(const std::string column)
{
  auto & column_manager = ColumnManager::get_instance();
  auto hash = column_manager.get_hash(column);
  return data_.count(hash) > 0;
}

std::unordered_set<std::string> Record::get_columns() const
{
  auto & column_manager = ColumnManager::get_instance();
  std::unordered_set<std::string> columns;
  for (auto pair : data_) {
    auto key = column_manager.get_column(pair.first);
    columns.emplace(key);
  }
  return columns;
}
