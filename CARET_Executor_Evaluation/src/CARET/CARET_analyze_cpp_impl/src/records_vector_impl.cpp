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
#include <iostream>
#include <fstream>
#include <algorithm>
#include <set>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <iterator>

#include "nlohmann/json.hpp"

#include "caret_analyze_cpp_impl/record.hpp"
#include "caret_analyze_cpp_impl/common.hpp"
#include "caret_analyze_cpp_impl/column_manager.hpp"
#include "caret_analyze_cpp_impl/progress.hpp"
#include "caret_analyze_cpp_impl/records.hpp"

RecordsVectorImpl::RecordsVectorImpl(std::vector<Record> records, std::vector<std::string> columns)
: RecordsVectorImpl(columns)
{
  for (auto & record : records) {
    append(record);
  }
}

RecordsVectorImpl::RecordsVectorImpl(std::vector<std::string> columns)
: RecordsBase(columns), data_(std::make_unique<DataT>())
{
}

RecordsVectorImpl::RecordsVectorImpl()
: RecordsVectorImpl(std::vector<std::string>())
{
}

RecordsVectorImpl::~RecordsVectorImpl()
{
  data_->clear();
}

RecordsVectorImpl::RecordsVectorImpl(const RecordsVectorImpl & records)
: RecordsVectorImpl(records.get_columns())
{
  for (auto & record : records.get_data()) {
    append(record);
  }
}

RecordsVectorImpl::RecordsVectorImpl(std::string json_path)
: RecordsVectorImpl()
{
  using json = nlohmann::json;
  std::ifstream json_file(json_path.c_str());
  json records_json;
  json_file >> records_json;
  for (auto & record_json : records_json) {
    Record record;
    for (auto & elem : record_json.items()) {
      auto & key = elem.key();
      auto & value = elem.value();
      record.add(key, value);
    }
    append(record);
  }
}

void RecordsVectorImpl::bind_drop_as_delay()
{
  auto & column_manager = ColumnManager::get_instance();
  sort_column_order(false, false);

  std::unordered_map<size_t, uint64_t> oldest_values;

  for (auto & record : *data_) {
    for (auto & key : get_columns()) {
      auto hash = column_manager.get_hash(key);
      bool has_value = record.has_column(key);
      bool has_value_ = oldest_values.count(hash) > 0;
      if (!has_value && has_value_) {
        record.add(key, oldest_values[hash]);
      }
      if (has_value) {
        oldest_values[hash] = record.get(key);
      }
    }
  }

  sort_column_order(true, true);
}

void RecordsVectorImpl::append(const Record & other)
{
  data_->emplace_back(other);
}

std::unique_ptr<RecordsBase> RecordsVectorImpl::clone() const
{
  return std::make_unique<RecordsVectorImpl>(*this);
}


std::vector<Record> RecordsVectorImpl::get_data() const
{
  return *data_;
}

template<typename Cont, typename Pred>
Cont filter(const Cont & container, Pred predicate)
{
  Cont result;
  std::copy_if(container.begin(), container.end(), std::back_inserter(result), predicate);
  return result;
}

void RecordsVectorImpl::filter_if(const std::function<bool(Record)> & f)
{
  *data_ = filter(*data_, f);
}

class RecordComp
{
public:
  RecordComp(std::string key, std::string sub_key, bool ascending)
  : key_(key), sub_key_(sub_key), ascending_(ascending)
  {
  }

  bool operator()(const Record & a, const Record & b) const noexcept
  {
    if (ascending_) {
      if (a.get(key_) != b.get(key_) || sub_key_ == "") {
        return a.get(key_) < b.get(key_);
      }
      return a.get(sub_key_) < b.get(sub_key_);
    } else {
      if (a.get(key_) != b.get(key_) || sub_key_ == "") {
        return a.get(key_) > b.get(key_);
      }
      return a.get(sub_key_) > b.get(sub_key_);
    }
  }

private:
  std::string key_;
  std::string sub_key_;
  bool ascending_;
};

void RecordsVectorImpl::sort(std::string key, std::string sub_key, bool ascending)
{
  std::sort(data_->begin(), data_->end(), RecordComp{key, sub_key, ascending});
}

class RecordCompColumnOrder
{
public:
  RecordCompColumnOrder(
    std::vector<std::string> columns,
    bool ascending,
    bool put_none_at_top
  )
  : columns_(columns), ascending_(ascending)
  {
    if (ascending_) {
      if (put_none_at_top) {
        default_value_ = UINT64_MAX;
      } else {
        default_value_ = 0;
      }
    } else {
      if (put_none_at_top) {
        default_value_ = 0;
      } else {
        default_value_ = UINT64_MAX;
      }
    }
  }

  bool operator()(const Record & a, const Record & b) const noexcept
  {
    if (ascending_) {
      for (auto & column : columns_) {
        auto left = a.get_with_default(column, default_value_);
        auto right = b.get_with_default(column, default_value_);
        if (left < right) {
          return true;
        } else if (left > right) {
          return false;
        }
      }
      return true;
    } else {
      for (auto & column : columns_) {
        auto left = a.get_with_default(column, default_value_);
        auto right = b.get_with_default(column, default_value_);
        if (left > right) {
          return true;
        } else if (left < right) {
          return false;
        }
      }
      return true;
    }
  }

private:
  std::vector<std::string> columns_;
  bool ascending_;
  uint64_t default_value_;
};

void RecordsVectorImpl::sort_column_order(bool ascending, bool put_none_at_top)
{
  std::sort(
    data_->begin(),
    data_->end(), RecordCompColumnOrder{get_columns(), ascending, put_none_at_top});
}

std::size_t RecordsVectorImpl::size() const
{
  return data_->size();
}

std::unique_ptr<IteratorBase> RecordsVectorImpl::begin()
{
  return std::make_unique<VectorIterator>(data_->begin(), data_->end());
}

std::unique_ptr<ConstIteratorBase> RecordsVectorImpl::cbegin() const
{
  return std::make_unique<VectorConstIterator>(data_->begin(), data_->end());
}

std::unique_ptr<IteratorBase> RecordsVectorImpl::rbegin()
{
  return std::make_unique<VectorIterator>(data_->rbegin(), data_->rend());
}

std::unique_ptr<ConstIteratorBase> RecordsVectorImpl::crbegin() const
{
  return std::make_unique<VectorConstIterator>(data_->rbegin(), data_->rend());
}
