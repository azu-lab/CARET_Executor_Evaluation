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
#include <exception>

#include "caret_analyze_cpp_impl/record.hpp"
#include "caret_analyze_cpp_impl/common.hpp"
#include "caret_analyze_cpp_impl/column_manager.hpp"
#include "caret_analyze_cpp_impl/progress.hpp"
#include "caret_analyze_cpp_impl/records.hpp"

enum Side {Left, Right};

class UniqueList
{
public:
  void add_columns(std::vector<std::string> columns)
  {
    for (auto & column : columns) {
      if (!has(column)) {
        columns_.push_back(column);
      }
    }
  }

  std::vector<std::string> as_list()
  {
    return columns_;
  }

private:
  bool has(std::string column)
  {
    return std::count(columns_.begin(), columns_.end(), column) > 0;
  }

  std::vector<std::string> columns_;
};


RecordsBase::RecordsBase()
: columns_({})
{
}

RecordsBase::RecordsBase(const std::vector<std::string> columns)
: columns_(columns)
{
}

RecordsBase::~RecordsBase()
{
}

// RecordsBase::RecordsBase(const RecordsBase & records){
//   throw std::exception();
// }

// RecordsBase& RecordsBase::operator=(const RecordsBase & records){
//   throw std::exception();
// }

// RecordsBase& RecordsBase::operator=(RecordsBase && records){
//   throw std::exception();
// }

// RecordsBase::RecordsBase(RecordsBase && records){
//   // throw std::exception();
// }

std::unique_ptr<RecordsBase> RecordsBase::clone() const
{
  throw std::exception();
  return std::make_unique<RecordsBase>();
}

std::vector<Record> RecordsBase::get_data() const
{
  throw std::exception();
  return std::vector<Record>();
}

std::vector<std::string> RecordsBase::get_columns() const
{
  return columns_;
}


template<typename Cont, typename Pred>
Cont filter(const Cont & container, Pred predicate)
{
  Cont result;
  std::copy_if(container.begin(), container.end(), std::back_inserter(result), predicate);
  return result;
}


void RecordsBase::concat(RecordsBase & other)
{
  for (auto it = other.begin(); it->has_next(); it->next() ) {
    auto & record = it->get_record();
    append(record);
  }
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


std::unique_ptr<RecordsBase> RecordsBase::merge(
  const RecordsBase & right_records,
  std::string join_left_key,
  std::string join_right_key,
  std::vector<std::string> columns,
  std::string how,
  std::string progress_label
)
{
  // [python side implementation]
  // assert how in ["inner", "left", "right", "outer"]

  bool merge_right_record = how == "right" || how == "outer";
  bool merge_left_record = how == "left" || how == "outer";

  auto left_records_copy = this->clone();
  auto right_records_copy = right_records.clone();

  auto column_side = "_tmp_merge_side";
  auto column_merge_stamp = "_tmp_merge_stamp";
  auto column_has_valid_join_key = "_tmp_merge_has_valid_join_key";
  auto column_join_key = "_tmp_merge_join_key";
  auto column_found_right_record = "_tmp_merge_found_right_record";


  left_records_copy->append_column(
    column_side,
    std::vector<uint64_t>(left_records_copy->size(), Left)
  );

  right_records_copy->append_column(
    column_side,
    std::vector<uint64_t>(right_records_copy->size(), Right)
  );

  auto asisgn_temporal_columns = [&](Record & record, std::string & join_key) {
      auto has_valid_join_key = record.has_column(join_key);
      record.add(column_has_valid_join_key, has_valid_join_key);

      if (has_valid_join_key) {
        record.add(column_merge_stamp, record.get(join_key));
        record.add(column_join_key, record.get(join_key));
      } else {
        record.add(column_merge_stamp, UINT64_MAX);
      }
    };

  for (auto it = left_records_copy->begin(); it->has_next(); it->next()) {
    auto & record = it->get_record();
    asisgn_temporal_columns(record, join_left_key);
  }

  for (auto it = right_records_copy->begin(); it->has_next(); it->next()) {
    auto & record = it->get_record();
    asisgn_temporal_columns(record, join_right_key);
  }

  auto concat_columns = UniqueList();
  concat_columns.add_columns(left_records_copy->get_columns());
  concat_columns.add_columns(right_records_copy->get_columns());
  concat_columns.add_columns(
  {
    column_side,
    column_has_valid_join_key,
    column_merge_stamp,
    column_join_key
  });

  RecordsMapImpl concat_records(concat_columns.as_list(), {column_merge_stamp, column_side});
  concat_records.concat(*left_records_copy);
  concat_records.concat(*right_records_copy);

  std::vector<Record *> empty_records;
  std::vector<Record *> left_records_;
  std::set<uint64_t> processed_stamps;

  auto merged_records = std::make_unique<RecordsVectorImpl>(columns);

  auto bar = Progress(concat_records.size(), progress_label);
  for (auto it = concat_records.begin(); it->has_next(); it->next()) {
    auto & record = it->get_record();
    bar.tick();
    if (!record.get(column_has_valid_join_key)) {
      empty_records.push_back(&record);
      continue;
    }

    auto join_value = record.get(column_join_key);
    if (processed_stamps.count(join_value) == 0) {
      for (auto & left_record : left_records_) {
        if (left_record->get(column_found_right_record) == false) {
          empty_records.push_back(left_record);
        }
      }
      left_records_.clear();
      processed_stamps.emplace(join_value);
    }
    if (record.get(column_side) == Left) {
      record.add(column_found_right_record, false);
      left_records_.push_back(&record);
      continue;
    }

    for (auto & left_record : left_records_) {
      left_record->add(column_found_right_record, true);
      auto merged_record = record;
      merged_record.merge(*left_record);
      merged_records->append(merged_record);
    }

    if (left_records_.size() == 0) {
      empty_records.push_back(&record);
    }
  }

  for (auto & left_record : left_records_) {
    if (left_record->get(column_found_right_record) == false) {
      empty_records.push_back(left_record);
    }
  }

  for (auto & record_ptr : empty_records) {
    auto & record = *record_ptr;
    if (record.get(column_side) == Left && merge_left_record) {
      merged_records->append(record);
    } else if (record.get(column_side) == Right && merge_right_record) {
      merged_records->append(record);
    }
  }

  merged_records->drop_columns(
    {column_side, column_merge_stamp, column_join_key,
      column_has_valid_join_key, column_found_right_record});

  return merged_records;
}


std::unique_ptr<RecordsBase> RecordsBase::merge_sequential(
  const RecordsBase & right_records,
  std::string left_stamp_key,
  std::string right_stamp_key,
  std::string join_left_key,
  std::string join_right_key,
  std::vector<std::string> columns,
  std::string how,
  std::string progress_label
)
{
  auto left_records_copy = this->clone();
  auto data = left_records_copy->get_data();
  auto right_records_copy = right_records.clone();

  bool merge_left = how == "left" || how == "outer" || how == "left_use_latest";
  bool merge_right = how == "right" || how == "outer";
  bool bind_latest_left_record = how == "left_use_latest";


  auto merged_records = std::make_unique<RecordsVectorImpl>(columns);

  auto column_side = "_merge_tmp_side";
  auto column_has_valid_join_key = "merge_tmp_has_valid_join_key";
  auto column_merge_stamp = "_merge_tmp_merge_stamp";
  auto column_has_merge_stamp = "_merge_tmp_has_merge_stamp";

  left_records_copy->append_column(
    column_side,
    std::vector<uint64_t>(left_records_copy->size(), Left)
  );

  right_records_copy->append_column(
    column_side,
    std::vector<uint64_t>(right_records_copy->size(), Right)
  );

  auto assign_temporal_columns = [&](Record & record, std::string & join_key) {
      record.add(
        column_has_valid_join_key,
        join_key == "" || record.has_column(join_key)
      );

      if (record.get(column_side) == Left && record.has_column(left_stamp_key)) {
        record.add(column_merge_stamp, record.get(left_stamp_key));
        record.add(column_has_merge_stamp, true);
      } else if (record.get(column_side) == Right && record.has_column(right_stamp_key)) {
        record.add(column_merge_stamp, record.get(right_stamp_key));
        record.add(column_has_merge_stamp, true);
      } else {
        record.add(column_merge_stamp, UINT64_MAX);
        record.add(column_has_merge_stamp, false);
      }
    };

  for (auto it = left_records_copy->begin(); it->has_next(); it->next()) {
    auto & record = it->get_record();
    assign_temporal_columns(record, join_left_key);
  }
  for (auto it = right_records_copy->begin(); it->has_next(); it->next()) {
    auto & record = it->get_record();
    assign_temporal_columns(record, join_right_key);
  }

  auto concat_columns = UniqueList();
  concat_columns.add_columns(left_records_copy->get_columns());
  concat_columns.add_columns(right_records_copy->get_columns());
  concat_columns.add_columns(
  {
    column_has_merge_stamp,
    column_merge_stamp,
    column_has_merge_stamp,
  });
  RecordsMapImpl concat_records({}, concat_columns.as_list(), {column_merge_stamp, column_side});
  concat_records.concat(*left_records_copy);
  concat_records.concat(*right_records_copy);

  auto get_join_value =
    [&join_left_key, &join_right_key, &column_side](Record & record) -> uint64_t {
      std::string join_key;
      if (record.get(column_side) == Left) {
        join_key = join_left_key;
      } else {
        join_key = join_right_key;
      }
      if (join_key == "") {
        return 0;
      } else if (record.has_column(join_key)) {
        return record.get(join_key);
      } else {
        return UINT64_MAX;  // use as None
      }
    };


  // std::unordered_map<int, uint64_t> sub_empty_records;
  std::unordered_map<uint64_t, Record *> to_left_record_index;
  std::unordered_map<Record *, std::vector<Record *>> to_sub_record_indices;

  for (auto it = concat_records.begin(); it->has_next(); it->next()) {
    auto & record = it->get_record();
    if (!record.get(column_has_merge_stamp)) {
      continue;
    }

    if (record.get(column_side) == Left) {
      to_sub_record_indices[&record] = std::vector<Record *>();

      auto join_value = get_join_value(record);
      if (join_value == UINT64_MAX) {
        continue;
      }
      to_left_record_index[join_value] = &record;
    } else if (record.get(column_side) == Right) {
      auto join_value = get_join_value(record);
      if (join_value == UINT64_MAX) {
        continue;
      }

      if (to_left_record_index.count(join_value) == 0) {
        continue;
      }
      auto & left_record = *to_left_record_index[join_value];
      to_sub_record_indices[&left_record].push_back(&record);
    }
  }

  std::unordered_set<const Record *> added;

  std::size_t records_size = concat_records.size();
  auto bar = Progress(records_size, progress_label);
  for (auto it = concat_records.begin(); it->has_next(); it->next()) {
    auto & current_record = it->get_record();

    bar.tick();
    bool is_recorded = added.count(&current_record) > 0;
    if (is_recorded) {
      continue;
    }

    if (!current_record.get(column_has_merge_stamp) ||
      !current_record.get(column_has_valid_join_key))
    {
      if (current_record.get(column_side) == Left && merge_left) {
        merged_records->append(current_record);
        added.insert(&current_record);
      } else if (current_record.get(column_side) == Right && merge_right) {
        merged_records->append(current_record);
        added.insert(&current_record);
      }
      continue;
    }

    if (current_record.get(column_side) == Right) {
      if (merge_right) {
        merged_records->append(current_record);
        added.insert(&current_record);
      }
      continue;
    }

    auto sub_record_indices = to_sub_record_indices[&current_record];
    if (sub_record_indices.size() == 0) {
      if (merge_left) {
        merged_records->append(current_record);
        added.insert(&current_record);
      }
      continue;
    }

    for (uint64_t j = 0; j < sub_record_indices.size(); j++) {
      auto & sub_record = *sub_record_indices[j];
      if (1 <= j && !bind_latest_left_record) {
        break;
      }

      if (added.count(&sub_record) > 0) {
        if (merge_left) {
          merged_records->append(current_record);
          added.insert(&current_record);
        }
        continue;
      }

      Record merge_record = current_record;
      merge_record.merge(sub_record);
      merged_records->append(merge_record);
      added.insert(&current_record);
      added.insert(&sub_record);
    }
  }

  merged_records->drop_columns(
  {
    column_side,
    column_has_merge_stamp,
    column_merge_stamp,
    column_has_valid_join_key,
  }
  );

  return merged_records;
}


void RecordsBase::reindex(std::vector<std::string> columns)
{
  set_columns(columns);
}

std::unique_ptr<RecordsBase> RecordsBase::merge_sequential_for_addr_track(
  std::string source_stamp_key,
  std::string source_key,
  const RecordsBase & copy_records,
  std::string copy_stamp_key,
  std::string copy_from_key,
  std::string copy_to_key,
  const RecordsBase & sink_records,
  std::string sink_stamp_key,
  std::string sink_from_key,
  std::string progress_label
)
{
  enum RecordType { Copy, Sink, Source};
  // [python side implementation]
  // assert how in ["inner", "left", "right", "outer"]

  auto column_type = "_tmp_type";
  auto column_timestamp = "_tmp_timestamp";

  auto source_records_tmp = this->clone();
  auto copy_records_tmp = copy_records.clone();
  auto sink_records_tmp = sink_records.clone();

  source_records_tmp->append_column(
    column_type,
    std::vector<uint64_t>(source_records_tmp->size(), Source)
  );

  copy_records_tmp->append_column(
    column_type,
    std::vector<uint64_t>(copy_records_tmp->size(), Copy)
  );

  sink_records_tmp->append_column(
    column_type,
    std::vector<uint64_t>(sink_records_tmp->size(), Sink)
  );

  std::vector<uint64_t> source_stamps;
  std::vector<uint64_t> sink_stamps;
  for (auto & record : source_records_tmp->get_data()) {
    source_stamps.emplace_back(record.get(source_stamp_key));
  }
  for (auto & record : sink_records_tmp->get_data()) {
    sink_stamps.emplace_back(record.get(sink_stamp_key));
  }

  source_records_tmp->append_column(column_timestamp, source_stamps);

  copy_records_tmp->rename_columns(
    std::unordered_map<std::string, std::string>(
  {
    {copy_stamp_key, column_timestamp}
  })
  );

  sink_records_tmp->append_column(column_timestamp, sink_stamps);

  auto merged_columns = UniqueList();
  merged_columns.add_columns(source_records_tmp->get_columns());
  merged_columns.add_columns(copy_records_tmp->get_columns());
  merged_columns.add_columns(sink_records_tmp->get_columns());

  auto merged_records = std::make_unique<RecordsVectorImpl>(merged_columns.as_list());

  RecordsMapImpl concat_records({}, merged_columns.as_list(), {column_timestamp, column_type});
  concat_records.concat(*source_records_tmp);
  concat_records.concat(*copy_records_tmp);
  concat_records.concat(*sink_records_tmp);

  std::unordered_map<uint64_t, Record *> processing_records;
  using StampSet = std::set<uint64_t>;
  std::unordered_map<uint64_t, std::shared_ptr<StampSet>> stamp_sets;

  auto merge_processing_record_keys =
    [&processing_records, &stamp_sets, &column_timestamp](Record & processing_record) {
      auto condition = [&processing_record, &stamp_sets, &column_timestamp](const Record & x) {
          std::shared_ptr<StampSet> & sink_set = stamp_sets[x.get(column_timestamp)];
          std::shared_ptr<StampSet> & processing_record_set =
            stamp_sets[processing_record.get(column_timestamp)];
          std::shared_ptr<StampSet> result = std::make_shared<StampSet>();

          std::set_intersection(
            sink_set->begin(), sink_set->end(),
            processing_record_set->begin(), processing_record_set->end(),
            std::inserter(*result, result->end())
          );
          return result->size() > 0 && sink_set.get() != processing_record_set.get();
        };
      for (auto & processing_record_pair_ : processing_records) {
        auto & processing_record_ = *processing_record_pair_.second;
        if (!condition(processing_record_)) {
          continue;
        }
        std::shared_ptr<StampSet> & processing_record_keys = stamp_sets[processing_record.get(
              column_timestamp)];
        std::shared_ptr<StampSet> & corresponding_record_keys =
          stamp_sets[processing_record_.get(column_timestamp)];
        std::shared_ptr<StampSet> merged_set = std::make_shared<StampSet>();

        std::set_union(
          processing_record_keys->begin(), processing_record_keys->end(),
          corresponding_record_keys->begin(), corresponding_record_keys->end(),
          std::inserter(*merged_set, merged_set->end())
        );
        processing_record_keys = merged_set;
        corresponding_record_keys = merged_set;
      }
    };


  auto bar = Progress(concat_records.size(), progress_label);
  for (auto it = concat_records.rbegin(); it->has_next(); it->next()) {
    auto & record = it->get_record();
    record.get_columns();
    record.get_data();
    bar.tick();
    if (record.get(column_type) == Sink) {
      auto timestamp = record.get(column_timestamp);
      auto stamp_set = std::make_shared<StampSet>();
      auto addr = record.get(sink_from_key);
      stamp_set->insert(addr);
      stamp_sets.insert(std::make_pair(timestamp, stamp_set));
      processing_records[addr] = &record;
    } else if (record.get(column_type) == Copy) {
      auto condition =
        [&stamp_sets, &copy_to_key, &record, &column_timestamp](const Record & x) {
          auto timestamp = x.get(column_timestamp);
          std::shared_ptr<StampSet> stamp_set = stamp_sets[timestamp];
          bool has_same_source_addrs = stamp_set->count(record.get(copy_to_key)) > 0;
          return has_same_source_addrs;
        };
      for (auto & processing_record_pair : processing_records) {
        auto & processing_record = *processing_record_pair.second;
        if (!condition(processing_record)) {
          continue;
        }
        auto timestamp = processing_record.get(column_timestamp);
        std::shared_ptr<StampSet> stamp_set = stamp_sets[timestamp];
        stamp_set->insert(record.get(copy_from_key));
        merge_processing_record_keys(processing_record);
        // No need for subsequent loops since we integreted them.
        break;
      }
    } else if (record.get(column_type) == Source) {
      auto condition =
        [&stamp_sets, &source_key, &record, &column_timestamp](const Record & x) {
          auto timestamp = x.get(column_timestamp);
          std::shared_ptr<StampSet> stamp_set = stamp_sets[timestamp];
          bool has_same_source_addrs = stamp_set->count(record.get(source_key)) > 0;
          return has_same_source_addrs;
        };
      std::vector<uint64_t> merged_addrs;

      for (auto & processing_record_pair : processing_records) {
        auto & processing_record = *processing_record_pair.second;
        if (!condition(processing_record)) {
          continue;
        }

        processing_record.merge(record);
        merged_records->append(processing_record);
        merged_addrs.emplace_back(processing_record.get(sink_from_key));
      }
      for (auto & merged_addr : merged_addrs) {
        if (processing_records.count(merged_addr) > 0) {
          processing_records.erase(merged_addr);
        }
      }
    }
  }

  // Delete temporal columns
  merged_records->drop_columns(
  {
    column_type, column_timestamp, sink_from_key, copy_from_key, copy_to_key, copy_stamp_key});

  return merged_records;
}

std::size_t RecordsBase::size() const
{
  throw std::exception();
  return 0;
}

std::unique_ptr<IteratorBase> RecordsBase::begin()
{
  throw std::exception();
  return std::make_unique<IteratorBase>();
}

std::unique_ptr<ConstIteratorBase> RecordsBase::cbegin() const
{
  throw std::exception();
  return std::make_unique<ConstIteratorBase>();
}

std::unique_ptr<IteratorBase> RecordsBase::rbegin()
{
  throw std::exception();
  return std::make_unique<IteratorBase>();
}

std::unique_ptr<ConstIteratorBase> RecordsBase::crbegin() const
{
  throw std::exception();
  return std::make_unique<ConstIteratorBase>();
}


void RecordsBase::append_column(const std::string column, const std::vector<uint64_t> values)
{
  if (size() != values.size()) {
    throw std::exception();
  }

  columns_.push_back(column);
  auto it = begin();
  auto it_val = values.begin();
  for (; it->has_next(); it->next(), ++it_val) {
    auto & record = it->get_record();
    auto & value = *it_val;
    record.add(column, value);
  }
}

void RecordsBase::append(const Record & record)
{
  (void) record;
  throw std::exception();
}

std::vector<std::unordered_map<std::string, uint64_t>> RecordsBase::get_named_data() const
{
  std::vector<std::unordered_map<std::string, uint64_t>> data;

  for (auto it = cbegin(); it->has_next(); it->next()) {
    auto & record = it->get_record();
    std::unordered_map<std::string, uint64_t> record_tmp;
    for (auto & column : record.get_columns()) {
      auto value = record.get(column);
      record_tmp[column] = value;
    }
    data.emplace_back(record_tmp);
  }
  return data;
}

bool RecordsBase::equals(const RecordsBase & other) const
{
  auto size_equal = size() == other.size();
  if (!size_equal) {
    return false;
  }

  auto it = cbegin();
  auto it_other = other.cbegin();
  for (; it->has_next(); it->next(), it_other->next()) {
    auto record = it->get_record();
    auto other_record = it_other->get_record();
    auto is_equal = record.equals(other_record);
    if (!is_equal) {
      return false;
    }
  }
  if (get_columns() != other.get_columns()) {
    return false;
  }

  return true;
}


void RecordsBase::drop_columns(std::vector<std::string> column_names)
{
  for (auto it = begin(); it->has_next(); it->next()) {
    auto & record = it->get_record();
    record.drop_columns(column_names);
  }

  auto has_key = [&](std::string column) -> bool {
      for (auto column_name : column_names) {
        if (column == column_name) {
          return true;
        }
      }
      return false;
    };

  auto columns_tmp = columns_;
  columns_.clear();
  for (auto & column_tmp : columns_tmp) {
    if (has_key(column_tmp)) {
      continue;
    }
    columns_.push_back(column_tmp);
  }
}

void RecordsBase::filter_if(const std::function<bool(Record)> & f)
{
  (void) f;
  throw std::exception();
}

void RecordsBase::sort(std::string key, std::string sub_key, bool ascending)
{
  (void) key;
  (void) sub_key;
  (void) ascending;
  throw std::exception();
}

void RecordsBase::sort_column_order(bool ascending, bool put_none_at_top)
{
  (void) ascending;
  (void) put_none_at_top;
  throw std::exception();
}

void RecordsBase::bind_drop_as_delay()
{
  throw std::exception();
}

std::map<std::tuple<uint64_t>, std::unique_ptr<RecordsBase>> RecordsBase::groupby(
  std::string column0)
{
  std::map<std::tuple<uint64_t>, std::unique_ptr<RecordsBase>> map;

  for (auto it = begin(); it->has_next(); it->next()) {
    auto & record = it->get_record();
    auto key = std::make_tuple(
      record.get_with_default(column0, UINT64_MAX)
    );
    if (map.count(key) == 0) {
      map[key] = std::make_unique<RecordsVectorImpl>(get_columns());
    }
    auto & records = map[key];
    records->append(record);
  }

  return map;
}

std::map<std::tuple<uint64_t, uint64_t>, std::unique_ptr<RecordsBase>> RecordsBase::groupby(
  std::string column0, std::string column1)
{
  std::map<std::tuple<uint64_t, uint64_t>, std::unique_ptr<RecordsBase>> map;

  for (auto it = begin(); it->has_next(); it->next()) {
    auto & record = it->get_record();
    auto key = std::make_tuple(
      record.get_with_default(column0, UINT64_MAX),
      record.get_with_default(column1, UINT64_MAX)
    );
    if (map.count(key) == 0) {
      map[key] = std::make_unique<RecordsVectorImpl>(get_columns());
    }
    auto & records = map[key];
    records->append(record);
  }

  return map;
}

std::map<std::tuple<uint64_t, uint64_t, uint64_t>,
  std::unique_ptr<RecordsBase>> RecordsBase::groupby(
  std::string column0, std::string column1, std::string column2)
{
  std::map<std::tuple<uint64_t, uint64_t, uint64_t>, std::unique_ptr<RecordsBase>> map;

  for (auto it = begin(); it->has_next(); it->next()) {
    auto & record = it->get_record();
    auto key = std::make_tuple(
      record.get_with_default(column0, UINT64_MAX),
      record.get_with_default(column1, UINT64_MAX),
      record.get_with_default(column2, UINT64_MAX)
    );
    if (map.count(key) == 0) {
      map[key] = std::make_unique<RecordsVectorImpl>(get_columns());
    }
    auto & records = map[key];
    records->append(record);
  }

  return map;
}

void RecordsBase::set_columns(const std::vector<std::string> columns)
{
  columns_ = columns;
}

void RecordsBase::rename_columns(
  std::unordered_map<std::string, std::string> renames)
{
  for (auto it = begin(); it->has_next(); it->next()) {
    auto & record = it->get_record();
    for (auto & pair : renames) {
      record.change_dict_key(pair.first, pair.second);
    }
  }

  for (auto & column : columns_) {
    if (renames.count(column) > 0) {
      column = renames[column];
    }
  }
}
