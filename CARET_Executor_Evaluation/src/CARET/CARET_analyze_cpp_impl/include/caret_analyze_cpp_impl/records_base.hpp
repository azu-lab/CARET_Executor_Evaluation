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

#ifndef CARET_ANALYZE_CPP_IMPL__RECORDS_BASE_HPP_

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
#include "caret_analyze_cpp_impl/record.hpp"
#include "caret_analyze_cpp_impl/iterator_base.hpp"


class RecordsBase
{
public:
  RecordsBase();
  explicit RecordsBase(const std::vector<std::string> columns);

  virtual ~RecordsBase();

  virtual std::vector<Record> get_data() const;
  virtual std::vector<std::string> get_columns() const;
  virtual size_t size() const;

  virtual std::unique_ptr<IteratorBase> begin();
  virtual std::unique_ptr<ConstIteratorBase> cbegin() const;
  virtual std::unique_ptr<IteratorBase> rbegin();
  virtual std::unique_ptr<ConstIteratorBase> crbegin() const;

  virtual std::unique_ptr<RecordsBase> clone() const;
  void append_column(const std::string column, const std::vector<uint64_t> values);
  void rename_columns(std::unordered_map<std::string, std::string> renames);
  virtual void append(const Record & record);
  void drop_columns(std::vector<std::string> column_names);

  void concat(RecordsBase & other);
  std::vector<std::unordered_map<std::string, uint64_t>> get_named_data() const;

  void set_columns(const std::vector<std::string> columns);
  virtual bool equals(const RecordsBase & other) const;
  virtual void filter_if(const std::function<bool(Record)> & f);
  virtual void sort(std::string key, std::string sub_key = "", bool ascending = true);
  virtual void sort_column_order(bool ascending = true, bool put_none_at_top = true);
  virtual void bind_drop_as_delay();

  void reindex(std::vector<std::string> columns);
  std::map<std::tuple<uint64_t>, std::unique_ptr<RecordsBase>> groupby(
    std::string column0
  );
  std::map<std::tuple<uint64_t, uint64_t>, std::unique_ptr<RecordsBase>> groupby(
    std::string column0,
    std::string column1
  );
  std::map<std::tuple<uint64_t, uint64_t, uint64_t>, std::unique_ptr<RecordsBase>> groupby(
    std::string column0,
    std::string column1,
    std::string column2
  );

  std::unique_ptr<RecordsBase> merge(
    const RecordsBase & right_records,
    std::string join_left_key,
    std::string join_right_key,
    std::vector<std::string> columns,
    std::string how,
    std::string progress_label = ""
  );

  std::unique_ptr<RecordsBase> merge_sequential(
    const RecordsBase & right_records,
    std::string left_stamp_key,
    std::string right_stamp_key,
    std::string join_left_key,
    std::string join_right_key,
    std::vector<std::string> columns,
    std::string how,
    std::string progress_label = "");

  std::unique_ptr<RecordsBase> merge_sequential_for_addr_track(
    std::string source_stamp_key,
    std::string source_key,
    const RecordsBase & copy_records,
    std::string copy_stamp_key,
    std::string copy_from_key,
    std::string copy_to_key,
    const RecordsBase & sink_records,
    std::string sink_stamp_key,
    std::string sink_from_key,
    std::string progress_label = ""
  );

private:
  std::vector<std::string> columns_;
};

#endif  // CARET_ANALYZE_CPP_IMPL__RECORDS_BASE_HPP_
#define CARET_ANALYZE_CPP_IMPL__RECORDS_BASE_HPP_
