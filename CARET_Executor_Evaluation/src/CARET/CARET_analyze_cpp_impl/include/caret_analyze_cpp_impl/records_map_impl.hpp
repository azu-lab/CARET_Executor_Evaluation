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

#ifndef CARET_ANALYZE_CPP_IMPL__RECORDS_MAP_IMPL_HPP_

#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <string>
#include <functional>
#include <memory>
#include <algorithm>
#include <utility>
#include <vector>
#include <iterator>

#include "caret_analyze_cpp_impl/column_manager.hpp"
#include "caret_analyze_cpp_impl/iterator_base.hpp"
#include "caret_analyze_cpp_impl/records_base.hpp"


class RecordsMapImpl : public RecordsBase
{
public:
  explicit RecordsMapImpl(const std::vector<std::string> key_columns);
  explicit RecordsMapImpl(
    const RecordsMapImpl & records,
    const std::vector<std::string> key_columns);
  RecordsMapImpl(
    std::vector<Record> records,
    const std::vector<std::string> columns,
    const std::vector<std::string> key_columns);
  explicit RecordsMapImpl(
    const std::vector<std::string> columns,
    const std::vector<std::string> key_columns);

  RecordsMapImpl(const RecordsMapImpl & records);

  ~RecordsMapImpl() override;

  using KeyT = std::tuple<std::uint64_t, uint64_t, uint64_t>;
  using DataT = std::multimap<KeyT, Record>;
  using Iterator = DataT::iterator;
  using ConstIterator = DataT::const_iterator;
  using ReverseIterator = DataT::reverse_iterator;
  using ConstReverseIterator = DataT::const_reverse_iterator;

  std::vector<Record> get_data() const override;
  void append(const Record & record) override;
  std::unique_ptr<RecordsBase> clone() const override;

  void filter_if(const std::function<bool(Record)> & f);
  void sort(std::string key, std::string sub_key = "", bool ascending = true);
  void sort_column_order(bool ascending = true, bool put_none_at_top = true);
  void bind_drop_as_delay();

  std::size_t size() const override;

  std::unique_ptr<IteratorBase> begin() override;
  std::unique_ptr<ConstIteratorBase> cbegin() const override;
  std::unique_ptr<IteratorBase> rbegin() override;
  std::unique_ptr<ConstIteratorBase> crbegin() const override;

private:
  KeyT make_key(const Record & record);

  std::unique_ptr<DataT> data_;
  std::vector<std::string> key_columns_;

  const size_t max_key_size_ = 3;
};


#endif  // CARET_ANALYZE_CPP_IMPL__RECORDS_MAP_IMPL_HPP_
#define CARET_ANALYZE_CPP_IMPL__RECORDS_MAP_IMPL_HPP_
