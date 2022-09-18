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

#include "caret_analyze_cpp_impl/records.hpp"
#include "caret_analyze_cpp_impl/iterator_map_impl.hpp"


MapIterator::MapIterator(RecordsMapImpl::Iterator it, RecordsMapImpl::Iterator end)
: is_forward_(true), it_(it), end_(end)
{
}

MapIterator::MapIterator(RecordsMapImpl::ReverseIterator rit, RecordsMapImpl::ReverseIterator rend)
: is_forward_(false), rit_(rit), rend_(rend)
{
}

Record & MapIterator::get_record() const
{
  if (is_forward_) {
    return (*it_).second;
  } else {
    return (*rit_).second;
  }
}


void MapIterator::next()
{
  if (is_forward_) {
    it_++;
  } else {
    rit_++;
  }
}

bool MapIterator::has_next() const
{
  if (is_forward_) {
    return it_ != end_;
  } else {
    return rit_ != rend_;
  }
}

MapConstIterator::MapConstIterator(
  RecordsMapImpl::ConstIterator it,
  RecordsMapImpl::ConstIterator end)
: is_forward_(true), it_(it), end_(end)
{
}

MapConstIterator::MapConstIterator(
  RecordsMapImpl::ConstReverseIterator rit,
  RecordsMapImpl::ConstReverseIterator rend)
: is_forward_(false), rit_(rit), rend_(rend)
{
}

const Record & MapConstIterator::get_record() const
{
  if (is_forward_) {
    return (*it_).second;
  } else {
    return (*rit_).second;
  }
}


void MapConstIterator::next()
{
  if (is_forward_) {
    it_++;
  } else {
    rit_++;
  }
}

bool MapConstIterator::has_next() const
{
  if (is_forward_) {
    return it_ != end_;
  } else {
    return rit_ != rend_;
  }
}
