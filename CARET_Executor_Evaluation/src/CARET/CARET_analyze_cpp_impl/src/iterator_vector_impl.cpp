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
#include "caret_analyze_cpp_impl/iterator_vector_impl.hpp"


VectorIterator::VectorIterator(
  RecordsVectorImpl::Iterator it,
  RecordsVectorImpl::Iterator end)
: is_forward_(true), it_(it), end_(end)
{
}

VectorIterator::VectorIterator(
  RecordsVectorImpl::ReverseIterator rit,
  RecordsVectorImpl::ReverseIterator rend)
: is_forward_(false), rit_(rit), rend_(rend)
{
}

Record & VectorIterator::get_record() const
{
  if (is_forward_) {
    return *it_;
  } else {
    return *rit_;
  }
}

void VectorIterator::next()
{
  if (is_forward_) {
    it_++;
  } else {
    rit_++;
  }
}

bool VectorIterator::has_next() const
{
  if (is_forward_) {
    return it_ != end_;
  } else {
    return rit_ != rend_;
  }
}

VectorConstIterator::VectorConstIterator(
  RecordsVectorImpl::ConstIterator it,
  RecordsVectorImpl::ConstIterator end)
: is_forward_(true), it_(it), end_(end)
{
}

VectorConstIterator::VectorConstIterator(
  RecordsVectorImpl::ConstReverseIterator rit,
  RecordsVectorImpl::ConstReverseIterator rend)
: is_forward_(false), rit_(rit), rend_(rend)
{
}

const Record & VectorConstIterator::get_record() const
{
  if (is_forward_) {
    return *it_;
  } else {
    return *rit_;
  }
}

void VectorConstIterator::next()
{
  if (is_forward_) {
    it_++;
  } else {
    rit_++;
  }
}

bool VectorConstIterator::has_next() const
{
  if (is_forward_) {
    return it_ != end_;
  } else {
    return rit_ != rend_;
  }
}
