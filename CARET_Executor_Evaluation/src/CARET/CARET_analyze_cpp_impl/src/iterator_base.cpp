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


Record & IteratorBase::get_record() const
{
  throw std::exception();
  static Record record;
  return record;
}

void IteratorBase::next()
{
  throw std::exception();
}

bool IteratorBase::has_next() const
{
  throw std::exception();
  return false;
}

const Record & ConstIteratorBase::get_record() const
{
  throw std::exception();
  static Record record;
  return record;
}

void ConstIteratorBase::next()
{
  throw std::exception();
}

bool ConstIteratorBase::has_next() const
{
  throw std::exception();
  return false;
}
