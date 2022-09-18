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

#ifndef CARET_ANALYZE_CPP_IMPL__ITERATOR_BASE_HPP_

#include "caret_analyze_cpp_impl/record.hpp"

class IteratorBase
{
public:
  virtual Record & get_record() const;
  virtual void next();
  virtual bool has_next() const;
};

class ConstIteratorBase
{
public:
  virtual const Record & get_record() const;
  virtual void next();
  virtual bool has_next() const;
};

#endif  // CARET_ANALYZE_CPP_IMPL__ITERATOR_BASE_HPP_
#define CARET_ANALYZE_CPP_IMPL__ITERATOR_BASE_HPP_
