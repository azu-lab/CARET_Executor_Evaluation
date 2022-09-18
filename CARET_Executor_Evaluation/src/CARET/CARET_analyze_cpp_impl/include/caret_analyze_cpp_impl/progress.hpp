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

#ifndef CARET_ANALYZE_CPP_IMPL__PROGRESS_HPP_

#include <memory>
#include <string>

#include "indicators/progress_spinner.hpp"

class Progress
{
public:
  explicit Progress(std::size_t max_progress, std::string label = "", float print_freq_limit = 10);
  ~Progress();
  void tick();

private:
  std::size_t max_progress_;
  std::size_t tick_count_;
  std::shared_ptr<indicators::ProgressSpinner> progress_;
  float print_freq_limit_;
  std::chrono::system_clock::time_point last_;
  bool enable_;
};

#endif  // CARET_ANALYZE_CPP_IMPL__PROGRESS_HPP_
#define CARET_ANALYZE_CPP_IMPL__PROGRESS_HPP_
