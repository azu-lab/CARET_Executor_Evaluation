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

#include <iostream>
#include <string>

#include "caret_analyze_cpp_impl/records.hpp"

void print_records(const RecordsBase & records);
void run_merge(std::string how);
void run_merge_with_drop(std::string how);
void run_merge_sequential_for_addr_track();
void run_merge_sequential_with_key(std::string how);
void run_merge_sequential_with_loss(std::string how);
void run_merge_sequential_without_key(std::string how);

int main(int argc, char ** argvs)
{
  (void) argc;
  (void) argvs;
  run_merge("inner");
  // run_merge("left");
  // run_merge("right");
  // run_merge("outer");

  // run_merge_with_drop("inner");
  // run_merge_with_drop("left");
  // run_merge_with_drop("right");
  // run_merge_with_drop("outer");

  // run_merge_sequential_for_addr_track();

  // run_merge_sequential_with_key("inner");
  // run_merge_sequential_without_key("inner");
  // run_merge_sequential_with_loss("inner");
  return 0;
}

void print_records(const RecordsBase & records)
{
  for (auto & record : records.get_named_data()) {
    for (auto & pair : record) {
      std::cout << pair.first << " " << pair.second << ", ";
    }
    std::cout << std::endl;
  }
}


void run_merge(std::string how)
{
  // RecordsVectorImpl left_records;
  // RecordsMapImpl left_records({"stamp"});
  // left_records.append(Record({{"stamp", 0}, {"value", 1}}));
  // left_records.append(Record({{"stamp", 2}, {"value", 2}}));
  // left_records.append(Record({{"stamp", 3}, {"value", 3}}));



  // RecordsVectorImpl right_records;
  // right_records.append(Record({{"stamp_", 4}, {"value", 2}}));
  // right_records.append(Record({{"stamp_", 5}, {"value", 3}}));
  // right_records.append(Record({{"stamp_", 6}, {"value", 4}}));

  RecordsMapImpl left_records({"stamp"});
  left_records.append(Record({{"stamp", 1}, {"value", 10}}));
  left_records.append(Record({{"stamp", 3}, {"value", 20}}));
  left_records.append(Record({{"stamp", 5}, {"value", 30}}));
  left_records.append(Record({{"stamp", 5}, {"value", 40}}));

  RecordsVectorImpl right_records;
  right_records.append(Record({{"stamp_", 2}, {"value", 10}}));
  right_records.append(Record({{"stamp_", 4}, {"value", 20}}));
  right_records.append(Record({{"stamp_", 6}, {"value", 30}}));
  right_records.append(Record({{"stamp_", 6}, {"value", 30}}));
  right_records.append(Record({{"stamp_", 10}, {"value", 50}}));

  auto merged_records = left_records.merge(
    right_records,
    "value",
    "value",
    {"stamp", "value", "stamp"},
    how
  );

  print_records(*merged_records);
}


void run_merge_with_drop(std::string how)
{
  RecordsVectorImpl left_records;
  left_records.append(Record({{"other_stamp", 4}, {"stamp", 1}, {"value", 1}}));
  left_records.append(Record({{"other_stamp", 8}}));
  left_records.append(Record({{"other_stamp", 12}, {"stamp", 9}, {"value", 2}}));
  left_records.append(Record({{"other_stamp", 16}}));

  RecordsVectorImpl right_records;
  right_records.append(Record({{"other_stamp_", 2}, {"stamp_", 3}, {"value", 2}}));
  right_records.append(Record({{"other_stamp_", 6}, {"stamp_", 7}, {"value", 1}}));
  right_records.append(Record({{"other_stamp_", 10}}));
  right_records.append(Record({{"other_stamp_", 14}}));

  auto merged_records = left_records.merge(
    right_records,
    "value",
    "value",
    {"other_stamp", "stamp", "value", "other_stamp_", "stamp_"},
    how
  );

  print_records(*merged_records);
}

void run_merge_sequential_for_addr_track()
{
  RecordsVectorImpl source_records;
  source_records.append(Record({{"source_addr", 1}, {"source_stamp", 0}}));
  source_records.append(Record({{"source_addr", 1}, {"source_stamp", 10}}));
  source_records.append(Record({{"source_addr", 3}, {"source_stamp", 20}}));

  RecordsVectorImpl copy_records;
  copy_records.append(Record({{"addr_from", 1}, {"addr_to", 13}, {"copy_stamp", 1}}));
  copy_records.append(Record({{"addr_from", 1}, {"addr_to", 13}, {"copy_stamp", 11}}));
  copy_records.append(Record({{"addr_from", 3}, {"addr_to", 13}, {"copy_stamp", 21}}));

  RecordsVectorImpl sink_records;
  sink_records.append(Record({{"sink_addr", 13}, {"sink_stamp", 2}}));
  sink_records.append(Record({{"sink_addr", 1}, {"sink_stamp", 3}}));
  sink_records.append(Record({{"sink_addr", 13}, {"sink_stamp", 12}}));
  sink_records.append(Record({{"sink_addr", 13}, {"sink_stamp", 22}}));
  sink_records.append(Record({{"sink_addr", 3}, {"sink_stamp", 23}}));
  sink_records.append(Record({{"sink_addr", 13}, {"sink_stamp", 24}}));
  sink_records.append(Record({{"sink_addr", 3}, {"sink_stamp", 25}}));

  auto merged_records = source_records.merge_sequential_for_addr_track(
    "source_stamp",
    "source_addr",
    copy_records,
    "copy_stamp",
    "addr_from",
    "addr_to",
    sink_records,
    "sink_stamp",
    "sink_addr"
  );

  print_records(*merged_records);
}

void run_merge_sequential_with_key(std::string how)
{
  RecordsVectorImpl left_records;
  left_records.append(Record({{"key", 1}, {"stamp", 0}}));
  left_records.append(Record({{"key", 2}, {"stamp", 1}}));
  left_records.append(Record({{"key", 1}, {"stamp", 6}}));
  left_records.append(Record({{"key", 2}, {"stamp", 7}}));

  RecordsVectorImpl right_records;
  right_records.append(Record({{"key", 2}, {"sub_stamp", 2}}));
  right_records.append(Record({{"key", 1}, {"sub_stamp", 3}}));
  right_records.append(Record({{"key", 1}, {"sub_stamp", 4}}));
  right_records.append(Record({{"key", 2}, {"sub_stamp", 5}}));

  auto merged_records = left_records.merge_sequential(
    right_records, "stamp", "sub_stamp", "key",
    "key", {"key", "stamp", "sub_stamp"},
    how);

  print_records(*merged_records);
}

void run_merge_sequential_without_key(std::string how)
{
  RecordsVectorImpl left_records;
  left_records.append(Record({{"stamp", 0}}));
  left_records.append(Record({{"stamp", 3}}));
  left_records.append(Record({{"stamp", 4}}));
  left_records.append(Record({{"stamp", 5}}));
  left_records.append(Record({{"stamp", 8}}));

  RecordsVectorImpl right_records;
  right_records.append(Record({{"sub_stamp", 1}}));
  right_records.append(Record({{"sub_stamp", 6}}));
  right_records.append(Record({{"sub_stamp", 7}}));

  auto merged_records = left_records.merge_sequential(
    right_records, "stamp", "sub_stamp",
    "", "", {"stamp", "sub_stamp"}, how);

  print_records(*merged_records);
}


void run_merge_sequential_with_loss(std::string how)
{
  RecordsVectorImpl left_records;
  left_records.append(Record({{"other_stamp", 4}, {"stamp", 1}, {"value", 1}}));
  left_records.append(Record({{"other_stamp", 8}}));
  left_records.append(Record({{"other_stamp", 12}, {"stamp", 9}, {"value", 1}}));
  left_records.append(Record({{"other_stamp", 16}}));

  RecordsVectorImpl right_records;
  right_records.append(Record({{"other_stamp_", 2}, {"stamp_", 3}, {"value", 1}}));
  right_records.append(Record({{"other_stamp_", 6}, {"stamp_", 7}, {"value", 1}}));
  right_records.append(Record({{"other_stamp_", 10}}));
  right_records.append(Record({{"other_stamp_", 14}}));

  auto merged_records = left_records.merge_sequential(
    right_records, "stamp", "sub_stamp", "value", "value",
    {"other_stamp", "stamp", "value", "other_stamp_"},
    how);

  print_records(*merged_records);
}
