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

#include <vector>
#include <unordered_map>
#include <string>
#include <map>
#include <tuple>
#include <memory>

#include "pybind11/iostream.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#include "caret_analyze_cpp_impl/records.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;

PYBIND11_MODULE(record_cpp_impl, m) {
  py::class_<Record>(m, "RecordBase")
  .def(py::init())
  .def(
    py::init(
      [](const Record & init) {
        return new Record(init);
      })
  )
  .def(
    py::init(
      [](std::unordered_map<std::string, uint64_t> init) {
        return new Record(init);
      })
  )
  .def(
    "change_dict_key", &Record::change_dict_key,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "equals", &Record::equals,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "merge", &Record::merge,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "add", &Record::add,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "drop_columns", &Record::drop_columns,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "get", &Record::get,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "get_with_default", &Record::get_with_default,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def_property_readonly(
    "data", &Record::get_data,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def_property_readonly(
    "columns", &Record::get_columns,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

  py::class_<RecordsBase>(m, "RecordsBase")
  .def(py::init())
  .def(
    py::init(
      [](const RecordsVectorImpl & init) {
        return new RecordsVectorImpl(init);
      })
  )
  .def(
    py::init(
      [](std::vector<Record> init, std::vector<std::string> columns) {
        return new RecordsVectorImpl(init, columns);
      })
  )
  .def(
    "append", &RecordsBase::append,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "append_column", &RecordsBase::append_column,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "clone", &RecordsBase::clone,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "equals", &RecordsBase::equals,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "drop_columns", &RecordsBase::drop_columns,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "rename_columns", &RecordsBase::rename_columns,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "filter_if", &RecordsBase::filter_if,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "reindex", &RecordsBase::reindex,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "concat", &RecordsBase::concat,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "sort", &RecordsBase::sort,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "sort_column_order", &RecordsBase::sort_column_order,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "merge", &RecordsBase::merge,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "merge_sequential", &RecordsBase::merge_sequential,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "bind_drop_as_delay", &RecordsBase::bind_drop_as_delay,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "merge_sequential_for_addr_track",
    &RecordsBase::merge_sequential_for_addr_track,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "groupby",
    static_cast<std::map<std::tuple<uint64_t>,
    std::unique_ptr<RecordsBase>>(RecordsBase::*)(std::string)>(&RecordsBase::groupby),
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "groupby",
    static_cast<std::map<std::tuple<uint64_t, uint64_t>,
    std::unique_ptr<RecordsBase>>(RecordsBase::*)(std::string, std::string)>(&RecordsBase::groupby),
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def(
    "groupby",
    static_cast<
      std::map<std::tuple<uint64_t, uint64_t, uint64_t>,
      std::unique_ptr<RecordsBase>>(RecordsBase::*)(
        std::string, std::string,
        std::string)>(&RecordsBase::groupby),
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def_property_readonly(
    "data", &RecordsBase::get_data,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
  .def_property_readonly(
    "columns", &RecordsBase::get_columns,
    py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
