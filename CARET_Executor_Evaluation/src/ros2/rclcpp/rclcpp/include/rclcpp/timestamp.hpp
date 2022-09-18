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

#ifndef RCLCPP__TIMESTAMP_HPP_
#define RCLCPP__TIMESTAMP_HPP_

#include <type_traits>
#include <utility>

namespace rclcpp
{
// Sec ans Nanosec check
template <typename M, typename = void>
struct HasSec : public std::false_type
{
};

template <typename M>
struct HasSec<M, decltype((void)M::sec)> : std::true_type
{
};

template <typename M, typename = void>
struct HasNanoSec : public std::false_type
{
};

template <typename M>
struct HasNanoSec<M, decltype((void)M::nanosec)> : std::true_type
{
};

template <typename M, typename Enable = void>
struct TimeStampHeaderStamp
{
  static std::pair<bool, int64_t> value(const M &) { return std::make_pair(false, 0); }
};

template <typename M>
struct TimeStampHeaderStamp<
  M, typename std::enable_if<HasSec<M>::value && HasNanoSec<M>::value>::type>
{
  static std::pair<bool, int64_t> value(const M & m)
  {
    const auto nanos = RCL_S_TO_NS(static_cast<int64_t>(m.sec)) + m.nanosec;
    return std::make_pair(true, nanos);
  }
};

// Stamp check
template <typename M, typename = void>
struct HasStamp : public std::false_type
{
};

template <typename M>
struct HasStamp<M, decltype((void)M::stamp)> : std::true_type
{
};

template <typename M, typename Enable = void>
struct TimeStampHeader
{
  static std::pair<bool, int64_t> value(const M &) { return std::make_pair(false, 0); }
};

template <typename M>
struct TimeStampHeader<M, typename std::enable_if<HasStamp<M>::value>::type>
{
  static std::pair<bool, int64_t> value(const M & m)
  {
    return TimeStampHeaderStamp<decltype(m.stamp)>::value(m.stamp);
  }
};

// Header check
template <typename M, typename = void>
struct HasHeader : public std::false_type
{
};

template <typename M>
struct HasHeader<M, decltype((void)M::header)> : std::true_type
{
};

template <typename M, typename Enable = void>
struct TimeStamp
{
  static std::pair<bool, int64_t> value(const M &) { return std::make_pair(false, 0); }
};

template <typename M>
struct TimeStamp<M, typename std::enable_if<HasHeader<M>::value>::type>
{
  static std::pair<bool, int64_t> value(const M & m)
  {
    return TimeStampHeader<decltype(m.header)>::value(m.header);
  }
};
}  // namespace rclcpp

#endif  // RCLCPP__TIMESTAMP_HPP_
