/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_API_PYTHON_CALIBRATION_CALIBRATION_ENV_HELPER_H_
#define ONEFLOW_API_PYTHON_CALIBRATION_CALIBRATION_ENV_HELPER_H_

#include "oneflow/core/common/maybe.h"

#ifdef WITH_TENSORRT
#include "oneflow/xrt/api.h"
#endif  // WITH_TENSORRT

namespace oneflow {

Maybe<void> CacheInt8Calibration() {
#ifdef WITH_TENSORRT
  xrt::tensorrt::CacheInt8Calibration();
#else
  CHECK_OR_RETURN(0) << "Please recompile with TensorRT.";
#endif  // WITH_TENSORRT
  return Maybe<void>::Ok();
}

Maybe<void> WriteInt8Calibration(const std::string& path) {
#ifdef WITH_TENSORRT
  xrt::tensorrt::CacheInt8Calibration();
  xrt::tensorrt::WriteInt8Calibration(path);
#else
  CHECK_OR_RETURN(0) << "Please recompile with TensorRT.";
#endif  // WITH_TENSORRT
  return Maybe<void>::Ok();
}

}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_CALIBRATION_CALIBRATION_ENV_HELPER_H_
