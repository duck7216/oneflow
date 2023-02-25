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
#include <cstdint>
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/sqrt_square_sum_kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/user/ops/npu_command.h"

namespace oneflow {

namespace user_op {


template<typename T>
class SqrtSquareSumNpuKernel final : public user_op::OpKernel {
 public:
  SqrtSquareSumNpuKernel() = default;
  ~SqrtSquareSumNpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    int64_t len_of_wrap = sizeof(float);
    std::vector<int64_t> batch_desc = {1};
    AclTensorWrapper tmp_warp(tmp->mut_dptr<void>(),
                              ACL_FLOAT, 
                              batch_desc.size(), 
                              batch_desc.data(),
                              ACL_FORMAT_ND,
                              len_of_wrap);
    // Actually, Only Support: All axes and p==2
    NpuCommand cmd1;
    cmd1.OpName("LpNormReduce")
        .Input(x)
        .Output(tmp_warp)
        .Attr("p", (int64_t)2)
        // .Attr("axes", dim)
        .Attr("keepdim", false)
        .Attr("epsilon", static_cast<float>(1e-12))
        .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
        .Run();

    NpuCommand cmd2;
    cmd2.OpName("LpNormUpdate")
        .Input(tmp_warp)
        .Output(y)
        .Attr("p", (int64_t)2)
        .Attr("epsilon", static_cast<float>(1e-12))
        .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
        .Run();
    }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SQUARE_SUM_NPU_KERNEL(dtype)                                     \
  REGISTER_USER_KERNEL("sqrt_square_sum")                                             \
      .SetCreateFn<SqrtSquareSumNpuKernel<dtype>>()            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kNPU) \
                    && (user_op::HobDataType("y", 0) ==  GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                   \
        return 32*sizeof(float);                     \
      });

REGISTER_SQUARE_SUM_NPU_KERNEL(float);
}  // namespace user_op

}  // namespace oneflow
