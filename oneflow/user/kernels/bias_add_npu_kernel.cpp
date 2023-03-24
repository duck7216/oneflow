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
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {

namespace {

class BiasAddNpuKernel final : public user_op::OpKernel {
 public:
  BiasAddNpuKernel() = default;
  ~BiasAddNpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* a_tensor = ctx->Tensor4ArgNameAndIndex("a", 0);
    user_op::Tensor* b_tensor = ctx->Tensor4ArgNameAndIndex("b", 0);
    if (a_tensor->shape_view().elem_cnt() == 0 || b_tensor->shape_view().elem_cnt() == 0) {
      return;
    }
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    // DimVector bias_shape_dim_v;
    // b_tensor->shape_view().ToDimVector(&bias_shape_dim_v);
    // std::vector<int64_t> bias_shape_vector;
    // for(auto beta_shape: bias_shape_dim_v){
    //   std::cout<<beta_shape<<" "<<std::endl;
    //   //bias_shape_vector.push_back(beta_shape);
    // }
    // std::cout<<"over "<<std::endl;
    std::string format = "channels_first";
    NpuCommand npu_command;
    npu_command.OpName("BiasAdd")
               .Input(a_tensor)
               .Input(b_tensor)
               .Output(out_tensor)
               .Attr("data_format", format)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run()
               .Realease();
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("bias_add")
    .SetCreateFn<BiasAddNpuKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU)
    .SetInplaceProposalFn([](const user_op::InferContext& ctx,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "a", 0, true));
      return Maybe<void>::Ok();
    });
}  // namespace

}  // namespace oneflow
