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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/ep/include/primitive/batch_matmul.h"
#include "oneflow/user/ops/npu_command.h"
namespace oneflow {

namespace {

class MatmulNpuKernel final : public user_op::OpKernel {
 public:
  MatmulNpuKernel() = default;
  ~MatmulNpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // std::cout<<"MatMul Npu"<<std::endl;
    bool trans_a = ctx->Attr<bool>("transpose_a");
    bool trans_b = ctx->Attr<bool>("transpose_b");
    user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    CHECK_EQ(a->shape_view().NumAxes(), 2);
    const DataType data_type = a->data_type();
    user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    CHECK_EQ(b->shape_view().NumAxes(), 2);
    CHECK_EQ(b->data_type(), data_type);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(out->shape_view().NumAxes(), 2);
    CHECK_EQ(out->data_type(), data_type);
    int64_t offset_x = 0;
    NpuCommand npu_command;
    npu_command.OpName("MatMulV2")
               .Input(a, "channels_nd")
               .Input(b, "channels_nd")
               .Output(out, "channels_nd")
               .Attr("transpose_x1", trans_a)
               .Attr("transpose_x2", trans_b)
               .Attr("offset_x", offset_x)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run()
               .Realease();
    // OF_NPU_CHECK(aclrtSynchronizeStream(ctx->stream()->As<ep::NpuStream>()->npu_stream()));   
    // PrintResult(out);
    // std::cout<<"Matmul Execute Over"<<std::endl; 
  }
};

REGISTER_USER_KERNEL("matmul")
    .SetCreateFn<MatmulNpuKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU);
    // dck_caution_here
    // .SetInplaceProposalFn([](const user_op::InferContext& ctx,
    //                          const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
    //   if (ctx.has_input("_add_to_output", 0)) {
    //     OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "_add_to_output", 0, true));
    //   }
    //   return Maybe<void>::Ok();
    // });    

class BatchMatmulNpuKernel final : public user_op::OpKernel {
 public:
  BatchMatmulNpuKernel() = default;
  ~BatchMatmulNpuKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    bool trans_a = ctx->Attr<bool>("transpose_a");
    bool trans_b = ctx->Attr<bool>("transpose_b");
    user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const DataType data_type = a->data_type();
    user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    CHECK_EQ(b->data_type(), data_type);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(out->data_type(), data_type);
    int64_t offset_x = 0;
    if (ctx->has_input("_add_to_output", 0)) {
      UNIMPLEMENTED();
    }
    // if (trans_a || trans_b) {
    //   std::cout<<a->shape_view().ToString()<<"|"<<b->shape_view().ToString()<<std::endl;
    //   UNIMPLEMENTED();
    // }
    DimVector a_shape_dim_v;
    a->shape_view().ToDimVector(&a_shape_dim_v);
    size_t a_len = a_shape_dim_v.size();
    int64_t a_first = 1;
    for(int i=0; i<a_len-2; ++i){
      a_first *= a_shape_dim_v.at(i);
    }
    std::vector<int64_t> a_shape_vector = {a_first, a_shape_dim_v.at(a_len-2), a_shape_dim_v.at(a_len-1)};

    DimVector b_shape_dim_v;
    b->shape_view().ToDimVector(&b_shape_dim_v);
    size_t b_len = b_shape_dim_v.size();
    int64_t b_first = 1;
    for(int i=0; i<b_len-2; ++i){
      b_first *= b_shape_dim_v.at(i);
    }
    std::vector<int64_t> b_shape_vector = {b_first, b_shape_dim_v.at(b_len-2), b_shape_dim_v.at(b_len-1)};

    DimVector out_shape_dim_v;
    out->shape_view().ToDimVector(&out_shape_dim_v);
    size_t out_len = out_shape_dim_v.size();
    int64_t out_first = 1;
    for(int i=0; i<out_len-2; ++i){
      out_first *= out_shape_dim_v.at(i);
    }
    std::vector<int64_t> out_shape_vector = {out_first, out_shape_dim_v.at(out_len-2), out_shape_dim_v.at(out_len-1)};

    NpuCommand npu_command;
    npu_command.OpName("BatchMatMul")
               .InputWithShape(a, a_shape_vector)
               .InputWithShape(b, b_shape_vector)
               .OutputWithShape(out, out_shape_vector)
               .Attr("adj_x1", trans_a)
               .Attr("adj_x2", trans_b)
               .Stream(ctx->stream()->As<ep::NpuStream>()->npu_stream())
               .Check();
    npu_command.Run()
               .Realease();
  }
};

REGISTER_USER_KERNEL("batch_matmul")
    .SetCreateFn<BatchMatmulNpuKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kNPU)
    .SetInplaceProposalFn([](const user_op::InferContext& ctx,
                             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {
      if (ctx.has_input("_add_to_output", 0)) {
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "_add_to_output", 0, true));
      }
      return Maybe<void>::Ok();
    });


} // namespace {anonymous}
} // namespace oneflow