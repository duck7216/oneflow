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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/user/ops/comm_net_device_infer_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> EagerNcclAllReduceOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EagerNcclAllReduceOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EagerNcclAllReduceOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().PartialSum(user_op::OpArg("in", 0)).Broadcast(user_op::OpArg("out", 0)).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerNcclAllReduceOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Device>> EagerNcclAllReduceOp::InferDevice(
    user_op::DeviceInferContext* ctx) {
  return DeviceInferFn<&IsAsyncLaunched>(ctx);
}

/* static */ Maybe<void> EagerNcclBroadcastOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EagerNcclBroadcastOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EagerNcclBroadcastOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().PartialSum(user_op::OpArg("in", 0)).Broadcast(user_op::OpArg("out", 0)).Build();
  ctx->NewBuilder().Broadcast(user_op::OpArg("in", 0)).Broadcast(user_op::OpArg("out", 0)).Build();
  ctx->NewBuilder().Split(user_op::OpArg("in", 0), 0).Broadcast(user_op::OpArg("out", 0)).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerNcclBroadcastOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Device>> EagerNcclBroadcastOp::InferDevice(
    user_op::DeviceInferContext* ctx) {
  return DeviceInferFn<&IsAsyncLaunched>(ctx);
}

/* static */ Maybe<void> EagerNcclTouchOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EagerNcclTouchOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerNcclTouchOp::GetSbp(user_op::SbpContext* ctx) {
  // local only
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerNcclTouchOp::InferDataType(user_op::InferContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Device>> EagerNcclTouchOp::InferDevice(user_op::DeviceInferContext* ctx) {
  return DeviceInferFn<&IsAsyncLaunched>(ctx);
}

/* static */ Maybe<void> EagerNcclReduceOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EagerNcclReduceOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EagerNcclReduceOp::GetSbp(user_op::SbpContext* ctx) {
  UNIMPLEMENTED_THEN_RETURN() << "consistent tensor are not supported";
}

/* static */ Maybe<void> EagerNcclReduceOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Device>> EagerNcclReduceOp::InferDevice(
    user_op::DeviceInferContext* ctx) {
  return DeviceInferFn<&SyncLaunched>(ctx);
}

/* static */ Maybe<void> EagerNcclReduceScatterOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerNcclReduceScatterOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  Shape* out_shape = ctx->OutputShape("out", 0);
  const Shape& shape = ctx->InputShape("in", 0);
  DimVector dim_vec;
  if (shape.NumAxes() > 0) {
    dim_vec.insert(dim_vec.end(), shape.dim_vec().cbegin(), shape.dim_vec().cend());
  }
  const SbpParallel& out_sbp_para = ctx->SbpParallel4ArgNameAndIndex("out", 0);
  const int64_t& parallel_num = ctx->parallel_ctx().parallel_num();
  if (parallel_num > 1) {
    const int64_t& split_axis = out_sbp_para.split_parallel().axis();
    CHECK_LT_OR_RETURN(split_axis, dim_vec.size());
    BalancedSplitter bs(shape.At(split_axis), parallel_num);
    dim_vec[split_axis] = bs.At(ctx->parallel_ctx().parallel_id()).size();
  }
  *out_shape = Shape(dim_vec);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerNcclReduceScatterOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> EagerNcclReduceScatterOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  const NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
  NdSbp* in_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* out_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
  for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
    CHECK_OR_RETURN(sbp_hint.has_partial_sum_parallel() || sbp_hint.has_broadcast_parallel());
  }
  in_nd_sbp->clear_sbp_parallel();
  out_nd_sbp->clear_sbp_parallel();

  // P2S or B2S
  const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
  CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
  in_nd_sbp->CopyFrom(in_dis_hint);
  for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
    out_nd_sbp->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerNcclReduceScatterOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Device>> EagerNcclReduceScatterOp::InferDevice(
    user_op::DeviceInferContext* ctx) {
  return DeviceInferFn<&SyncLaunched>(ctx);
}

/* static */ Maybe<void> EagerNcclAllGatherOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
  *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EagerNcclAllGatherOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EagerNcclAllGatherOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> EagerNcclAllGatherOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  const NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
  NdSbp* in_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* out_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
  for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
    CHECK_OR_RETURN(sbp_hint.has_split_parallel());
    CHECK_EQ_OR_RETURN(sbp_hint.split_parallel().axis(), 0);
  }

  in_nd_sbp->clear_sbp_parallel();
  out_nd_sbp->clear_sbp_parallel();

  // S(0)->B
  const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
  CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
  for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
    in_nd_sbp->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
    out_nd_sbp->add_sbp_parallel()->mutable_broadcast_parallel();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerNcclAllGatherOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Device>> EagerNcclAllGatherOp::InferDevice(
    user_op::DeviceInferContext* ctx) {
  return DeviceInferFn<&SyncLaunched>(ctx);
}

/* static */ Maybe<void> EagerNcclS2sOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
  *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerNcclS2sOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> EagerNcclS2sOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  const int64_t in_split_axis = ctx->user_op_conf().attr<int64_t>("in_split_axis");
  const int64_t out_split_axis = ctx->user_op_conf().attr<int64_t>("out_split_axis");
  const NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
  NdSbp* in_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
  NdSbp* out_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
  CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
  for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
    CHECK_OR_RETURN(sbp_hint.has_split_parallel());
    CHECK_EQ_OR_RETURN(sbp_hint.split_parallel().axis(), in_split_axis);
  }

  in_nd_sbp->clear_sbp_parallel();
  out_nd_sbp->clear_sbp_parallel();

  // S(in)->S(out)
  const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
  CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
  for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
    in_nd_sbp->add_sbp_parallel()->mutable_split_parallel()->set_axis(in_split_axis);
    out_nd_sbp->add_sbp_parallel()->mutable_split_parallel()->set_axis(out_split_axis);
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> EagerNcclS2sOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Device>> EagerNcclS2sOp::InferDevice(user_op::DeviceInferContext* ctx) {
  return DeviceInferFn<&SyncLaunched>(ctx);
}

}  // namespace oneflow
