#include "oneflow/core/operator/all_reduce_facade_op.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void AllReduceFacadeOp::InitFromOpConf() {
  CHECK(op_conf().has_all_reduce_facade_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& AllReduceFacadeOp::GetCustomizedConf() const {
  return op_conf().all_reduce_facade_conf();
}

void AllReduceFacadeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

LogicalNode* AllReduceFacadeOp::NewProperLogicalNode() const {
  return new AllReduceFacadeLogicalNode();
}

void AllReduceFacadeOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  for (const auto& ibn : input_bns()) {
    CHECK(SbpInferHint4Ibn(ibn).sbp_parallel().has_partial_sum_parallel());
  }
  SbpSignatureBuilder().PartialSum(input_bns()).Broadcast(output_bns()).Build(sbp_signature);
}

REGISTER_OP(OperatorConf::kAllReduceFacadeConf, AllReduceFacadeOp);

}  // namespace oneflow
