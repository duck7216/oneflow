#ifndef ONEFLOW_CORE_OPERATOR_REDUCE_IDENTITY_OP_H_
#define ONEFLOW_CORE_OPERATOR_REDUCE_IDENTITY_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class ReduceIdentityOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceIdentityOp);
  ReduceIdentityOp() = default;
  ~ReduceIdentityOp() = default;

  LogicalNode* NewProperLogicalNode() const override { return new ReduceIdentityLogicalNode; }
  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().reduce_identity_conf(); }
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }

 private:
  void InferSbpSignature(SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
                         const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
                         std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
                         const ParallelDesc& parallel_desc) const override;

  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_IDENTITY_OP_H_
