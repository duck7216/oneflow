#ifndef ONEFLOW_CORE_OPERATOR_ACC_OP_H_
#define ONEFLOW_CORE_OPERATOR_ACC_OP_H_

#include "oneflow/core/operator/accumulate_op.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class AccOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccOp);
  AccOp() = default;
  ~AccOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  LogicalNode* NewProperLogicalNode() const override { return new AccLogicalNode; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void InferOutputBlobTimeShape(std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
                                const ParallelContext* parallel_ctx,
                                Shape* time_shape) const override;

 private:
  void InferHasBatchDim(std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ACC_OP_H_
