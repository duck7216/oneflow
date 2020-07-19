from __future__ import absolute_import

import os

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("constant")
def constant(value, dtype=None, shape=None, name=None):
    if name is None:
        name = id_util.UniqueStr("Constant_")
    assert value is not None
    assert dtype is not None

    if not isinstance(value, (int, float)):
        raise NotImplementedError

    if isinstance(value, float):
        is_floating_value = True
    else:
        is_floating_value = False
    if shape is not None:
        assert isinstance(shape, (list, tuple))
    else:
        shape = []
    return (
        flow.user_op_builder(name)
        .Op("constant")
        .Output("out")
        .Attr("floating_value", float(value))
        .Attr("integer_value", int(value))
        .Attr("is_floating_value", is_floating_value)
        .Attr("dtype", dtype)
        .Attr("shape", shape)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("constant_scalar")
def constant_scalar(value, dtype=None, name=None):
    return flow.constant(value, dtype=dtype, shape=[1])


@oneflow_export("constant_like")
def constant_like(like, value, dtype=None, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ConstantLike_"),
    )
    setattr(op_conf.constant_like_conf, "like", like.unique_name)
    if isinstance(value, int):
        op_conf.constant_like_conf.int_operand = value
    elif isinstance(value, float):
        op_conf.constant_like_conf.float_operand = value
    else:
        raise NotImplementedError
    if dtype is not None:
        setattr(op_conf.constant_like_conf, "data_type", dtype)
    setattr(op_conf.constant_like_conf, "out", "out")
    interpret_util.Forward(op_conf)
    out_lbi = logical_blob_id_util.LogicalBlobId()
    setattr(out_lbi, "op_name", op_conf.name)
    setattr(out_lbi, "blob_name", "out")
    return remote_blob_util.RemoteBlob(out_lbi)


@oneflow_export("ones_like")
def ones_like(like, dtype=None, name=None):
    return constant_like(like, 1, dtype=dtype, name=name)


@oneflow_export("zeros_like")
def zeros_like(like, dtype=None, name=None):
    return constant_like(like, 0, dtype=dtype, name=name)
