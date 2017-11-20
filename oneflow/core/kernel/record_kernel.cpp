#include "oneflow/core/kernel/record_kernel.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

namespace {

template<typename T>
void RecordBlobImpl(PersistentOutStream& out_stream, const Blob* blob) {
  CHECK_EQ(GetDataType<T>::val, blob->data_type());
  const T* dptr = blob->dptr<T>();
  for (int64_t i = 0; i < blob->shape().At(0); ++i) {
    if (blob->has_data_id()) {
      for (size_t j = 0; j != JobDesc::Singleton()->SizeOfOneDataId(); ++j) {
        if (*(blob->data_id(i) + j) == '\0') { break; }
        out_stream.Write(blob->data_id(i) + j, 1);
      }
      out_stream.Write(" ", 1);
    }
    for (int64_t j = 0; j < blob->shape().Count(1); ++j) {
      out_stream << std::to_string(*dptr++) << ' ';
    }
    out_stream << '\n';
  }
}

void RecordBlob(PersistentOutStream& out_stream, const Blob* blob) {
  static const HashMap<int, void (*)(PersistentOutStream&, const Blob*)>
      record_funcs = {
#define RECORD_KERNEL_ENTRY(type_cpp, type_proto) \
  {type_proto, &RecordBlobImpl<type_cpp>},
          OF_PP_FOR_EACH_TUPLE(RECORD_KERNEL_ENTRY, ALL_DATA_TYPE_SEQ)};
  record_funcs.at(blob->data_type())(out_stream, blob);
}

}  // namespace

void RecordKernel::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int64_t parallel_id = reinterpret_cast<int64_t>(kernel_ctx.other);
  const std::string& root_path = op_conf().record_conf().record_path();
  OF_CALL_ONCE(root_path, GlobalFS()->MakeEmptyDir(root_path));
  for (const std::string& ibn : kernel_conf().input_bns()) {
    const std::string& lbn = Lbn4BnInOp(ibn);
    const Blob* blob = BnInOp2Blob(ibn);
    std::pair<std::string, std::string> parsed_lbn = ParseLbn(lbn);
    const std::string& op_name = parsed_lbn.first;
    const std::string& bn_in_op = parsed_lbn.second;
    std::string op_dir = JoinPath(root_path, op_name);
    OF_CALL_ONCE(op_dir, GlobalFS()->CreateDir(op_dir));
    std::string bn_in_op_dir = JoinPath(op_dir, bn_in_op);
    OF_CALL_ONCE(bn_in_op_dir, GlobalFS()->CreateDir(bn_in_op_dir));
    std::string file_path =
        JoinPath(bn_in_op_dir, "part-" + std::to_string(parallel_id));
    auto out_stream =
        RuntimeCtx::Singleton()->GetPersistentOutStream(file_path);
    RecordBlob(*out_stream, blob);
    out_stream->Flush();
  }
}

}  // namespace oneflow
