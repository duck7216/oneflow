#include "oneflow/user/ops/npu_command.h"
#include "oneflow/core/device/npu_util.h"
namespace oneflow{
    GlobalStream::GlobalStream(){
        OF_NPU_CHECK(aclrtGetCurrentContext(&context_));
        OF_NPU_CHECK(aclrtCreateStream(&stream_));
    }
    void GlobalStream::Free(){
        std::once_flag flag;
        std::call_once(flag, [&](){
            OF_NPU_CHECK(aclrtSetCurrentContext(context_));
            OF_NPU_CHECK(aclrtSynchronizeStream(&stream_));
            OF_NPU_CHECK(aclrtDestroyStream(&stream_));
        });
    }
    void GlobalStream::Sync(){
        OF_NPU_CHECK(aclrtSynchronizeStream(&stream_));
    }
} // namespace oneflow
