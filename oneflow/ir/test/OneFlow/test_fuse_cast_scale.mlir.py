"""
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
"""
import unittest
import numpy as np
import oneflow as flow
import oneflow.typing as oft

func_config = flow.FunctionConfig()


@flow.unittest.skip_unless_1n1d()
class TestMLIROptimizations(flow.unittest.TestCase):
    def test_fuse_cast_scale_mlir(test_case):
        @flow.global_function(function_config=func_config)
        def FuseCastScaleJob(x: oft.Numpy.Placeholder((96, 96))) -> oft.Numpy:
            scale = flow.get_variable(
                "scale",
                shape=(1,),
                dtype=flow.float16,
                initializer=flow.random_uniform_initializer(),
                trainable=False,
            )
            loss = flow.cast(x, dtype=flow.float16) * scale
            return loss

        x = np.random.rand(96, 96).astype(np.float32) - 1
        FuseCastScaleJob(x)


if __name__ == "__main__":
    unittest.main()
