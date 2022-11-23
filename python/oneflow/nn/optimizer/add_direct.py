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
import collections
from typing import Callable, Dict, Iterator, List, Union

import oneflow as flow
from oneflow.nn.parameter import Parameter

from .optimizer import Optimizer, ParamGroup

class ADD_DIRECT(Optimizer):
    def __init__(self, params, 
                    lr: float = 0.001):
        if  lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        
        options = dict()
        options["lr"] = lr
        super(ADD_DIRECT, self).__init__(params, options)
        for param_group in self.param_groups:
            for param in param_group.parameters:
                assert param.is_leaf, "parameters must be leaf tensor"
                self._state[param] = dict()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        with flow.no_grad():
            loss = None
            if closure is not None:
                loss = closure()

            for group in self.param_groups:
                for p in group.parameters:
                    if p.grad is None:
                        continue
                    d_p = p.grad
                    alpha=-group['lr']
                    p.add_(alpha*d_p)
            self._state["step"] = self._state["step"] + 1
            return loss
