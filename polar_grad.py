# Copyright 2025 Tim Tsz-Kit Lau.
# Adopted from Muon by Keller Jordan at https://github.com/KellerJordan/Muon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import torch
from polar import polar

class PolarGrad(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0., momentum=0.95, polar_first=False, method='qdwh', inner_steps=2, a=3.4445, b=-4.7750, c=2.031):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, polar_first=polar_first, method=method, inner_steps=inner_steps, a=a, b=b, c=c)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(g)
                m = state['momentum']
                if group['polar_first']:
                    U = polar(g, method=group['method'], max_iterations=group["inner_steps"], ns_coeffs=(group['a'], group['b'], group['c']))[0]
                    nuc_norm = torch.sum(g.type_as(U) * U)
                    m.lerp_(U, 1 - group["momentum"])
                    g = nuc_norm * m
                else:
                    m.lerp_(g, 1 - group["momentum"])
                    U = polar(m, method=group['method'], max_iterations=group["inner_steps"], ns_coeffs=(group['a'], group['b'], group['c']))[0]
                    nuc_norm = torch.sum(m.type_as(U) * U)
                    g = nuc_norm * U
                p.data.mul_(1 - group['lr'] * group['weight_decay']).add_(g, alpha=-group['lr'])
        return loss
