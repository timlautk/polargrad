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

import os
import torch
import torch.distributed as dist
from torch import Tensor

from polar import polar


class PolarGrad(torch.optim.Optimizer):
    """
    PolarGrad - Polar Gradient Method (with nuclear norm scaling)

    Arguments:
        lr: The learning rate used by PolarGrad.
        weight_decay: The weight decay used by PolarGrad.
        momentum: The momentum used by PolarGrad.
        nesterov: Whether to use Nesterov-style momentum in PolarGrad. (recommended)
        method: The method used for polar decomposition. (default: 'qdwh'; 'zolo-pd' and 'ns' are also available)
        a, b, c: The coefficients used for the Newton-Schulz iteration.
        inner_steps: The number of the QDWH algorithm or Newton-Schulz iteration steps to use.
        rank: The rank of the current process in the distributed group.
        world_size: The total number of processes in the distributed group.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, polar_first=False, method='qdwh', inner_steps=5, a=3.4445, b=-4.7750, c=2.0315, rank=None, world_size=None):
        if (rank is None) or (world_size is None):
            raise Exception("world_size and rank params required, if you want to use this optimizer on a single GPU, pass rank=0 and world_size=1.")
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, polar_first=polar_first, method=method, inner_steps=inner_steps, a=a, b=b, c=c)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                        update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world), alpha=-group['lr'])
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    if group['polar_first']:
                        if g.dtype != torch.bfloat16:
                            buf = buf.bfloat16()
                        if g.ndim == 4: # for the case of conv filters
                            g = g.view(len(g), -1)
                        if g.ndim == 3:
                            g_list = []
                            for i in range(g.size(0)):
                                g_mat = g[i]
                                buf_mat = buf[i]
                                if g_mat.dtype == torch.bfloat16 and group['method'] == 'qdwh':
                                    g_mat = g_mat.float()
                                U_mat = polar(g_mat, method=group['method'], max_iterations=group["inner_steps"], ns_coeffs=(group['a'], group['b'], group['c']))[0].bfloat16()
                                nuc_norm = torch.sum(g_mat.type_as(U_mat) * U_mat)
                                buf_mat.lerp_(U_mat, 1 - group["momentum"])
                                g_mat = nuc_norm * buf_mat
                                g_list.append(g_mat)
                            g = torch.stack(g_list, dim=0)
                        if g.ndim == 2:
                            if g.dtype == torch.bfloat16 and group['method'] == 'qdwh':
                                g = g.float()
                            U = polar(g, method=group['method'], max_iterations=group["inner_steps"], ns_coeffs=(group['a'], group['b'], group['c']))[0].bfloat16()
                            nuc_norm = torch.sum(g.type_as(U) * U)
                            buf.lerp_(U, 1 - group["momentum"])
                            g = nuc_norm * buf
                    else:
                        g = buf.lerp_(g, 1 - group["momentum"])
                        if g.ndim == 4: # for the case of conv filters
                            g = g.view(len(g), -1)
                        if g.ndim == 3:
                            g_list = []
                            for i in range(g.size(0)):
                                g_mat = g[i]
                                if g_mat.dtype == torch.bfloat16 and group['method'] == 'qdwh':
                                    g_mat = g_mat.float()
                                U_mat = polar(g_mat, method=group['method'], max_iterations=group["inner_steps"], ns_coeffs=(group['a'], group['b'], group['c']))[0].bfloat16()
                                nuc_norm = torch.sum(g_mat.type_as(U_mat) * U_mat)
                                g_mat = nuc_norm * U_mat
                                g_list.append(g_mat)
                            g = torch.stack(g_list, dim=0)
                        if g.ndim == 2:
                            if g.dtype == torch.bfloat16 and group['method'] == 'qdwh':
                                g = g.float()
                            U = polar(g, method=group['method'], max_iterations=group["inner_steps"], ns_coeffs=(group['a'], group['b'], group['c']))[0].bfloat16()
                            nuc_norm = torch.sum(g.type_as(U) * U)
                            g = nuc_norm * U
                    g = g.flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, method='qdwh', a=3.4445, b=-4.7750, c=2.0315, inner_steps=5, rank=None, world_size=None):
        if (rank is None) or (world_size is None):
            raise Exception("world_size and rank params required, if you want to use this optimizer on a single GPU, pass rank=0 and world_size=1.")
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, method=method, inner_steps=inner_steps, a=a, b=b, c=c)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                        update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    if g.ndim == 4: # for the case of conv filters
                        g = g.view(len(g), -1)
                    if g.ndim == 3:
                        g_list = []
                        for i in range(g.size(0)):
                            g_mat = g[i]
                            g_mat = polar(g_mat, method=group['method'], max_iterations=group["inner_steps"], ns_coeffs=(group['a'], group['b'], group['c']))[0].bfloat16()
                            g_list.append(g_mat)
                        g = torch.stack(g_list, dim=0).flatten()
                    if g.ndim == 2:
                        g = polar(g, method=group['method'], max_iterations=group["inner_steps"], ns_coeffs=(group['a'], group['b'], group['c']))[0].bfloat16().flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()
