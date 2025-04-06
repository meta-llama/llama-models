# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from typing import Any, Dict, List

from flax import nnx


class FeedForward(nnx.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        do_reduce: bool = True,
    ):
        super().__init__()
        self.do_reduce = do_reduce

        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        if prefix + "mlp.fc1_weight" in state_dict:
            w1, w3 = state_dict.pop(prefix + "mlp.fc1_weight").chunk(2, dim=0)
            state_dict[prefix + "w1.weight"] = w1
            state_dict[prefix + "w3.weight"] = w3
            state_dict[prefix + "w2.weight"] = state_dict.pop(prefix + "mlp.fc2_weight")

    def __call__(self, x):
        x = F.silu(F.linear(x, self.w1.weight)) * F.linear(x, self.w3.weight)
        out = F.linear(x, self.w2.weight)
        if self.do_reduce:
            return reduce_from_model_parallel_region(out)
        return out
