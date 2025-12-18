"""Muon optimizer (single-process variant) ported from nanochat."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor


@torch.compile
def _zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """Orthogonalize update using quintic Newton–Schulz iteration."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Momentum + Newton–Schulz orthogonalization (single-process)."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        # group by parameter size (matches upstream nanochat grouping)
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

        for group in self.param_groups:
            group["initial_lr"] = group["lr"]

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]
                buf.lerp_(g, 1 - momentum)
                g = g.lerp_(buf, momentum) if nesterov else buf
                g = _zeropower_via_newtonschulz5(g, steps=ns_steps)
                scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5) if p.ndim >= 2 else 1.0
                p.add_(g, alpha=-lr * scale)
