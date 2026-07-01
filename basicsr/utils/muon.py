"""
Muon optimizer — MomentUm Orthogonalized by Newton-schulz.

Matches the torch.optim.Muon interface from PyTorch 2.12:
  lr, weight_decay, momentum, nesterov, ns_steps, eps

Since torch.optim.Muon only covers 2D parameters, this wrapper
adds an internal AdamW group for 1D params (biases, LayerNorm weights)
so BasicSR's single-optimizer training loop works unchanged.

YAML usage (same keys as torch.optim.Muon):
  optim_g:
    type: Muon
    lr: !!float 5e-4
    weight_decay: 0
    momentum: 0.95
    nesterov: true
    ns_steps: 5
"""

import torch
from torch.optim import Optimizer


@torch.no_grad()
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon with AdamW fallback for 1-D parameters.

    Interface matches torch.optim.Muon (PyTorch 2.12):
        lr, weight_decay, momentum, nesterov, ns_steps, eps

    The betas parameter (default (0.9, 0.99)) is used only for the
    internal AdamW group that handles biases and LayerNorm weights.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
        betas: tuple = (0.9, 0.99),
    ):
        raw = list(params)
        if raw and isinstance(raw[0], dict):
            all_params = [p for g in raw for p in g['params']]
        else:
            all_params = raw

        muon_params  = [p for p in all_params if p.requires_grad and p.ndim >= 2]
        adamw_params = [p for p in all_params if p.requires_grad and p.ndim <  2]

        groups = []
        if muon_params:
            groups.append({
                'params':       muon_params,
                '_mode':        'muon',
                'lr':           lr,
                'momentum':     momentum,
                'nesterov':     nesterov,
                'ns_steps':     ns_steps,
                'eps':          eps,
                'weight_decay': weight_decay,
            })
        if adamw_params:
            groups.append({
                'params':       adamw_params,
                '_mode':        'adamw',
                'lr':           lr,
                'betas':        tuple(betas),
                'weight_decay': weight_decay,
                'eps':          1e-8,
            })

        if not groups:
            raise ValueError('Muon received no trainable parameters.')

        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps, eps=eps,
                        betas=tuple(betas), _mode='muon')
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group['_mode'] == 'muon':
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group):
        lr       = group['lr']
        momentum = group['momentum']
        nesterov = group['nesterov']
        ns_steps = group['ns_steps']
        eps      = group['eps']
        wd       = group['weight_decay']

        for p in group['params']:
            if p.grad is None:
                continue

            g = p.grad
            state = self.state[p]

            if 'buf' not in state:
                state['buf'] = torch.clone(g).detach()
            else:
                state['buf'].mul_(momentum).add_(g)

            update = g.add(state['buf'], alpha=momentum) if nesterov else state['buf'].clone()

            orig_shape = update.shape
            update_2d  = update.reshape(orig_shape[0], -1)
            update_2d  = _zeropower_via_newtonschulz5(update_2d, steps=ns_steps, eps=eps)
            update     = update_2d.reshape(orig_shape)

            scale = g.norm() / (update.norm() + 1e-8)
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.add_(update, alpha=-lr * scale)

    def _adamw_step(self, group):
        lr  = group['lr']
        b1, b2 = group['betas']
        wd  = group['weight_decay']
        eps = group['eps']

        for p in group['params']:
            if p.grad is None:
                continue

            g = p.grad
            state = self.state[p]

            if 'step' not in state:
                state['step'] = 0
                state['exp_avg']    = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

            state['step'] += 1
            t = state['step']
            m = state['exp_avg']
            v = state['exp_avg_sq']

            m.mul_(b1).add_(g, alpha=1.0 - b1)
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

            m_hat = m / (1.0 - b1 ** t)
            v_hat = v / (1.0 - b2 ** t)

            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
