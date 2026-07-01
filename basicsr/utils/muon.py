"""
Muon optimizer — MomentUm Orthogonalized by Newton-schulz.

Reference: https://github.com/KellerJordan/modded-nanogpt

Design for BasicSR integration
───────────────────────────────
A single torch.optim.Optimizer with two internal param groups:
  Group 0 (Muon)  : ndim >= 2 — weight matrices, conv kernels
  Group 1 (AdamW) : ndim <  2 — biases, LayerNorm weights/biases

This lets BasicSR's MultiStepLR scheduler update both groups' lr
proportionally (both multiplied by gamma at each milestone), and keeps
the single-optimizer assumption throughout the training loop.

YAML usage
──────────
  optim_g:
    type: Muon
    lr: 0.01               # Muon lr for 2D+ params
    momentum: 0.95
    nesterov: true
    ns_steps: 5
    adamw_lr: !!float 3e-4 # AdamW lr for 1D params
    adamw_betas: [0.9, 0.99]
    adamw_wd: 0.0
"""

import torch
from torch.optim import Optimizer


@torch.no_grad()
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration: maps G → nearest matrix with unit spectral norm.
    Runs in bfloat16 for numerical stability on A100/H100.
    Input must be 2-D (reshape before calling).
    """
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    # Work in the smaller-dimension orientation for efficiency
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
    Muon optimizer with AdamW fallback for 1-D parameters.

    Args:
        params      : model parameters (list of tensors or param-groups).
        lr          : learning rate for Muon (2D+ weight matrices).
        momentum    : SGD momentum coefficient for Muon.
        nesterov    : use Nesterov-style momentum (recommended).
        ns_steps    : Newton-Schulz iteration count (5 is default; 3 is faster).
        adamw_lr    : learning rate for AdamW (1D params: biases, LN weights).
        adamw_betas : (beta1, beta2) for AdamW.
        adamw_wd    : weight decay for AdamW.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple = (0.9, 0.99),
        adamw_wd: float = 0.0,
        betas: tuple = None,
        weight_decay: float = None,
    ):
        if betas is not None:
            adamw_betas = tuple(betas)
        if weight_decay is not None:
            adamw_wd = weight_decay
        # Flatten to a plain list of tensors (handle both tensor lists and param-group dicts)
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
                'params':    muon_params,
                '_mode':     'muon',
                'lr':        lr,
                'momentum':  momentum,
                'nesterov':  nesterov,
                'ns_steps':  ns_steps,
            })
        if adamw_params:
            groups.append({
                'params':      adamw_params,
                '_mode':       'adamw',
                'lr':          adamw_lr,
                'betas':       tuple(adamw_betas),
                'weight_decay': adamw_wd,
                'eps':         1e-8,
            })

        if not groups:
            raise ValueError('Muon received no trainable parameters.')

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        betas=tuple(adamw_betas), weight_decay=adamw_wd, eps=1e-8, _mode='muon')
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

        for p in group['params']:
            if p.grad is None:
                continue

            g = p.grad
            state = self.state[p]

            # Initialise momentum buffer
            if 'buf' not in state:
                state['buf'] = torch.clone(g).detach()
            else:
                state['buf'].mul_(momentum).add_(g)

            update = g.add(state['buf'], alpha=momentum) if nesterov else state['buf'].clone()

            # Orthogonalise: reshape to 2-D, apply Newton-Schulz, reshape back
            orig_shape = update.shape
            update_2d = update.reshape(orig_shape[0], -1)
            update_2d = _zeropower_via_newtonschulz5(update_2d, steps=ns_steps)
            update = update_2d.reshape(orig_shape)

            # Scale by RMS of the original gradient for magnitude control
            scale = g.norm() / (update.norm() + 1e-8)
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
            t   = state['step']
            m   = state['exp_avg']
            v   = state['exp_avg_sq']

            m.mul_(b1).add_(g, alpha=1.0 - b1)
            v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

            # Bias-corrected estimates
            m_hat = m / (1.0 - b1 ** t)
            v_hat = v / (1.0 - b2 ** t)

            # Weight decay (decoupled)
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)

            p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
