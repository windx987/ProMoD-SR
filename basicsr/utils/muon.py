"""
Muon optimizer wrapper around the official torch.optim.Muon (PyTorch >= 2.9).

torch.optim.Muon only handles 2D hidden-layer weight matrices; per its docs,
"other parameters, such as bias, and embedding, should be optimized by a
standard method such as AdamW". BasicSR's training loop expects a single
optimizer object, so this wrapper splits parameters into

    ndim == 2  -> torch.optim.Muon   (Newton-Schulz orthogonalized momentum)
    otherwise  -> torch.optim.AdamW  (biases, norms, conv kernels)

and exposes the combined param_groups so LR schedulers and BasicSR's
warmup/_set_lr machinery mutate the real underlying groups.

YAML usage:
  optim_g:
    type: Muon
    lr: !!float 5e-4
    weight_decay: 0
    betas: [0.9, 0.99]      # AdamW group only
    # momentum: 0.95, nesterov: true, ns_steps: 5   (Muon group, optional)

Requires the SISR29 env (PyTorch 2.9.1+cu126) on glider.
"""

import torch
from torch.optim import Optimizer


class Muon(Optimizer):

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        betas: tuple = (0.9, 0.99),
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
        adjust_lr_fn=None,
    ):
        try:
            from torch.optim import Muon as TorchMuon
        except ImportError:
            raise ImportError(
                f'torch.optim.Muon requires PyTorch >= 2.9 (found {torch.__version__}). '
                'Use the SISR29 conda env, or optim type AdamW.')

        raw = list(params)
        if raw and isinstance(raw[0], dict):
            all_params = [p for g in raw for p in g['params']]
        else:
            all_params = raw

        muon_params = [p for p in all_params if p.requires_grad and p.ndim == 2]
        adamw_params = [p for p in all_params if p.requires_grad and p.ndim != 2]
        if not muon_params:
            raise ValueError('Muon received no 2D trainable parameters.')

        self._muon = TorchMuon(
            muon_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            adjust_lr_fn=adjust_lr_fn,
        )
        self._adamw = torch.optim.AdamW(
            adamw_params, lr=lr, betas=tuple(betas), weight_decay=weight_decay)

        # Deliberately no super().__init__(): param_groups is a property over
        # the sub-optimizers' live group dicts, so scheduler lr writes reach
        # the real optimizers. isinstance(self, Optimizer) still holds for
        # torch LR scheduler type checks.
        self.defaults = dict(self._muon.defaults)

    @property
    def param_groups(self):
        return self._muon.param_groups + self._adamw.param_groups

    @property
    def state(self):
        combined = {}
        combined.update(self._muon.state)
        combined.update(self._adamw.state)
        return combined

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._muon.step()
        self._adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            'muon': self._muon.state_dict(),
            'adamw': self._adamw.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self._muon.load_state_dict(state_dict['muon'])
        self._adamw.load_state_dict(state_dict['adamw'])
