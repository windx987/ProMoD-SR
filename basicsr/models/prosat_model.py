from basicsr.utils.registry import MODEL_REGISTRY
from .pmd_model import PMDModel


@MODEL_REGISTRY.register()
class ProSATModel(PMDModel):
    """PMDModel (SRModel + patchwise testing) plus the ProSAT MoD warmup:
    sets the network's mod_ramp scalar from the current iteration so routing
    goes dense -> target capacity over [mod_ramp_start, mod_ramp_end]
    (optimizer steps, defaults per ProSAT.md: 50K dense, ramp to 100K).
    """

    def optimize_parameters(self, current_iter):
        train_opt = self.opt['train']
        start = int(train_opt.get('mod_ramp_start', 50000))
        end = int(train_opt.get('mod_ramp_end', 100000))
        if current_iter <= start:
            ramp = 0.0
        elif current_iter >= end:
            ramp = 1.0
        else:
            ramp = (current_iter - start) / float(end - start)
        self.get_bare_model(self.net_g).mod_ramp = ramp
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.mod_ramp = ramp
        super().optimize_parameters(current_iter)
