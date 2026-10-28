from torch import is_grad_enabled, nn
from torch.utils.checkpoint import checkpoint


def checkpoint_wrapper(module: nn.Module) -> nn.Module:
    """Recompute ``module``'s activations during backward to save memory.

    Patches ``forward`` in place (like fairscale's wrapper), so parameter names
    and state-dict keys are unchanged. Without gradients it runs as usual.
    """
    forward = module.forward

    def checkpointed_forward(*args, **kwargs):
        if not is_grad_enabled():
            return forward(*args, **kwargs)
        return checkpoint(forward, *args, use_reentrant=False, **kwargs)

    module.forward = checkpointed_forward
    return module
