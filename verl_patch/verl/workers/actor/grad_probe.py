import torch


class GradProbe:

    def __init__(self, params, reference='pg', sharded=True, dtype=torch.float32):
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("GradProbe needs at least one parameter requiring grad")
        self.reference = reference
        self.sharded = sharded
        self.dtype = dtype
        self._prev = [None] * len(self.params)
        self._ref = [None] * len(self.params)
        self._have_ref = False
        self._reset()

    def _reset(self):
        self.order = []
        self.sq = {}
        self.dot = {}
        self.cum_sq = []
        self._have_ref = False


