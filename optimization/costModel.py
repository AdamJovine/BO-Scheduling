import torch
import torch.nn as nn

class MyTorchCostModel(nn.Module):
    """
    A torch.nn.Module cost model that predicts evaluation cost
    from normalized inputs X_norm (shape [N, d]) by reconstructing
    the discrete block_config and calling optimizer._get_block_info.
    """

    def __init__(self, optimizer):
        """
        optimizer: your CostAwareSchedulingOptimizer instance,
                   which must have .block_params and .bounds
                   already set up.
        """
        super().__init__()
        self.optimizer = optimizer
        # how many dims encode the block_config
        self.n_block = len(optimizer.block_params)

        # grab the original bounds so we can unnormalize
        # bounds is [d, 2]
        device = optimizer.bounds.device
        lb = optimizer.bounds[:, 0].to(device)
        ub = optimizer.bounds[:, 1].to(device)
        rng = ub - lb
        rng[rng==0.0] = 1.0

        # only keep the block‐param slices
        self.lb_block    = lb[:self.n_block]
        self.range_block = rng[:self.n_block]

    def forward(self, X_norm: torch.Tensor) -> torch.Tensor:
        """
        X_norm: [N, d] tensor on GPU, in normalized [0,1] space.
        Returns: [N] tensor of costs, on the same device/dtype.
        """
        # 1) unnormalize the first n_block dims and round to int
        X_block = self.lb_block + X_norm[:, :self.n_block] * self.range_block
        X_block_int = X_block.round().long().cpu().tolist()

        # 2) look up cost for each config
        costs = []
        for row in X_block_int:
            block_config = tuple(int(v) for v in row)
            _, base_cost, _ = self.optimizer._get_block_info(block_config)
            costs.append(base_cost)

        # 3) return torch tensor on the original device
        return torch.tensor(costs, dtype=X_norm.dtype, device=X_norm.device)
