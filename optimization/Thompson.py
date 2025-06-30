from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling
from botorch.acquisition import AcquisitionFunction
import torch

class CostCooledTS(AcquisitionFunction):
    def __init__(
        self,
        model,
        alpha: float = 1.0,
        posterior_transform=None,
    ):
        super().__init__(model=model)
        # No num_samples or sample_shape here!
        self.ts = PathwiseThompsonSampling(
            model=model,
            posterior_transform=posterior_transform,
        )  # draws exactly one joint Thompson sample per batch :contentReference[oaicite:0]{index=0}
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # ensure X is (b, q, d)
        if X.dim() == 2:
            X = X.unsqueeze(0)
        # PathwiseTS will lazily initialize its path of length = batch_size on first call
        samples = self.ts(X)        # → Tensor of shape (b, q)
        return samples  * self.alpha