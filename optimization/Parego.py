import torch
import torch.nn as nn
import time
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.acquisition import AcquisitionFunction


class CostCooledEI(AcquisitionFunction):
    def __init__(self, base_acq: AcquisitionFunction, cost_model: nn.Module,
                 bounds: torch.Tensor, alpha: float = 1.0):
        super().__init__(model=base_acq.model)
        self.base_acq = base_acq
        self.cost_model = cost_model
        self.bounds = bounds
        self.alpha = alpha
        print(f"[CostCooledEI] Initialized with model={base_acq.model.__class__.__name__}, alpha={alpha}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # NOTE: No @torch.no_grad -- allow gradients for optimize_acqf
        t_start = time.time()
        batch, q, d = X.shape
        print(f"[CostCooledEI] forward start: X.shape={X.shape}")

        # 1) base acquisition (requires grad)
        t0 = time.time()
        ei = self.base_acq(X).view(batch)  # [batch]
        t1 = time.time()
        print(f"[CostCooledEI] base_acq computed in {(t1 - t0):.4f}s: ei.shape={ei.shape}")

        # 2) cost prediction (may be non-differentiable)
        t0 = time.time()
        X_flat = X.view(-1, d)
        X_unn = unnormalize(X_flat, self.bounds)
        # wrap cost prediction in no_grad to avoid unwanted grad ops
        with torch.no_grad():
            costs_flat = self.cost_model.predict(X_unn)
        costs = costs_flat.view(batch, q)
        t1 = time.time()
        print(f"[CostCooledEI] cost_model.predict in {(t1 - t0):.4f}s: costs.shape={costs.shape}, range=({costs.min().item():.3f}, {costs.max().item():.3f})")

        # 3) cooling (gradient flows through ei only)
        t0 = time.time()
        max_cost = costs.max(dim=-1).values  # [batch]
        cooled = ei / (max_cost.pow(self.alpha) + 1e-12)
        t1 = time.time()
        print(f"[CostCooledEI] cooled EI computed in {(t1 - t0):.4f}s: cooled.shape={cooled.shape}")

        print(f"[CostCooledEI] forward total time={(time.time() - t_start):.4f}s")
        return cooled


def build_parego_ei(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    bounds: torch.Tensor,
    ref_point: torch.Tensor,
    #cost_model_fn,
    alpha: float = 1.0,
    q: int = 1,
    num_restarts: int = 5,
    raw_samples: int = 128,
):
    overall_start = time.time()
    print(f"[build] Starting build_parego_cost_cooled with q={q}, alpha={alpha}, restarts={num_restarts}, samples={raw_samples}")

    device = train_X.device
    print(f"[build] Using device: {device}")
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    bounds = bounds.to(device)
    ref_point = ref_point.to(device)

    # Wrap cost function
    wrap_start = time.time()
    class CostWrapper(nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn
        @torch.no_grad()
        def predict(self, X):
            return self.fn(X)
    #cost_model = CostWrapper(cost_model_fn).to(device)
    print(f"[build] Wrapped cost model in {(time.time() - wrap_start):.4f}s")

    # Pareto weights and scalarize
    pw_start = time.time()
    w = torch.rand(train_Y.size(-1), device=device)
    w /= w.sum()
    y_scalar = ((train_Y - ref_point) * w).max(dim=-1).values.unsqueeze(-1)
    print(f"[build] Scalarized Y in {(time.time() - pw_start):.4f}s: y_scalar.shape={y_scalar.shape}, weights={w}")

    # Fit GP
    gp_start = time.time()
    gp = SingleTaskGP(
        train_X, y_scalar,
        input_transform=Normalize(d=train_X.size(-1)),
        outcome_transform=Standardize(m=1),
    ).to(device)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    print(f"[build] GP initialized in {(time.time() - gp_start):.4f}s")

    fit_start = time.time()
    fit_gpytorch_mll(mll, options={"maxiter": 20})
    print(f"[build] GP fit in {(time.time() - fit_start):.4f}s")

    best_f = y_scalar.max().item()
    print(f"[build] best_f={best_f:.4f}")

    # Choose acquisition
    acq_start = time.time()
    if q == 1:
        base_acq = LogExpectedImprovement(model=gp, best_f=best_f)
        print("[build] Using LogExpectedImprovement (analytic EI)")
    else:
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([raw_samples]))
        base_acq = qExpectedImprovement(
            model=gp, best_f=best_f, sampler=sampler#, prune_baseline=True
        )
        print("[build] Using qExpectedImprovement (MC EI)")
    print(f"[build] Acquisition created in {(time.time() - acq_start):.4f}s")

    # Wrap with CostCooledEI
    #wrap_acq_start = time.time()
    #cost_cooled = CostCooledEI(
    #    base_acq=base_acq,
    #    cost_model=cost_model,
    #    bounds=bounds,
    #    alpha=alpha,
    #)
    #print(f"[build] CostCooledEI wrap in {(time.time() - wrap_acq_start):.4f}s")

    # Optimize
    print(f"[build] Total build_parego_cost_cooled time={(time.time() - overall_start):.4f}s")
    return base_acq
