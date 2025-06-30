# Helper Functions (can be in the same file or imported from .utils)
# Standard Library Imports
import itertools
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Optional, Tuple, Dict

# Third-Party Imports
import numpy as np
import pandas as pd
import torch
from gpytorch.mlls import MarginalLogLikelihood # Base class for MLL

# BoTorch Imports
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.fit import fit_gpytorch_model
from botorch.models.model import Model # Model base class
from botorch.models.pairwise_gp import (
    PairwiseGP,
    PairwiseLaplaceMarginalLogLikelihood,
)
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize

# Local Application Imports (Ensure paths are correct)
from utils.helpers import compute_hypervolume, compute_pareto_frontier_mask
from .utils import evaluate_candidate_wrapper, flatten_to_1d, evaluate_batch,track_iteration_metrics ,   fit_preference_model # Assuming these exist


# --- Custom Acquisition Function: UCB on PairwiseGP Latent Function ---
# (Keep the AnalyticLatentUCB class definition as provided in the previous answer)
class AnalyticLatentUCB(AnalyticAcquisitionFunction):
    """Analytic UCB based on the latent function of a PairwiseGP model."""
    def __init__(
        self, model: PairwiseGP, beta: float, maximize: bool = True
    ) -> None:
        super().__init__(model=model)
        self.maximize = maximize
        if not isinstance(beta, torch.Tensor):
             # Ensure beta matches model's dtype for calculations
             beta = torch.tensor(beta, dtype=model.datapoints.dtype if hasattr(model, 'datapoints') else torch.get_default_dtype())
        self.register_buffer("beta", beta.to(model.device if hasattr(model, 'device') else beta.device)) # Ensure beta is on model's device

    @torch.enable_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate UCB on the candidate set X (batch_shape x q x d)."""
        posterior = self.model.posterior(X)
        mean = posterior.mean
        variance = posterior.variance.clamp_min(1e-9)
        delta = (self.beta.sqrt() * variance.sqrt())
        ucb = (mean + delta) if self.maximize else (mean - delta)
        return ucb.view(X.shape[:-2] + X.shape[-2:-1])


# --- Helper for Generating Noisy Comparisons ---
# (Keep the generate_noisy_comparisons function definition as provided in the previous answer)
def generate_noisy_comparisons(
    Y_latent: torch.Tensor,
    comparison_noise: float = 0.1,
    max_pairs: Optional[int] = None,
    indices: Optional[List[int]] = None
) -> torch.Tensor:
    """Generates noisy pairwise comparisons based on latent values."""
    n = Y_latent.shape[0]
    device = Y_latent.device
    dtype = Y_latent.dtype
    if n < 2: return torch.empty((0, 2), dtype=torch.long, device=device)
    all_local_pairs = np.array(list(itertools.combinations(range(n), 2)))
    if not all_local_pairs.size: return torch.empty((0, 2), dtype=torch.long, device=device)
    if max_pairs is not None and max_pairs < len(all_local_pairs):
        select_idx = np.random.choice(len(all_local_pairs), min(max_pairs, len(all_local_pairs)), replace=False)
        comp_local_pairs = all_local_pairs[select_idx]
    else: comp_local_pairs = all_local_pairs
    if comp_local_pairs.shape[0] == 0: return torch.empty((0, 2), dtype=torch.long, device=device)
    idx0, idx1 = comp_local_pairs[:, 0], comp_local_pairs[:, 1]
    c0 = Y_latent[idx0] + torch.randn_like(Y_latent[idx0]) * comparison_noise
    c1 = Y_latent[idx1] + torch.randn_like(Y_latent[idx1]) * comparison_noise
    winner_is_c0 = (c0 >= c1)
    # Use tensors directly for where
    winners_local = torch.where(winner_is_c0, torch.tensor(idx0, device=device), torch.tensor(idx1, device=device))
    losers_local = torch.where(winner_is_c0, torch.tensor(idx1, device=device), torch.tensor(idx0, device=device))
    if indices is not None:
        original_indices_t = torch.tensor(indices, device=device, dtype=torch.long)
        comp_pairs_original = torch.stack((original_indices_t[winners_local], original_indices_t[losers_local])).t()
    else: comp_pairs_original = torch.stack((winners_local, losers_local)).t()
    return comp_pairs_original.long()


# --- Helper for Optimizing Acquisition Function ---
def optimize_ucb_acquisition(
    model: PairwiseGP,
    beta: float,
    bounds_t: torch.Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Optimizes the AnalyticLatentUCB acquisition function."""
    print("[Helper] Optimizing acquisition function...")
    dim = bounds_t.shape[1]
    acq_func = AnalyticLatentUCB(model=model, beta=beta)

    try:
        new_X_unit, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.tensor([[0.0] * dim, [1.0] * dim], device=device, dtype=dtype), # Unit cube
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
        )
        new_X = model.input_transform.untransform(new_X_unit.detach())
        print(f"[Helper] Acquisition optimization successful, got {new_X.shape[0]} candidates.")
        return new_X.to(device=device, dtype=dtype) # Ensure device/dtype

    except Exception as e:
        print(f"[Helper] Warning: Acquisition optimization failed: {e}. Sampling randomly.")
        # Fallback: Ensure random samples match shape expected (q x dim)
        new_X = draw_sobol_samples(bounds=bounds_t, n=1, q=q).squeeze(0)
        return new_X.to(device=device, dtype=dtype) # Ensure device/dtype


# --- Main BO Function ---
def run_pairwise_bo_latent_ucb(
    objective_fn: Callable,
    bounds: np.ndarray,
    licenses: list,
    utility_fn: Callable,
    ref_point: tuple, # Renamed for clarity, expects tuple/list
    initial_points: int = 10,
    q: int = 1,
    iterations: int = 10,
    num_restarts: int = 10,
    raw_samples: int = 512,
    beta: float = 2.0,
    comparison_noise: float = 0.1,
    new_comps_strategy: str = "within_batch",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise BO using UCB on the latent utility function of PairwiseGP.
    Refactored with helper functions.
    """
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    print(f"[PairwiseUCB] Running on device: {device}")

    dim = bounds.shape[0]
    bounds_t = torch.tensor(bounds.T, dtype=dtype, device=device)
    metrics = {'iteration': [], 'best_utility': [], 'hypervolume': []}

    # --- Initial Design & Evaluation ---
    X_init = draw_sobol_samples(bounds=bounds_t, n=initial_points, q=1).squeeze(1)
    X_init = X_init.to(device=device, dtype=dtype)
    Y_raw_init = evaluate_batch(X_init, objective_fn, licenses, device, dtype)
    num_objectives = Y_raw_init.shape[1]

    # --- Initial Utility & Comparisons ---
    U_latent_init = utility_fn(Y_raw_init).view(-1).to(device, dtype=dtype)
    print(f"[PairwiseUCB] Initial dataset: X={X_init.shape}, Y_raw={Y_raw_init.shape}, U_latent={U_latent_init.shape}")
    comps_init = generate_noisy_comparisons(U_latent_init, comparison_noise=comparison_noise)
    print(f"[PairwiseUCB] Generated {comps_init.shape[0]} initial comparison pairs on {comps_init.device}.")

    # --- Ref Point & Initial Metrics ---
    ref_point_t = torch.tensor(ref_point, dtype=dtype, device=device)
    if ref_point_t.shape[0] != num_objectives:
        raise ValueError(f"Ref point dim ({ref_point_t.shape[0]}) != objectives dim ({num_objectives}).")
    track_iteration_metrics(metrics, 0, U_latent_init, Y_raw_init, ref_point_t)

    # --- Initialize Data Tensors ---
    X = X_init
    Y_raw_all = Y_raw_init
    U_latent_all = U_latent_init
    comps = comps_init

    # --- BO Loop ---
    for it in range(1, iterations + 1):
        current_n_points = X.shape[0]
        print(f"\n[PairwiseUCB] Iteration {it}/{iterations}")
        print(f"[PairwiseUCB] Dataset size: X={X.shape}, comps={comps.shape} on device {X.device}/{comps.device}")

        # --- Fit Model & Acquire Candidates ---
        model = fit_preference_model(X, comps, bounds_t, device, dtype)

        if model is None: # Handle case where fitting failed
            print("[PairwiseUCB] Model fitting failed. Sampling randomly.")
            new_X = draw_sobol_samples(bounds=bounds_t, n=1, q=q).squeeze(0).to(device, dtype=dtype)
        else:
            new_X = optimize_ucb_acquisition(
                model, beta, bounds_t, q, num_restarts, raw_samples, device, dtype
            )

        # --- Evaluate New Batch ---
        new_Y_raw = evaluate_batch(new_X, objective_fn, licenses, device, dtype)

        # --- Calculate *True* Latent Utility & Generate New Comparisons ---
        new_U_latent = utility_fn(new_Y_raw).view(-1).to(device, dtype=dtype)
        new_indices = list(range(current_n_points, current_n_points + q))
        new_comps = torch.empty((0, 2), dtype=torch.long, device=device) # Default empty

        if q >= 2 and new_comps_strategy == "within_batch":
            new_comps = generate_noisy_comparisons(
                new_U_latent, comparison_noise=comparison_noise, indices=new_indices
            )
        elif new_comps_strategy == "vs_best":
            if X.shape[0] > 0 and model is not None:
                 try:
                     with torch.no_grad():
                         latent_means = model.posterior(X).mean.view(-1)
                     best_current_idx = torch.argmax(latent_means).item()
                     best_current_u_latent = U_latent_all[best_current_idx]
                     new_comps_list = []
                     for i in range(q):
                          u_new_noisy = new_U_latent[i] + torch.randn(1, device=device).item() * comparison_noise
                          u_best_noisy = best_current_u_latent + torch.randn(1, device=device).item() * comparison_noise
                          winner = new_indices[i] if u_new_noisy >= u_best_noisy else best_current_idx
                          loser = best_current_idx if u_new_noisy >= u_best_noisy else new_indices[i]
                          new_comps_list.append([winner, loser])
                     if new_comps_list:
                         new_comps = torch.tensor(new_comps_list, dtype=torch.long, device=device)
                 except Exception as comp_err: print(f"[PairwiseUCB] Warning: Error generating vs_best comparisons: {comp_err}")
            else: print("[PairwiseUCB] Cannot generate vs_best comparisons (no prior points or model fit failed).")
        else: print(f"[PairwiseUCB] No new comparisons generated (q={q} or invalid strategy).")

        print(f"[PairwiseUCB] Generated {new_comps.shape[0]} new comparison pairs on {new_comps.device}.")

        # --- Append Data ---
        X = torch.cat([X, new_X], dim=0)
        Y_raw_all = torch.cat([Y_raw_all, new_Y_raw], dim=0)
        U_latent_all = torch.cat([U_latent_all, new_U_latent], dim=0)
        comps = torch.cat([comps, new_comps], dim=0)

        # --- Track Metrics ---
        track_iteration_metrics(metrics, it, U_latent_all, Y_raw_all, ref_point_t)

    # --- End of BO Loop ---

    # --- Build Final DataFrames ---
    print("[PairwiseUCB] Preparing final results...")
    sample_cols = [f"param_{i}" for i in range(dim)] + [f"raw_obj_{j}" for j in range(num_objectives)]
    samples_np = torch.cat([X, Y_raw_all], dim=1).cpu().numpy()
    samples_df = pd.DataFrame(samples_np, columns=sample_cols)
    metrics_df = pd.DataFrame(metrics)
    print("[PairwiseUCB] Optimization complete.")
    return samples_df, metrics_df