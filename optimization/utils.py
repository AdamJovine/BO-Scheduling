from gpytorch import ExactMarginalLogLikelihood
import torch
import json
import itertools

from datetime import datetime
import pandas as pd 
import numpy as np
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor

import multiprocessing as mp
from itertools import combinations, cycle
from botorch.fit import fit_gpytorch_model
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from utils.helpers import compute_hypervolume, compute_pareto_frontier_mask

import os, glob, re, time, random
import torch
from torch.quasirandom import SobolEngine
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_model

def load_block_files(blocks_dir, bad_prefix="bad"):
    """Scan a directory for all valid block CSVs, parse their
    size_cutoff, frontloading, num_blocks, and return a list of dicts."""
    files = glob.glob(os.path.join(blocks_dir, '*.csv'))
    good = [f for f in files if not os.path.basename(f).startswith(bad_prefix)]
    blocks = []
    for path in good:
        fn = os.path.basename(path)
        m = {k: int(v) for k,v in re.findall(r"(size_cutoff|frontloading|num_blocks)(\d+)", fn)}
        if len(m)==3:
            blocks.append({**m, 'path': path})
    return blocks

def setup_bounds_and_names(block_params: dict, seq_params: dict):
    """Turn your two dicts into:
       - a single bounds tensor [[lb,ub],…]  
       - a flat list of param_names in the same order."""
    bounds = []
    names = []
    # blocks first
    for name, vals in block_params.items():
        lb, ub = float(min(vals)), float(max(vals))
        bounds.append([lb, ub])
        names.append(name)
    # then sequencing
    for name, (lb, ub) in seq_params.items():
        bounds.append([lb, ub])
        names.append(name)
    return torch.tensor(bounds, dtype=torch.double), names

def sobol_samples(n: int, dim: int, scramble=True):
    """Draw n points in [0,1]^dim using Sobol."""
    return SobolEngine(dim, scramble=scramble).draw(n)

def time_and_cost(fn, *args, base_cost=1.0, cost_ratio=1.0):
    """Run fn(*args), measure elapsed, and return (result, cost)."""
    start = time.time()
    out = fn(*args)
    elapsed = time.time() - start
    return out, max(base_cost, elapsed)

def fit_twin_gps(X: torch.Tensor, Y: torch.Tensor):
    """Fit an independent SingleTaskGP per output dim, with standard transforms."""
    # Ensure X is 2D: [n_samples, n_features]
    if X.dim() > 2:
        X = X.reshape(X.size(0), -1)
    
    # Ensure Y is 2D: [n_samples, n_outputs]
    if Y.dim() == 3:
        Y = Y.squeeze(1)  # Remove middle dimension if it's [n,1,D]
    elif Y.dim() == 1:
        Y = Y.unsqueeze(1)  # Add output dimension if it's [n]
    
    models = []
    for i in range(Y.size(-1)):
        y = Y[:, i:i+1] + 1e-6 * torch.randn_like(Y[:, i:i+1])
        gp = SingleTaskGP(
            train_X=X,
            train_Y=y,
            input_transform=None,
            outcome_transform=None,
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)
        models.append(gp)
    return ModelListGP(*models)

def pareto_mask(Y: torch.Tensor):
    """Return a boolean mask of Pareto‐optimal rows in Y."""
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.double)
    n = Y.size(0)
    mask = torch.ones(n, dtype=torch.bool)
    for i in range(n):
        if not mask[i]:
            continue
        others = Y[mask]
        if ((others <= Y[i]).all(dim=1) & (others < Y[i]).any(dim=1)).any():
            mask[i] = False
    return mask








# --- Helper for Tracking Metrics ---
def track_iteration_metrics(
    metrics_dict: Dict,
    iteration: int,
    U_latent_all: torch.Tensor,
    Y_raw_all: torch.Tensor,
    ref_point_t: torch.Tensor,
) -> None:
    """Calculates and records metrics for the current iteration."""
    if U_latent_all.numel() == 0:
         print(f"[Helper] No utility data to track for iteration {iteration}.")
         best_u_true = -float('inf')
         hv = 0.0
    else:
        best_u_true = U_latent_all.max().item()
        Y_raw_all_np = Y_raw_all.cpu().numpy()
        pareto_mask = compute_pareto_frontier_mask(Y_raw_all_np)
        front = Y_raw_all_np[pareto_mask]
        hv = compute_hypervolume(front, ref_point_t.cpu().numpy()) if front.size > 0 else 0.0

    metrics_dict['iteration'].append(iteration)
    metrics_dict['best_utility'].append(best_u_true)
    metrics_dict['hypervolume'].append(hv)
    print(f"[PairwiseUCB] Iter {iteration}: Best True Utility = {best_u_true:.4f}, Hypervolume = {hv:.4f}")


def flatten_to_1d(arr):
    """
    Convert the input (which may be 1D, 2D, or higher) to a 1D NumPy array.
    For 2D arrays with a single row, this returns a 1D array.
    For arrays with more than one row, it flattens them entirely.
    """
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return np.array([arr])
    elif arr.ndim == 1:
        return arr
    elif arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr.squeeze(axis=0)
        else:
            return arr.flatten()
    else:
        # For ndarrays with ndim > 2, flatten all but the first dimension if it is 1,
        # otherwise flatten completely.
        if arr.shape[0] == 1:
            return arr.reshape(arr.shape[1])
        else:
            return arr.flatten()
def evaluate_candidate(x, license, objective_fn):
    """
    Evaluate a single candidate with a given license/environment.
    Automatically handles both torch.Tensor and np.ndarray inputs.
    """
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    return objective_fn(x, license)
def evaluate_candidate_star(args):
    return evaluate_candidate(*args)

# utils.py or wherever this is defined

def evaluate_candidate_wrapper(args):
    return evaluate_candidate(*args)

def evaluate_candidates_in_parallel(candidates, objective_fn, licenses):

    tasks = [(torch.tensor(c, dtype=torch.double), license, objective_fn)
             for c, license in zip(candidates, cycle(licenses))]

    with mp.get_context("spawn").Pool(processes=len(licenses)) as pool:
        metrics = pool.map(evaluate_candidate_wrapper, tasks)

    return list(zip(candidates, metrics))  


# ========= Outcome & Preference Modeling =========

def fit_outcome_model(X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
    model = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        input_transform=Normalize(d=X.shape[-1]),
        outcome_transform=Standardize(m=Y.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def fit_pref_model(Y: torch.Tensor, comps: torch.Tensor) -> PairwiseGP:
    model = PairwiseGP(Y, comps, input_transform=Normalize(d=Y.shape[-1]))
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model

def fit_preference_model(
    X: torch.Tensor,
    comps: torch.Tensor,
    bounds_t: torch.Tensor, # Original bounds tensor (can be on GPU)
    device: torch.device,   # The *target* device (GPU or CPU)
    dtype: torch.dtype,
    ):
    """
    Fits the PairwiseGP model, FORCING fitting on CPU to avoid internal device errors.
    The fitted model is returned on the original target device.
    """
    print("[Helper] Fitting preference model (forcing CPU for internal steps)...")
    if comps.shape[0] == 0:
        print("[Helper] Warning: No comparison data provided. Cannot fit model.")
        return None

    # --- Force CPU for Fitting ---
    cpu_device = torch.device("cpu")
    X_cpu = X.to(cpu_device)
    comps_cpu = comps.to(cpu_device)
    # Bounds tensor also needs to be on CPU for Normalize transform if applied before fitting
    bounds_t_cpu = bounds_t.to(cpu_device)
    # Define input transform on CPU
    input_transform_cpu = Normalize(d=X.shape[-1], bounds=bounds_t_cpu)

    # Initialize model ON CPU
    model_cpu = PairwiseGP(
        X_cpu,
        comps_cpu,
        input_transform=input_transform_cpu
    ).to(cpu_device, dtype=dtype) # Ensure model itself and params are CPU/correct dtype

    mll_cpu = PairwiseLaplaceMarginalLogLikelihood(model_cpu.likelihood, model_cpu)
    # --- End Force CPU ---

    fitted_model = None
    
    fitted_model = model_cpu.to(device)

    return fitted_model



# --- Helper for Evaluating a Batch of Candidates ---
def evaluate_batch(
    X_batch: torch.Tensor,
    objective_fn: Callable,
    licenses: list,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Evaluates a batch of points in parallel."""
    print(f"[Helper] Evaluating batch of size {X_batch.shape[0]}...")
    max_workers = min(len(licenses), X_batch.shape[0])
    eval_args = [(X_batch[i].cpu(), lic, objective_fn) # Pass CPU tensor
                 for i, lic in zip(range(X_batch.shape[0]), itertools.cycle(licenses))]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        raw_results = list(executor.map(evaluate_candidate_wrapper, eval_args))

    flat_results = [flatten_to_1d(r) for r in raw_results]
    Y_raw_batch = torch.tensor(np.vstack(flat_results), dtype=dtype, device=device) # Back to device
    print(f"[Helper] Evaluation complete, result shape: {Y_raw_batch.shape}")
    return Y_raw_batch

def generate_pref_comps(utils: torch.Tensor) -> torch.Tensor:
    """
    Generate pairwise preference comparisons based on utility values.
    Compares adjacent elements after sorting by utility.

    Args:
        utils: A 1D tensor of utility values.

    Returns:
        A tensor of comparison pairs (shape: [n_comps, 2]), where
        comps[i] = [idx_winner, idx_loser].
    """
    n = utils.shape[0]
    if n < 2:
        return torch.empty((0, 2), dtype=torch.long) # Return empty tensor, device doesn't matter yet

    # Ensure utils is 1D
    if utils.ndim != 1:
        raise ValueError(f"Expected 1D utility tensor, got shape {utils.shape}")

    # Sort by utility value to get indices
    # Perform sorting on the device of the input tensor
    sorted_indices = torch.argsort(utils, descending=True)

    # Create initial comparison pairs (adjacent elements in sorted list)
    # Ensure pairs are created on the same device as sorted_indices
    comps = torch.stack([sorted_indices[:-1], sorted_indices[1:]], dim=1)

    # --- FIX: Ensure 'flip' is on CPU if 'comps' is on CPU ---
    # Determine the target device (usually CPU for this kind of logic unless explicitly handled otherwise)
    # It seems 'comps' ends up on CPU implicitly based on the error.
    # Let's be explicit and perform the comparison check potentially on the original device,
    # but move the resulting boolean mask `flip` to CPU before indexing.

    # Check if flipping is needed (utils[comps[:, 1]] > utils[comps[:, 0]])
    # Perform comparison on original device
    should_flip = utils[comps[:, 1]] > utils[comps[:, 0]]

    # Move the boolean mask `should_flip` to CPU *before* using it for indexing
    flip = should_flip.to('cpu') # <<< MOVE TO CPU

    # Perform the flip operation - NOW 'flip' is CPU, 'comps' is likely CPU.
    # Ensure comps is explicitly CPU before modification if needed, although
    # stack likely defaults to CPU if indices are CPU. Let's ensure comps is CPU.
    comps_cpu = comps.to('cpu')
    if torch.any(flip): # Only clone and modify if any flips are needed
        # Clone to avoid modifying original tensor if it's needed elsewhere (though unlikely here)
        temp = comps_cpu[flip, 0].clone()
        comps_cpu[flip, 0] = comps_cpu[flip, 1]
        comps_cpu[flip, 1] = temp

    # Return the potentially flipped comparisons (on CPU)
    # The calling function (`run_eubo_optimization`) should move it back to GPU if needed.
    return comps_cpu.long() # Ensure LongTensor type


def write_candidate_metrics_to_file(X_tensor, objective_fn, licenses, path=None):
    if path is None:
        #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        path = f"./preferences_pending_{timestamp}.json"

    print(f"[DEBUG] Saving candidate metrics to {path}")
    X_np = X_tensor.numpy()
    results = evaluate_candidates_in_parallel(X_np, objective_fn, licenses)

    pending = []

    for idx, (candidate, metrics) in enumerate(results):
        print(f"\n[TRACE] Candidate #{idx}")
        print(f"  Candidate: {candidate}")
        print(f"  Raw Metrics Type: {type(metrics)}")

        clean_metrics = []
        for m in metrics:
            if isinstance(m, (pd.Series, np.ndarray)):
                clean_val = m.values[0] if hasattr(m, "values") else m[0]
                clean_metrics.append(float(clean_val))
            elif torch.is_tensor(m):
                clean_metrics.append(m.item())
            else:
                clean_metrics.append(float(m))

        print(f"  Cleaned Metrics: {clean_metrics}")

        pending.append({
            "id": idx,
            "metrics": clean_metrics,
            "params": candidate.tolist()
        })

    try:
        with open(path, "w") as f:
            json.dump(pending, f, indent=2)
        print(f"[SUCCESS] Saved {len(pending)} candidates to {path}")
    except Exception as e:
        print(f"[FATAL] Failed to write JSON: {e}")
        print(f"[TRACE] Problem entry: {pending[0]}")
        raise

    print(f"[EUBO_M] Please manually rank candidates in: {path}")
    exit()

def get_latest_preferences_file():
    files = sorted(Path(".").glob("preferences_pending_*.json"), reverse=True)
    if not files:
        return None
    return files[0]  # Most recent based on timestamp
 
def load_ranked_preferences(path="./preferences_pending.json"):
    """
    Load manually ranked preferences from file and return candidates + comparisons.
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Check that each entry has a rank
    if not all("rank" in entry for entry in data):
        raise ValueError("❌ All entries must contain a 'rank' key.")

    data_sorted = sorted(data, key=lambda x: x["rank"])
    X = torch.tensor([d["params"] for d in data_sorted], dtype=torch.double)
    comps = torch.tensor(list(combinations(range(len(X)), 2)), dtype=torch.long)
    print('SUCCESSSSSS')
    return X, comps


