import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple

# Third-Party Imports
import numpy as np
import pandas as pd
import torch

# BoTorch Imports
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption # Correct path
from botorch.fit import fit_gpytorch_model
from botorch.models import ModelListGP # Preferred path
from botorch.models.deterministic import FixedSingleSampleModel
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.pairwise_gp import (
    PairwiseGP,
    PairwiseLaplaceMarginalLogLikelihood, # Keep this specific MLL
)
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf # Preferred path
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize

#from utils.helpers import compute_hypervolume, compute_pareto_frontier # Helpers likely separate
from optimization.utils import ( # Use relative import for utils within the same package/directory
    evaluate_candidate,
    evaluate_candidate_wrapper,
    flatten_to_1d,
    fit_outcome_model,
    fit_preference_model, 
    track_iteration_metrics, 
    evaluate_batch,
    fit_pref_model,           # Keep unique utils functions
    generate_pref_comps,
    write_candidate_metrics_to_file, # Keep unique utils functions
    load_ranked_preferences,  # Keep unique utils functions
    evaluate_candidate_star,  # Keep unique utils functions
    # Note: compute_hypervolume was removed from here, assumed to be in utils.helpers
)


def run_eubo_optimization_with_manual_preferences(objective_fn, bounds, licenses,
                                                 initial_points=10, iterations=5,
                                                 q=2,  # must be 2 for AnalyticExpectedUtilityOfBestOption
                                                 num_restarts=10, raw_samples=64):
    """
    Run EUBO with manually provided preferences using a JSON ranking file.
    On first run, writes candidate metrics to JSON and exits. On subsequent runs,
    uses previous JSON to fit preference model and proposes q=2 new candidates.
    """
    def get_latest_valid_pref_file():
        all_files = sorted(Path('.').glob('preferences_pending_*.json'), reverse=True)
        for f in all_files:
            try:
                data = json.load(open(f))
            except:
                continue
            if all('rank' in item for item in data):
                return f
        return None
    print("DOING EUBOM STUFF")
    latest = get_latest_valid_pref_file()
    if latest is None:
        # Initial generation of candidates
        dim = bounds.shape[0]
        lb = torch.tensor(bounds[:, 0], dtype=torch.double)
        ub = torch.tensor(bounds[:, 1], dtype=torch.double)
        init_X = lb + (ub - lb) * torch.rand(initial_points, dim, dtype=torch.double)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = Path(f'preferences_pending_{timestamp}.json')
        write_candidate_metrics_to_file(init_X, objective_fn, licenses=licenses, path=path)
        print(f'[EUBO_M] Initial suggestions written to {path}. Please rank and re-run.')
        exit()

    # Load and evaluate ranked preferences
    X, comps = load_ranked_preferences(latest)
    # Re-evaluate metrics
    inputs = [(x, licenses[i % len(licenses)], objective_fn) for i,x in enumerate(X)]
    with torch.multiprocessing.get_context('spawn').Pool(len(licenses)) as pool:
        Y_vals = pool.starmap(evaluate_candidate, inputs)
    Y = torch.tensor(np.array(Y_vals), dtype=torch.double)
    if Y.ndim == 1: Y = Y.unsqueeze(-1)
    if Y.ndim == 3: Y = Y.squeeze(-1)
    outcome_model = fit_outcome_model(X, Y)

    for it in range(iterations):
        print(f'[EUBO_M] Iteration {it+1}/{iterations}')
        pref_model = fit_pref_model(Y, comps)
        acq = AnalyticExpectedUtilityOfBestOption(
            pref_model=pref_model,
            outcome_model=FixedSingleSampleModel(model=outcome_model)
        )
        # Enforce q=2 for EHBO
        cand_X, _ = optimize_acqf(
            acq_function=acq,
            q=q,
            bounds=torch.tensor(bounds.T, dtype=torch.double),
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        next_path = Path(f'preferences_pending_{timestamp}.json')
        write_candidate_metrics_to_file(cand_X, objective_fn, licenses=licenses, path=next_path)
        print(f'[EUBO_M] Suggestions written to {next_path}. Please rank and re-run.')
        exit()

    # After manual loop, return final dataframe
    param_names = [f'param_{i}' for i in range(X.shape[1])]
    obj_names = [f'obj_{i}' for i in range(Y.shape[1])]
    data = torch.hstack([X, Y]).numpy()
    import pandas as pd
    return pd.DataFrame(data, columns=param_names + obj_names)

# Helper to compute Pareto front (non-dominated points)
def pareto_filter(points: np.ndarray) -> np.ndarray:
    is_dominated = np.zeros(points.shape[0], dtype=bool)
    for i, p in enumerate(points):
        if is_dominated[i]:
            continue
        is_dominated |= np.all(points <= p, axis=1) & np.any(points < p, axis=1)
        is_dominated[i] = False
    return points[~is_dominated]

import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple, Dict

# Third-Party Imports
import numpy as np
import pandas as pd
import torch
# Import gpytorch MLL explicitly if needed elsewhere, but fit_gpytorch handles it
# from gpytorch import ExactMarginalLogLikelihood # Included via fit_outcome_model helper

# BoTorch Imports
# Correct path for AnalyticExpectedUtilityOfBestOption
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_model
# Preferred path for ModelListGP (though not used directly here)
# from botorch.models import ModelListGP
from botorch.models.deterministic import FixedSingleSampleModel
# PairwiseGP models and MLL are handled by helpers
# from botorch.models.gp_regression import SingleTaskGP
# from botorch.models.pairwise_gp import (
#     PairwiseGP,
#     PairwiseLaplaceMarginalLogLikelihood,
# )
# Transforms handled by helpers
# from botorch.models.transforms.input import Normalize
# from botorch.models.transforms.outcome import Standardize
# Preferred path for optimize_acqf
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize

# Import local helpers explicitly (assuming they are in './utils')
from .utils import ( # Use relative import
    evaluate_batch, # Use this for evaluating batches
    fit_outcome_model, # Use this helper
    fit_preference_model, # Use this helper (handles CPU fitting)
    generate_pref_comps, # Use this helper
    track_iteration_metrics, # Use this helper for metrics
)

# Import shared helpers (adjust path if needed)
#from utils.helpers import compute_hypervolume, compute_pareto_frontier, compute_pareto_frontier_mask


import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Tuple, Dict

# Third-Party Imports
import numpy as np
import pandas as pd
import torch

# BoTorch Imports
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.fit import fit_gpytorch_model
from botorch.models.deterministic import FixedSingleSampleModel
# PairwiseGP models and MLL are handled by helpers
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize

# Import local helpers explicitly (assuming they are in './utils' or similar relative path)
from .utils import (
    evaluate_batch,
    fit_outcome_model,
    fit_preference_model, # Uses CPU fitting internally
    generate_pref_comps, # Returns CPU comparisons
    track_iteration_metrics,
    # Manual mode helpers are not needed for the automated function
    # write_candidate_metrics_to_file,
    # load_ranked_preferences,
)

# Import shared helpers (adjust path if needed)
# Assuming these helpers correctly handle numpy/torch CPU tensors
from utils.helpers import compute_hypervolume, compute_pareto_frontier, compute_pareto_frontier_mask

# Define the utility function structure - assuming custom_utility is defined elsewhere
# and imported or passed correctly. We need its signature here for clarity.
# from somewhere import custom_utility # Example import


# --- Main EUBO Function (Automated Preferences via Utility Function - CPU Execution) ---
def run_eubo_optimization(
    objective_fn: Callable,
    bounds: np.ndarray,
    licenses: list,
    utility_fn: Callable,       # e.g., custom_utility
    ref_point_values: tuple, # Reference point for raw objective hypervolume
    initial_points: int = 10,
    q: int = 2,                # Number of points per iteration (batch size)
    iterations: int = 5,
    num_restarts: int = 10,    # Restarts for acquisition function optimization
    raw_samples: int = 512,    # Samples for acquisition function optimization
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs Expected Utility of Best Option (EUBO) optimization using a provided
    utility function to simulate pairwise preferences automatically. Executes entirely on CPU.

    Maintains separate outcome (utility ~ params) and preference models.
    Preference model fitting uses a helper that forces CPU execution.
    Hypervolume is calculated based on the raw objectives returned by objective_fn.
    The provided utility_fn is used solely to generate scalar utilities for modeling
    and preference generation.

    Parameters:
        objective_fn: Evaluates candidates -> raw multi-objective values tensor.
        bounds: Parameter bounds array (dim, 2).
        licenses: List for parallel evaluation distribution.
        utility_fn: Function mapping raw multi-objective tensor (from objective_fn)
                    to a scalar utility tensor (shape: [..., 1]). E.g., custom_utility.
        ref_point_values: Tuple/list of reference point values for hypervolume calculation.
                          Length MUST match the dimension of the raw objectives
                          returned by objective_fn.
        initial_points: Number of initial Sobol points.
        q: Number of points selected per BO iteration (batch size). Must be >= 2.
        iterations: Number of BO iterations.
        num_restarts: Restarts for optimizing the acquisition function.
        raw_samples: Candidate samples for optimizing the acquisition function.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - samples_df: DataFrame of parameters, raw objectives, and utility.
            - metrics_df: DataFrame tracking best utility and hypervolume per iteration.
    """
    # --- Setup ---
    if q < 2:
        raise ValueError(f"EUBO requires q >= 2 for pairwise comparisons, but got q={q}.")

    # Force CPU execution
    device = torch.device("cpu")
    print(f"[EUBO] Execution device forced to: {device}")
    dtype = torch.double # Use double precision

    dim = bounds.shape[0]
    # Bounds tensor on CPU device
    bounds_t = torch.tensor(bounds.T, dtype=dtype, device=device)

    metrics = {'iteration': [], 'best_utility': [], 'hypervolume': []}

    # --- Initial Sobol Design & Evaluation ---
    print(f"[EUBO] Generating {initial_points} initial points on CPU...")
    # Generate on CPU device directly
    X_init = draw_sobol_samples(bounds=bounds_t, n=initial_points, q=1).squeeze(1)
    X_init = X_init.to(device=device, dtype=dtype) # Ensure CPU/dtype

    print(f"[EUBO] Evaluating {initial_points} initial points using helper...")
    # evaluate_batch helper handles parallel execution and returns results on the specified device (CPU)
    Y_raw_init = evaluate_batch(X_init, objective_fn, licenses, device, dtype)
    num_raw_objectives = Y_raw_init.shape[1]
    print(f"[EUBO] Raw objective dimension: {num_raw_objectives}")

    # --- Validate Reference Point Dimension against Raw Objectives ---
    if len(ref_point_values) != num_raw_objectives:
        raise ValueError(
            f"Reference point dimension ({len(ref_point_values)}) must match the "
            f"raw objective function output dimension ({num_raw_objectives})."
        )
    # Reference point tensor for hypervolume on CPU
    ref_point_t_hv = torch.tensor(ref_point_values, dtype=dtype, device=device)
    print(f"[EUBO] Reference point for HV validated (Dim: {ref_point_t_hv.shape[0]})")

    # --- Initial Utility Calculation ---
    # utility_fn takes raw objectives (CPU tensor) and returns scalar utility (CPU tensor)
    print("[EUBO] Calculating initial utility using provided utility_fn...")
    U_init = utility_fn(Y_raw_init).view(-1, 1) # Ensure 2D (N, 1) for outcome model
    U_init = U_init.to(device=device, dtype=dtype) # Ensure CPU/dtype
    print(f"[EUBO] Initial dataset: X={X_init.shape}, Y_raw={Y_raw_init.shape}, U={U_init.shape}")

    # --- Initial Metrics ---
    # track_iteration_metrics calculates HV on Y_raw_init using ref_point_t_hv
    print("[EUBO] Calculating initial metrics (Best Utility, Hypervolume on raw objectives)...")
    track_iteration_metrics(metrics, 0, U_init.view(-1), Y_raw_init, ref_point_t_hv)

    # --- Initialize Data Tensors for the Loop (all on CPU) ---
    X = X_init
    Y_raw_all = Y_raw_init
    U_all = U_init # Shape (N, 1)

    # --- Initialize Models (Fit once initially on CPU) ---
    outcome_model = None
    pref_model = None

    print("[EUBO] Fitting initial outcome model (Utility ~ Params) on CPU...")
    try:
        # fit_outcome_model returns model on CPU
        outcome_model = fit_outcome_model(X, U_all)
        if outcome_model:
            print("[EUBO] Initial outcome model fitted successfully.")
        else:
            print("[EUBO] Warning: Initial outcome model fitting returned None.")
    except Exception as e:
         print(f"[EUBO] Warning: Initial outcome model fitting failed: {e}.")

    # Initial preference model fitting (might have 0 comparisons initially)
    print("[EUBO] Attempting initial preference model fit (may have 0 comparisons)...")
    comps_init_cpu = generate_pref_comps(U_all.view(-1)) # Generate initial comparisons on CPU
    if comps_init_cpu.shape[0] > 0:
        try:
            # fit_preference_model handles CPU fitting and returns model on CPU
            pref_model = fit_preference_model(X, comps_init_cpu, bounds_t, device, dtype)
            if pref_model:
                print("[EUBO] Initial preference model fitted successfully.")
            else:
                 print("[EUBO] Warning: Initial preference model fitting returned None.")
        except Exception as e:
            print(f"[EUBO] Warning: Initial preference model fitting failed: {e}")
    else:
        print("[EUBO] No initial comparisons to fit preference model.")


    # --- BO Loop ---
    for it in range(1, iterations + 1):
        print(f"\n[EUBO] Iteration {it}/{iterations}")
        print(f"[EUBO] Dataset size: X={X.shape}, Y_raw={Y_raw_all.shape}, U={U_all.shape}")

        # --- 1. Generate Pairwise Comparisons from all utilities ---
        # generate_pref_comps takes 1D CPU tensor and returns comps on CPU
        comps_cpu = generate_pref_comps(U_all.view(-1))
        print(f"[EUBO] Generated {comps_cpu.shape[0]} comparison pairs from {U_all.shape[0]} utility values (on CPU).")

        # --- 2. Fit Preference Model ---
        # fit_preference_model handles CPU fitting internally and returns model on CPU
        print("[EUBO] Fitting preference model on CPU...")
        if comps_cpu.shape[0] > 0:
            try:
                pref_model = fit_preference_model(
                    X, comps_cpu, bounds_t, device, dtype # Ensure all inputs are CPU
                )
                if pref_model:
                    print("[EUBO] Preference model fitted successfully.")
                else:
                    print("[EUBO] Warning: Preference model fitting returned None.")
                    pref_model = None # Ensure it's None
            except Exception as e:
                 print(f"[EUBO] Warning: Preference model fitting failed: {e}")
                 pref_model = None # Ensure it's None on failure
        else:
            print("[EUBO] Warning: No comparison data available, cannot fit preference model.")
            pref_model = None


        # --- 3. Refit Outcome Model ---
        print("[EUBO] Refitting outcome model (Utility ~ Params) on CPU...")
        try:
            outcome_model = fit_outcome_model(X, U_all) # Fit on all CPU data
            if outcome_model:
                print("[EUBO] Outcome model refitted successfully.")
            else:
                print("[EUBO] Warning: Outcome model refitting returned None.")
                outcome_model = None # Ensure it's None
        except Exception as e:
             print(f"[EUBO] Warning: Outcome model refitting failed: {e}.")
             outcome_model = None # Ensure it's None on failure


        # --- 4. Select New Candidates ---
        new_X = None
        if pref_model is None or outcome_model is None:
            # Fallback to random sampling if either model failed
            print("[EUBO] Warning: One or both models unavailable. Sampling randomly on CPU.")
            # Generate on CPU device
            new_X = draw_sobol_samples(bounds=bounds_t, n=q, q=1).squeeze(1).to(device, dtype=dtype)
        else:
            # --- Define Acquisition Function (AEUBO) ---
            # Models are already on CPU
            print("[EUBO] Creating AEUBO acquisition function...")
            try:
                acq_func = AnalyticExpectedUtilityOfBestOption(
                    pref_model=pref_model, # CPU model
                    # Wrap outcome model (predicting scalar utility)
                    outcome_model=FixedSingleSampleModel(model=outcome_model) # CPU model
                )
                print("[EUBO] Acquisition function created successfully.")

                # --- Optimize Acquisition Function ---
                print("[EUBO] Optimizing acquisition function on CPU...")
                # Bounds for optimization are [0, 1]^d normalized space, on CPU
                norm_bounds_t = torch.tensor([[0.0] * dim, [1.0] * dim], device=device, dtype=dtype)

                new_X_unit, acq_value = optimize_acqf(
                    acq_function=acq_func,
                    bounds=norm_bounds_t, # Pass CPU bounds
                    q=q, # Request q candidates
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    options={"batch_limit": 5, "maxiter": 200},
                )
                print(f"[EUBO] Acquisition optimization complete. Acq value: {acq_value.item():.4f}")

                # Unnormalize candidates (all operations on CPU)
                print("[EUBO] Unnormalizing candidates...")
                input_transform = None
                if hasattr(outcome_model, 'input_transform') and outcome_model.input_transform is not None:
                     input_transform = outcome_model.input_transform
                     print("[EUBO] Using input transform from outcome model for unnormalization.")
                elif hasattr(pref_model, 'input_transform') and pref_model.input_transform is not None:
                     input_transform = pref_model.input_transform
                     print("[EUBO] Using input transform from preference model for unnormalization.")

                if input_transform is not None:
                     # Ensure transform is on CPU if it was somehow moved (shouldn't happen with helpers)
                     input_transform = input_transform.to(device)
                     new_X = input_transform.untransform(new_X_unit.detach())
                else:
                    print("[EUBO] Warning: Cannot find input_transform on models. Using manual unnormalize.")
                    new_X = unnormalize(new_X_unit.detach(), bounds=bounds_t)

                # Ensure final candidates are on CPU
                new_X = new_X.to(device, dtype=dtype)
                print(f"[EUBO] Generated {new_X.shape[0]} new candidates on CPU.")

            except Exception as e:
                print(f"[EUBO] Warning: Acquisition function creation or optimization failed: {e}. Sampling randomly.")
                # Generate on CPU device
                new_X = draw_sobol_samples(bounds=bounds_t, n=q, q=1).squeeze(1).to(device, dtype=dtype)

        # --- 5. Evaluate New Batch ---
        print(f"[EUBO] Evaluating {new_X.shape[0]} new candidates using helper...")
        # evaluate_batch returns results on the specified device (CPU)
        new_Y_raw = evaluate_batch(new_X, objective_fn, licenses, device, dtype)

        # --- 6. Calculate Utility for New Points ---
        # utility_fn operates on CPU tensors
        print("[EUBO] Calculating utility for new points...")
        new_U = utility_fn(new_Y_raw).view(-1, 1) # Ensure 2D (q, 1)
        new_U = new_U.to(device, dtype=dtype) # Ensure CPU/dtype
        print(f"[EUBO] Calculated utility for new points, shape: {new_U.shape}")

        # --- 7. Append Data (all tensors on CPU) ---
        X = torch.cat([X, new_X], dim=0)
        Y_raw_all = torch.cat([Y_raw_all, new_Y_raw], dim=0)
        U_all = torch.cat([U_all, new_U], dim=0)

        # --- 8. Track Metrics for this Iteration ---
        print("[EUBO] Calculating metrics for iteration", it)
        # Use the helper function to calculate and record metrics
        # Pass raw objectives (Y_raw_all) and the CPU ref_point_t_hv
        track_iteration_metrics(metrics, it, U_all.view(-1), Y_raw_all, ref_point_t_hv)


    # --- End of BO Loop ---

    # --- Build Final DataFrames ---
    print("\n[EUBO] Optimization loop finished. Preparing final results...")
    num_objectives_final = Y_raw_all.shape[1] # Use final number of raw objectives
    param_cols = [f"param_{i}" for i in range(dim)]
    raw_obj_cols = [f"raw_obj_{j}" for j in range(num_objectives_final)]
    utility_col = ["utility"]
    all_cols = param_cols + raw_obj_cols + utility_col

    # Concatenate final data (already on CPU) and convert to NumPy array for DataFrame
    samples_np = torch.cat([X, Y_raw_all, U_all], dim=1).numpy() # No need for .cpu()
    samples_df = pd.DataFrame(samples_np, columns=all_cols)

    metrics_df = pd.DataFrame(metrics)

    print("[EUBO] Optimization complete (CPU execution).")
    print(f"Final dataset size: {samples_df.shape[0]} samples.")
    # Handle case where metrics might be empty or NaN
    final_util = metrics_df['best_utility'].iloc[-1] if not metrics_df.empty else float('nan')
    final_hv = metrics_df['hypervolume'].iloc[-1] if not metrics_df.empty else float('nan')
    print(f"Final best utility: {final_util:.4f}")
    print(f"Final hypervolume (on raw objectives): {final_hv:.4f}")

    return samples_df, metrics_df

