import itertools
from concurrent.futures import ProcessPoolExecutor

# Third-Party Imports
import numpy as np
import pandas as pd
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from torch.quasirandom import SobolEngine

# BoTorch Imports
from botorch.acquisition.multi_objective import (
    # qExpectedHypervolumeImprovement, # Keep if used elsewhere
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_model
from botorch.models import ModelListGP # Preferred path for ModelListGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize # Consolidated transform import
from botorch.models.transforms.outcome import Standardize # Consolidated transform import
from botorch.optim import optimize_acqf # Preferred path for optimize_acqf
from botorch.sampling import SobolQMCNormalSampler # Correct path for sampler
from botorch.utils.transforms import unnormalize

# Local Application Imports (Ensure paths are correct relative to your file structure)
# Assuming config/ and utils/ are sibling directories or accessible
from config.settings import AQ_FUNCTION, REF_POINT # If needed globally or in main script
from utils.helpers import compute_hypervolume, compute_pareto_frontier # If needed
# Assuming .utils is a sub-module in the same directory as the optimization functions
from .utils import evaluate_candidate, evaluate_candidate_wrapper, flatten_to_1d

# --- Optional GPU Check (keep separate for clarity) ---
print('torch.cuda.is_available():', torch.cuda.is_available())
if torch.cuda.is_available():
    try:
        # Note: nccl version might not be available on all PyTorch builds/OS
        nccl_version = torch.cuda.nccl.version()
        print('torch.cuda.nccl.version():', nccl_version)
    except (AttributeError, NameError):
        print("torch.cuda.nccl.version() not available.")
print('torch.cuda.is_available() : ' , torch.cuda.is_available())         # Should print: True
print('torch.cuda.nccl.version()) : ' , torch.cuda.nccl.version()) 
def evaluate_candidate_wrapper(args):
    """
    args is a 3‑tuple (x_tensor, license, objective_fn).
    Moves x onto CPU, then calls evaluate_candidate(x, lic, objective_fn).
    """
    x_tensor, lic, objective_fn = args
    x_cpu = x_tensor.detach().cpu().numpy()
    return evaluate_candidate(x_cpu, lic, objective_fn)
#def evaluate_candidate_wrapper(args):
#    return evaluate_candidate(*args)


# ==============================================================================
# Random Baseline Version (for baseline comparisons)
# ==============================================================================
def run_random_baseline(objective_fn, bounds, licenses, 
                        initial_points=10, iterations=10, 
                        n_candidates=1000, n_objectives=8):
    """
    Random parameter generation baseline.
    
    This function mimics the iterative structure of the BO routines:
      1. Evaluate an initial set of random candidates.
      2. For a number of iterations, generate n_candidates random points,
         randomly select a number equal to the number of licenses, evaluate them,
         and update the candidate set.
    
    Parameters:
      - objective_fn: The function that evaluates a candidate.
      - bounds: A (dim x 2) array where each row is [lower_bound, upper_bound] for a parameter.
      - licenses: A list specifying how many evaluations can run in parallel.
      - initial_points: Number of initial random candidates.
      - iterations: Number of iterations to run after the initial set.
      - n_candidates: Number of random candidate points to sample each iteration.
      - n_objectives: Number of objectives returned by objective_fn (only the first n_objectives are used).
    
    Returns:
      A pandas DataFrame with the parameter settings, objective values, and Pareto frontier.
    """
    dim = bounds.shape[0]
    
    # Generate initial candidate points.
    X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(initial_points, dim))
    print("Evaluating initial candidates (Random Baseline)...")
    with ProcessPoolExecutor(max_workers=len(licenses)) as executor:
        inputs = [
            (torch.tensor(candidate, dtype=torch.double), license, objective_fn)
            for candidate, license in zip(X, itertools.cycle(licenses))
        ]
        initial_results = list(executor.map(evaluate_candidate_wrapper, inputs))
    # Process the results to a 2D array.
    Y = np.array([flatten_to_1d(result[:n_objectives]) for result in initial_results])
    
    # Iteratively sample and evaluate additional random candidates.
    for it in range(iterations):
        print(f"Random Baseline Iteration {it+1}/{iterations}")
        candidate_X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_candidates, dim))
        # Randomly select a number of candidates equal to the number of available licenses.
        selected_indices = np.random.choice(n_candidates, size=len(licenses), replace=False)
        top_candidates = candidate_X[selected_indices]
        with ProcessPoolExecutor(max_workers=len(licenses)) as executor:
            inputs = [
                (torch.tensor(candidate, dtype=torch.double), license, objective_fn)
                for candidate, license in zip(top_candidates, itertools.cycle(licenses))
            ]
            batch_results = list(executor.map(evaluate_candidate_wrapper, inputs))
        for candidate, result in zip(top_candidates, batch_results):
            candidate_2d = np.atleast_2d(candidate)
            result_1d = flatten_to_1d(result[:n_objectives])
            X = np.vstack([X, candidate_2d])
            Y = np.vstack([Y, result_1d])
    
    param_names = [f"param_{i}" for i in range(dim)]
    obj_names = [f"obj_{i}" for i in range(n_objectives)]
    data = np.hstack((X, Y))
    df = pd.DataFrame(data, columns=param_names + obj_names)
    df['pareto'] = list(compute_pareto_frontier(Y))
    return df

def run_bayesian_optimization_ehvi(
    objective_fn,
    bounds: np.ndarray,
    ref_point_values: tuple, # Pass REF_POINT values explicitly
    licenses: list,
    initial_points: int = 10,
    iterations: int = 10,
    outer_n_samples: int = 256, # Number of samples for qNEHVI expectation
    num_restarts: int = 10,     # Restarts for acquisition optimization
    raw_samples: int = 512,     # Candidate samples for acquisition optimization
    n_objectives: int = None,    # Optional: limit number of objectives
    n_candidates: int = None 
) -> pd.DataFrame:
    """
    Multi-objective BO using qNEHVI with input/output transforms.

    Uses BoTorch's Standardize transform for outcomes and Normalize for inputs.
    Trains GPs on original data scales. Passes original reference point to qNEHVI.

    Parameters:
        objective_fn: Function evaluating candidates -> objective values.
        bounds: Parameter bounds array (dim, 2).
        ref_point_values: Tuple/list of reference point values for hypervolume.
        licenses: List of license dicts for parallel evaluation.
        initial_points: Number of initial Sobol points.
        iterations: Number of BO iterations.
        outer_n_samples: MC samples for qNEHVI acquisition function.
        num_restarts: Restarts for optimizing the acquisition function.
        raw_samples: Candidate samples for optimizing the acquisition function.
        n_objectives: Optional number of objectives to consider.

    Returns:
        DataFrame with parameters, objectives, and Pareto indicator.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    dim = bounds.shape[0]
    bounds_t = torch.tensor(bounds.T, dtype=dtype, device=device)

    # --- Initial Sobol Design ---
    sobol = SobolEngine(dimension=dim, scramble=True, seed=np.random.randint(1, 10000))
    X_init_unit = sobol.draw(initial_points, dtype=dtype).to(device=device)
    X_init = unnormalize(X_init_unit, bounds=bounds_t) # Get points in original space

    # --- Initial Evaluation ---
    max_workers = min(len(licenses), initial_points)
    print(f"[EHVI] Evaluating {initial_points} initial points using {max_workers} workers on {device}")
    init_args = [(X_init[i].cpu(), lic, objective_fn)
                 for i, lic in zip(range(initial_points), itertools.cycle(licenses))]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        raw_init_results = list(executor.map(evaluate_candidate_wrapper, init_args))

    # Process initial results (ensure consistent shape, convert to tensor)
    flat_init_results = [flatten_to_1d(r) for r in raw_init_results]
    Y_init = torch.tensor(np.vstack(flat_init_results), dtype=dtype, device=device)

    # --- Objective and Reference Point Handling ---
    if n_objectives is not None and n_objectives < Y_init.shape[1]:
        print(f"[EHVI] Truncating objectives from {Y_init.shape[1]} to {n_objectives}")
        Y_init = Y_init[:, :n_objectives]
        # Ensure ref_point_values tuple matches truncated objectives
        if len(ref_point_values) > n_objectives:
             ref_point_values = ref_point_values[:n_objectives]

    # Convert reference point tuple to tensor
    ref_point_t = torch.tensor(ref_point_values, dtype=dtype, device=device)
    num_objectives_actual = Y_init.shape[1] # Actual number used

    if ref_point_t.shape[0] != num_objectives_actual:
        raise ValueError(f"Reference point dimension ({ref_point_t.shape[0]}) does not match "
                         f"number of objectives ({num_objectives_actual}) after potential truncation.")

    print(f"[EHVI] Using reference point: {ref_point_t.cpu().numpy()}")
    print(f"[EHVI] Initial dataset: X shape {X_init.shape}, Y shape {Y_init.shape}")

    # --- Initialize Data Tensors ---
    X = X_init
    Y = Y_init # Keep Y in original scale

    
    def get_fitted_model(X_train, Y_train, bounds_t):
        """Fits a ModelListGP with input/output transforms correctly applied."""
        device = X_train.device
        num_outputs = Y_train.shape[-1]
        models = []

        # Define input transform (applied by the ModelListGP container)
        input_transform = Normalize(d=X_train.shape[-1], bounds=bounds_t)

        # Fit individual models, applying outcome transform within each SingleTaskGP
        for i in range(num_outputs):
            Y_train_i = Y_train[:, i : i + 1] # Get data for this output
            # Define outcome transform for this specific output
            outcome_transform = Standardize(m=1) # m=1 for single output

            try:
                # Create SingleTaskGP: Pass original X_train, Y_train_i
                # Apply ONLY the outcome_transform here. Input transform is handled by ModelListGP later.
                gp = SingleTaskGP(
                    X_train,
                    Y_train_i,
                    outcome_transform=outcome_transform
                ).to(device)

                mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
                # Fit the model (GPyTorch handles the outcome transform internally during fitting)
                fit_gpytorch_model(mll)
                models.append(gp) # Add the fitted GP to the list
            except Exception as e:
                print(f"[EHVI] Warning: Error fitting GP for objective {i}: {e}. Using default GP.")
                # Fallback: Untrained GP with transform (might still work for acquisition)
                gp = SingleTaskGP(
                    X_train,
                    Y_train_i,
                    outcome_transform=outcome_transform
                ).to(device)
                models.append(gp)

        # Create the final ModelListGP container with the individually fitted models
        final_model = ModelListGP(*models)

        # Attach the input transform to the ModelListGP container
        # This ensures it's applied *before* data reaches the individual GPs during prediction/acquisition
        final_model.input_transform = input_transform

        return final_model.to(device) # Ensure final model is on the 

    # --- Initial Model Fit ---
    model = get_fitted_model(X, Y, bounds_t) # MODIFIED CALL
    # --- BO Loop ---
    for it in range(1, iterations + 1):
        print(f"[EHVI] Iter {it}/{iterations} · X.shape={X.shape}, Y.shape={Y.shape}")

        # 1. Calculate current Pareto front from ORIGINAL Y data
        Y_np = Y.cpu().numpy() # Use original Y for Pareto calculation
        is_efficient = compute_pareto_frontier(Y_np) # Assumes compute_pareto returns boolean mask or indices
        # Need X corresponding to the pareto points in Y
        pareto_mask = np.zeros(Y_np.shape[0], dtype=bool)
        # This part depends on compute_pareto_frontier's return type.
        # Assuming it returns the actual Pareto points Y_pareto:
        # We need to find the indices in the original Y that match Y_pareto
        # A simpler way is to recalculate the pareto mask directly:
        pareto_mask = np.ones(Y_np.shape[0], dtype=bool)
        for i, y_i in enumerate(Y_np):
             if pareto_mask[i]: # Only check points not yet known to be dominated
                # Check if any other point dominates y_i
                dominates_y_i = np.any(
                    np.all(Y_np[pareto_mask & (np.arange(len(Y_np)) != i)] <= y_i, axis=1) &
                    np.any(Y_np[pareto_mask & (np.arange(len(Y_np)) != i)] < y_i, axis=1)
                )
                if dominates_y_i:
                    pareto_mask[i] = False # y_i is dominated
                else: # Check if y_i dominates any other points
                    dominated_by_y_i = (
                        np.all(y_i <= Y_np[pareto_mask & (np.arange(len(Y_np)) != i)], axis=1) &
                        np.any(y_i < Y_np[pareto_mask & (np.arange(len(Y_np)) != i)], axis=1)
                    )
                    pareto_mask[pareto_mask & (np.arange(len(Y_np)) != i)] = ~dominated_by_y_i


        X_baseline = X[pareto_mask] # Use original X corresponding to Pareto Y

        if X_baseline.shape[0] == 0:
            print("[EHVI] Warning: No non-dominated points found. Using all points as baseline.")
            X_baseline = X # Fallback

        # 2. Define Acquisition Function (qNEHVI)
        try:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([outer_n_samples]), seed=it)
            # Pass ORIGINAL ref_point_t. BoTorch handles transform internally.
            acq_func = qNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point_t, # Use original reference point tensor
                X_baseline=X_baseline, # Baseline points in original input space
                sampler=sampler,
                prune_baseline=True, # Prune points dominated by ref_point
            )

            # 3. Optimize Acquisition Function
            batch_size = min(len(licenses), 10) # Limit batch size for stability

            # optimize_acqf expects bounds in unit cube if model has input transform
            # It works correctly with Normalize transform
            candidates_unit, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.tensor([[0.0] * dim, [1.0] * dim], device=device, dtype=dtype), # Optimize in unit cube
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples, # Use raw_samples passed to function
                options={"batch_limit": 5, "maxiter": 200},
            )
            # Unnormalize candidates back to original space
            new_X = model.input_transform.untransform(candidates_unit.detach())

        except Exception as e:
            print(f"[EHVI] Warning: Error in acquisition optimization: {e}")
            print("[EHVI] Falling back to random sampling for this iteration.")
            # Fallback: sample random points in original space
            sobol_fallback = SobolEngine(dimension=dim, scramble=True, seed=1000 + it)
            new_X_unit = sobol_fallback.draw(len(licenses), dtype=dtype).to(device=device)
            new_X = unnormalize(new_X_unit, bounds=bounds_t) # Convert to original space

        # 4. Evaluate New Batch
        batch_size = new_X.shape[0]
        new_eval_args = [(new_X[i].cpu(), lic, objective_fn)
                         for i, lic in zip(range(batch_size), itertools.cycle(licenses))]

        with ProcessPoolExecutor(max_workers=batch_size) as executor:
            raw_new_results = list(executor.map(evaluate_candidate_wrapper, new_eval_args))

        # Process results (ensure consistent shape, convert to tensor)
        flat_new_results = [flatten_to_1d(r) for r in raw_new_results]
        new_Y = torch.tensor(np.vstack(flat_new_results), dtype=dtype, device=device)

        # Truncate if necessary
        if new_Y.shape[1] > num_objectives_actual:
             new_Y = new_Y[:, :num_objectives_actual]

        # 5. Update Dataset (original scale)
        X = torch.cat([X, new_X], dim=0)
        Y = torch.cat([Y, new_Y], dim=0) # Append original scale Y

        # 6. Refit Model with updated data
        model = get_fitted_model(X, Y, bounds_t) #

    # --- End of BO Loop ---

    # --- Final Results Processing ---
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy() # Final Y is already in original scale

    # Create column names
    param_cols = [f"param_{i}" for i in range(dim)]
    obj_cols = [f"obj_{j}" for j in range(Y_np.shape[1])]

    # Construct DataFrame
    df = pd.DataFrame(np.hstack([X_np, Y_np]), columns=param_cols + obj_cols)

    # Calculate final Pareto frontier on the full dataset
    final_pareto_mask = np.ones(Y_np.shape[0], dtype=bool)
    for i, y_i in enumerate(Y_np):
         if final_pareto_mask[i]:
            # Check if any other point dominates y_i
             dominates_y_i = np.any(
                 np.all(Y_np[final_pareto_mask & (np.arange(len(Y_np)) != i)] <= y_i, axis=1) &
                 np.any(Y_np[final_pareto_mask & (np.arange(len(Y_np)) != i)] < y_i, axis=1)
             )
             if dominates_y_i:
                 final_pareto_mask[i] = False
             else: # Check if y_i dominates any other points
                 dominated_by_y_i = (
                     np.all(y_i <= Y_np[final_pareto_mask & (np.arange(len(Y_np)) != i)], axis=1) &
                     np.any(y_i < Y_np[final_pareto_mask & (np.arange(len(Y_np)) != i)], axis=1)
                 )
                 final_pareto_mask[final_pareto_mask & (np.arange(len(Y_np)) != i)] = ~dominated_by_y_i

    # Add Pareto indicator to DataFrame
    df["pareto"] = final_pareto_mask

    print(f"[EHVI] Optimization complete. Found {np.sum(final_pareto_mask)} Pareto-optimal points.")
    print(f"[EHVI] Final dataset size: {df.shape}")

    return df

def run_bayesian_optimization_mobo(
    objective_fn, 
    bounds, 
    licenses, 
    initial_points=10, 
    iterations=10, 
    n_candidates=200, 
    n_objectives=8
):
    """
    Multi-objective BO using random scalarization (MOBO).
    This version has been modified to:
    1. Generate fewer candidate points per iteration (default n_candidates=200)
    2. Scale objectives based on REF_POINT
    3. Use hardware acceleration where available
    """
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get reference point for normalization
    ref_pt = torch.tensor(REF_POINT, dtype=torch.double, device=device)
    
    dim = bounds.shape[0]
    bounds_t = torch.tensor(bounds.T, dtype=torch.double, device=device)
    
    # Initial candidate points using Sobol (better than random uniform)
    sobol = SobolEngine(dimension=dim, scramble=True)
    unit_X = sobol.draw(initial_points).to(dtype=torch.double, device=device)
    X_t = bounds_t[0] + (bounds_t[1] - bounds_t[0]) * unit_X
    X = X_t.cpu().numpy()
    
    print("Evaluating initial candidates (MOBO)...")
    
    with ProcessPoolExecutor(max_workers=len(licenses)) as executor:
        inputs = [
            (X_t[i].cpu(), license, objective_fn)
            for i, license in zip(range(initial_points), itertools.cycle(licenses))
        ]
        initial_results = list(executor.map(evaluate_candidate_wrapper, inputs))
    
    # Extract and normalize objectives
    Y_raw = np.array([flatten_to_1d(result[:n_objectives]) for result in initial_results])
    Y = Y_raw / ref_pt.cpu().numpy()
    
    # Main BO loop
    for it in range(iterations):
        print(f"MOBO Iteration {it+1}/{iterations}")
        
        # Use different kernels for each objective if needed
        gps = []
        for obj in range(n_objectives):
            # Can replace with GPyTorch for GPU acceleration in a more complex implementation
            kernel = Matern(nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
            gp.fit(X, Y[:, obj])
            gps.append(gp)
        
        # Generate random weights for scalarization
        weights = np.random.dirichlet(np.ones(n_objectives))
        
        # Generate candidate points - could use QMC sampling for better coverage
        sobol_candidates = SobolEngine(dimension=dim, scramble=True, seed=it)
        unit_candidates = sobol_candidates.draw(n_candidates).to(dtype=torch.double, device=device)
        candidate_X_t = bounds_t[0] + (bounds_t[1] - bounds_t[0]) * unit_candidates
        candidate_X = candidate_X_t.cpu().numpy()
        
        # Predict and scalarize objectives
        scalarized = np.zeros(n_candidates)
        for i, gp in enumerate(gps):
            # Could potentially batch this with GPyTorch on GPU
            sample = gp.sample_y(candidate_X, n_samples=1)
            sample = sample[0, :, 0] if sample.ndim == 3 else sample.flatten()
            scalarized += weights[i] * sample
        
        # Select best candidates (one per available license)
        top_indices = np.argsort(scalarized)[:len(licenses)]
        top_candidates_t = candidate_X_t[top_indices].to(device)
        top_candidates = top_candidates_t.cpu().numpy()
        
        with ProcessPoolExecutor(max_workers=len(licenses)) as executor:
            inputs = [
                (top_candidates_t[i].cpu(), license, objective_fn)
                for i, license in zip(range(len(licenses)), itertools.cycle(licenses))
            ]
            batch_results = list(executor.map(evaluate_candidate_wrapper, inputs))
        
        # Process and add batch results to dataset
        for candidate, result in zip(top_candidates, batch_results):
            candidate_2d = np.atleast_2d(candidate)
            result_raw = flatten_to_1d(result[:n_objectives])
            result_normalized = result_raw / ref_pt.cpu().numpy()
            
            X = np.vstack([X, candidate_2d])
            Y = np.vstack([Y, result_normalized])
    
    # Prepare return dataframe
    param_names = [f"param_{i}" for i in range(dim)]
    obj_names = [f"obj_{i}" for i in range(n_objectives)]
    
    # Rescale Y back to original space for return value
    Y_original = Y * ref_pt.cpu().numpy()
    
    data = np.hstack((X, Y_original))
    df = pd.DataFrame(data, columns=param_names + obj_names)
    
    # Calculate Pareto frontier directly instead of using the external function
    is_pareto = np.ones(Y_original.shape[0], dtype=bool)
    for i, y in enumerate(Y_original):
        # A point is not on Pareto frontier if another point dominates it
        if any(np.all(other <= y) and np.any(other < y) for j, other in enumerate(Y_original) if j != i):
            is_pareto[i] = False
    
    # Add Pareto indicator to DataFrame
    df['pareto'] = is_pareto
    
    return df

def run_bayesian_optimization_moucb(
    objective_fn,
    bounds: np.ndarray,
    licenses: list,
    initial_points: int = 10,
    iterations: int = 10,
    n_candidates: int = 200,
    beta: float = 2.0,
) -> pd.DataFrame:
    """
    Multi-objective BO using a UCB-like acquisition (MOUCB),
    with objectives normalized by REF_POINT and GPU acceleration.
    """
    # 1) Setup device, ref_point, and bounds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_point = torch.tensor(REF_POINT, dtype=torch.double, device=device)
    dim = bounds.shape[0]
    bounds_tensor = torch.tensor(bounds.T, dtype=torch.double, device=device)

    # 2) Initial Sobol design in [0,1]^d → scale to [lb, ub]
    sobol = SobolEngine(dimension=dim, scramble=True)
    unit_X = sobol.draw(initial_points).to(dtype=torch.double, device=device)
    X = bounds_tensor[0] + (bounds_tensor[1] - bounds_tensor[0]) * unit_X  # (n0 × d)

    # 3) Evaluate initial batch in parallel on CPU
    with ProcessPoolExecutor(max_workers=min(len(licenses), initial_points)) as exec:
        inputs = [(X[i].cpu(), lic, objective_fn)
                  for i, lic in zip(range(initial_points), itertools.cycle(licenses))]
        init_results = list(exec.map(evaluate_candidate_wrapper, inputs))
    
    # Extract results and ensure proper dimensionality
    flat_init = [flatten_to_1d(r) for r in init_results]
    Y = torch.tensor(np.vstack(flat_init), dtype=torch.double, device=device)  # (n0 × m)

    # 4) Normalize objectives using ref_point directly
    Y_norm = Y / ref_point

    # 5) Fit one SingleTaskGP per objective, wrap in ModelListGP
    gps = []
    for j in range(Y_norm.shape[1]):
        # Ensure proper tensor shape with unsqueeze
        y_j = Y_norm[:, j].unsqueeze(-1).contiguous()
        gp = SingleTaskGP(X, y_j)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
        fit_gpytorch_model(mll)
        gps.append(gp.to(device))
    
    model = ModelListGP(*gps).to(device)

    # BO loop
    for it in range(1, iterations + 1):
        print(f"[MOUCB] Iter {it}/{iterations}  X.shape={X.shape}  Y.shape={Y.shape}")

        # 7) Sample a random Dirichlet weight vector
        w = torch.from_numpy(np.random.dirichlet(np.ones(Y_norm.shape[1]))).to(device).double()

        # 8) Generate candidate pool via Sobol with iteration-specific seed for diversity
        sobol_candidates = SobolEngine(dimension=dim, scramble=True, seed=it)
        unit_cand = sobol_candidates.draw(n_candidates).to(dtype=torch.double, device=device)
        candidate_X = bounds_tensor[0] + (bounds_tensor[1] - bounds_tensor[0]) * unit_cand  # (Nc × d)

        # 9) Predictive posterior for each GP
        means = []
        stds  = []
        for gp in model.models:
            post = gp.posterior(candidate_X)
            means.append(post.mean.view(-1))
            stds.append(post.variance.clamp_min(1e-12).sqrt().view(-1))
        means = torch.stack(means, dim=1)  # (Nc × m)
        stds  = torch.stack(stds,  dim=1)  # (Nc × m)

        # 10) Compute weighted LCB: μ - β·σ, then weighted sum
        lcb = means - beta * stds                 # (Nc × m)
        scores = (w * lcb).sum(dim=1)            # (Nc,)

        # 11) Select best `len(licenses)` candidates (smallest score)
        batch_size = min(len(licenses), n_candidates)
        topk = torch.topk(-scores, k=batch_size).indices
        batch_X = candidate_X[topk]              # (batch_size × d)

        # 12) Evaluate selected candidates
        with ProcessPoolExecutor(max_workers=batch_size) as exec:
            inputs = [(batch_X[i].cpu(), lic, objective_fn)
                      for i, lic in zip(range(batch_size), itertools.cycle(licenses))]
            batch_results = list(exec.map(evaluate_candidate_wrapper, inputs))
        
        # Extract results with flatten_to_1d to ensure consistent dimensions
        flat_batch = [flatten_to_1d(r) for r in batch_results]
        new_Y = torch.tensor(np.vstack(flat_batch), dtype=torch.double, device=device)

        # 13) Append and re-normalize
        X = torch.cat([X, batch_X], dim=0)
        Y = torch.cat([Y, new_Y], dim=0)
        Y_norm = Y / ref_point

        # 14) Create new GP models with updated data for stability
        gps = []
        for j in range(Y_norm.shape[1]):
            # Ensure proper tensor shape with unsqueeze
            y_j = Y_norm[:, j].unsqueeze(-1).contiguous()
            gp = SingleTaskGP(X, y_j)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(device)
            fit_gpytorch_model(mll)
            gps.append(gp.to(device))
        
        model = ModelListGP(*gps).to(device)

    # 15) Return DataFrame of raw (un-normalized) results + Pareto flag
    X_cpu = X.cpu().numpy()
    Y_cpu = Y.cpu().numpy()
    df = pd.DataFrame(
        np.hstack([X_cpu, Y_cpu]),
        columns=[f"param_{i}" for i in range(dim)]
               +[f"obj_{j}"   for j in range(Y_cpu.shape[1])]
    )
    
    # Calculate Pareto frontier directly instead of using the external function
    is_pareto = np.ones(Y_cpu.shape[0], dtype=bool)
    for i, y in enumerate(Y_cpu):
        # A point is not on Pareto frontier if another point dominates it
        if any(np.all(other <= y) and np.any(other < y) for j, other in enumerate(Y_cpu) if j != i):
            is_pareto[i] = False
    
    # Add Pareto indicator to DataFrame
    df["pareto"] = is_pareto
    
    return df

def run_bayesian_optimization(
    objective_fn,
    bounds,
    licenses,
    ref_point_values=None, # Add parameter, default to None or fetch from config
    initial_points=10,
    iterations=10,
    # n_candidates=1000, # Remove, not used by EHVI directly
    n_objectives=8,
    outer_n_samples=256, # Rename n_samples for clarity/consistency
    num_restarts=10,     # Add parameter for EHVI
    raw_samples=512,     # Add parameter for EHVI
    aq_function=None,
):
    """
    Wrapper that selects the BO method based on the acquisition function.

    Parameters:
      - objective_fn: function to evaluate candidates
      - bounds: array of parameter bounds
      - licenses: list of license dicts
      - ref_point_values: tuple/list of reference point values for hypervolume.
                          If None, uses REF_POINT from config.settings.
      - initial_points: number of random initial evaluations
      - iterations: number of BO iterations
      - n_objectives: number of objectives
      - outer_n_samples: MC samples for qNEHVI acquisition function.
      - num_restarts: Restarts for optimizing the EHVI acquisition function.
      - raw_samples: Candidate samples for optimizing the EHVI acquisition function.
      - aq_function: string flag ('MOBO', 'EHVI', 'MOUCB', 'RANDOM')
    """
    # Lock in the acquisition function once:
    if aq_function is None:
        from config.settings import AQ_FUNCTION
        aq_function = AQ_FUNCTION

    # Handle default reference point
    if ref_point_values is None:
        # from config.settings import REF_POINT # Already imported above
        print(f"Using default REF_POINT from config: {REF_POINT}")
        ref_point_values = REF_POINT
    elif not isinstance(ref_point_values, (list, tuple)):
         raise TypeError(f"ref_point_values must be a tuple or list, got {type(ref_point_values)}")


    acq = aq_function.upper()
    print("Acquisition function locked to:", acq)

    if acq == 'EHVI':
        print("Using Expected Hypervolume Improvement (EHVI).")
        # Pass all necessary arguments to ehvi function
        return run_bayesian_optimization_ehvi(
            objective_fn=objective_fn,
            bounds=bounds,
            ref_point_values=ref_point_values, # Pass the value
            licenses=licenses,
            initial_points=initial_points,
            iterations=iterations,
            outer_n_samples=outer_n_samples, # Pass correctly named arg
            num_restarts=num_restarts,       # Pass new arg
            raw_samples=raw_samples,         # Pass new arg
            n_objectives=n_objectives,
            # n_candidates is removed
        )

    elif acq == 'MOBO':
        print("Using MOBO (random scalarization).")
        # MOBO doesn't need ref_point_values directly, but uses REF_POINT internally
        # Ensure run_bayesian_optimization_mobo handles REF_POINT correctly if needed
        return run_bayesian_optimization_mobo(
            objective_fn=objective_fn,
            bounds=bounds,
            licenses=licenses,
            initial_points=initial_points,
            iterations=iterations,
            n_candidates=raw_samples, # MOBO *does* use n_candidates, maybe map raw_samples?
            n_objectives=n_objectives,
        )

    elif acq in ('MOUCB', 'MO-UCB'):
        print("Using Multi-Objective UCB (MOUCB).")
        # MOUCB also uses REF_POINT internally for normalization
        return run_bayesian_optimization_moucb(
            objective_fn=objective_fn,
            bounds=bounds,
            licenses=licenses,
            initial_points=initial_points,
            iterations=iterations,
            n_candidates=raw_samples, # MOUCB uses n_candidates, map raw_samples?
            beta=2.0,
            # Consider passing ref_point_values explicitly if MOUCB is refactored later
        )

    elif acq == 'RANDOM':
        print("Using Random Baseline.")
        # Random doesn't need ref_point_values
        return run_random_baseline(
            objective_fn=objective_fn,
            bounds=bounds,
            licenses=licenses,
            initial_points=initial_points,
            iterations=iterations,
            n_candidates=raw_samples, # Random uses n_candidates, map raw_samples?
            n_objectives=n_objectives,
        )

    else:
        raise ValueError(f"Unknown acquisition function: {aq_function}")