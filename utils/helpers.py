import numpy as np
import torch
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import itertools
from concurrent.futures import ProcessPoolExecutor
import torch
import numpy as np
import numpy as np
import pygmo as pg
from config.settings import REF_POINT, INDEX_WEIGHTS

def compute_hypervolume(front: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Compute the normalized hypervolume dominated by the Pareto front relative to a reference point,
    using PyGMO's exact hypervolume computation. Handles points outside the reference point.

    Parameters:
        front (np.ndarray): Pareto frontier array, shape (n_points, n_objectives).
        ref_point (np.ndarray): 1D array of length n_objectives representing the worst-case bounds.

    Returns:
        float: Normalized hypervolume (in [0, 1]).
    """
    front = np.asarray(front, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float).flatten()

    if front.size == 0:
        return 0.0

    if front.ndim != 2 or ref_point.ndim != 1:
        raise ValueError(f"Expected front 2D, ref_point 1D, got shapes {front.shape}, {ref_point.shape}")
    if front.shape[1] != ref_point.shape[0]:
        raise ValueError(f"Mismatch: front obj {front.shape[1]}, ref_point obj {ref_point.shape[0]}")

    # --- FILTERING STEP ---
    # Keep only points P where P < ref_point element-wise (strictly better)
    dominated_by_ref = np.all(front < ref_point, axis=1)
    filtered_front = front[dominated_by_ref]

    if filtered_front.size == 0:
         print("Warning: All Pareto points were worse than or equal to the reference point in at least one dimension. Hypervolume is 0.")
         return 0.0
    # --- END FILTERING ---

    # Scale the *filtered* front to [0,1] by dividing each objective by its corresponding ref_point dimension
    scaled_front = filtered_front / ref_point
    scaled_ref = np.ones_like(ref_point)  # Reference point becomes [1,1,...,1]

    # Check for numerical issues after scaling (should be < 1)
    if np.any(scaled_front >= 1.0):
         # This might happen due to floating point precision if a point was *very* close to the ref_point
         print("Warning: Scaled front contains values >= 1.0 after filtering. Clamping.")
         scaled_front = np.minimum(scaled_front, 1.0 - 1e-9) # Clamp slightly below 1

    # Compute hypervolume using pygmo
    try:
        hv = pg.hypervolume(scaled_front) # Pass list of lists or ndarray
        # Pygmo versions might differ slightly in API, ensure list format if needed
        # raw_hv = hv.compute(scaled_ref.tolist())
        raw_hv = hv.compute(scaled_ref) # Newer pygmo might accept ndarray
    except ValueError as e:
         print(f"Error during pygmo hypervolume computation on scaled data:\n{e}")
         print("Scaled Front:\n", scaled_front)
         print("Scaled Ref:\n", scaled_ref)
         # As a fallback, return 0 or re-raise, depending on desired behavior
         return 0.0 # Or raise e

    # The total possible volume in scaled space is 1.0 (product of scaled_ref dimensions)
    # Since scaled_ref is all 1s, the total volume is 1.
    return float(raw_hv)

def compute_pareto_frontier(Y):
    """
    Compute the Pareto frontier (for minimization) from an array of objective values.
    
    A point is Pareto efficient if no other point dominates it.
    Here, point j dominates point i if:
        Y[j] <= Y[i] (elementwise)  and  Y[j] < Y[i] for at least one objective.
    
    Parameters:
        Y (np.ndarray): Array of shape (n_points, n_objectives).
        
    Returns:
        np.ndarray: The subset of Y that forms the Pareto frontier.
    """
    Y = np.asarray(Y)
    n_points = Y.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        # Only check points that haven't already been dominated.
        if is_efficient[i]:
            for j in range(n_points):
                if i == j:
                    continue
                if is_efficient[j]:
                    # If j dominates i, mark i as inefficient.
                    if np.all(Y[j] <= Y[i]) and np.any(Y[j] < Y[i]):
                        is_efficient[i] = False
                        break
    return Y[is_efficient]


def compute_pareto_frontier_mask(Y: np.ndarray) -> np.ndarray:
    """
    Computes the Pareto frontier mask for a set of points (minimization).

    A point is Pareto efficient if no other point dominates it. Point `j`
    dominates point `i` if `Y[j] <= Y[i]` element-wise and `Y[j] < Y[i]`
    for at least one objective.

    Args:
        Y: A NumPy array of objective values with shape (n_points, n_objectives).

    Returns:
        A boolean NumPy array of shape (n_points,) where `True` indicates
        that the corresponding point is on the Pareto frontier (is efficient).
    """
    num_points = Y.shape[0]
    if num_points == 0:
        return np.array([], dtype=bool)

    # Initialize all points as potentially efficient
    is_efficient = np.ones(num_points, dtype=bool)

    for i in range(num_points):
        if not is_efficient[i]:
            # If point i is already marked as dominated, skip it
            continue

        # Find points 'other' that dominate point 'i'
        # other <= Y[i] elementwise AND other < Y[i] for at least one objective
        dominates_i_mask = (
            np.all(Y <= Y[i], axis=1) &  # All objectives <=
            np.any(Y < Y[i], axis=1)     # At least one objective <
        )
        # If any point in dominates_i_mask (other than i itself) is True, then i is dominated
        if np.any(dominates_i_mask):
             is_efficient[i] = False


    return is_efficient
def simple_sum_utility(Y: torch.Tensor) -> torch.Tensor:
    """Simple utility function: negative sum (minimization)."""
    return -Y.sum(dim=-1)

def double_utility(Y: torch.Tensor) -> torch.Tensor:
    """
    Utility = –( first_metric + second_to_last_metric )
    i.e. only metrics 0 and -2 participate (equal weight).
    """
    return -(Y[..., 0] + Y[..., -2])

def neg_l1_utility(Y: torch.Tensor) -> torch.Tensor:
    """Negative L1 distance from the point (0.5, ..., 0.5)"""
    target = torch.full_like(Y, 0.5)
    return -torch.norm(Y - target, p=1, dim=-1)
import torch
from config.settings import REF_POINT



# 2) Pre‑build a weight tensor (same length as REF_POINT)
_ref = torch.tensor(REF_POINT, dtype=torch.float32)
_weights = torch.zeros_like(_ref)
print('INDEX_WEIGHTS : ' ,INDEX_WEIGHTS )
print('_weights : ' , _weights )
for idx, w in INDEX_WEIGHTS.items():
    _weights[idx] = w


def custom_utility(Y: torch.Tensor) -> torch.Tensor:
    """
    Custom utility over 8 composite objectives.

    Can accept either 14 raw metrics (which it reduces to 8) or
    the 8 pre-computed composite metrics directly.

    Args:
        Y: Tensor of objective values. Shape (..., 14) or (..., 8).

    Returns:
        Tensor of scalar utility values. Shape (...).
    """
    input_dim = Y.shape[-1]

    if input_dim == 14:
        # --- Case 1: Input has 14 raw metrics ---
        print("[custom_utility] Input has 14 dims, reducing to 8.")
        # Combine raw metrics to 12 intermediate values
        # (Indices correspond to the description in your original docstring)
        combined = torch.stack([
            Y[..., 0],                      # [0] conflicts
            Y[..., 1],                      # [1] quints
            Y[..., 2],                      # [2] quads
            Y[..., 3],                      # [3] four in five slots
            Y[..., 4] + Y[..., 5],          # [4] triple in 24h + same day
            Y[..., 6],                      # [5] three in four slots
            Y[..., 7] + Y[..., 8],          # [6] all b2bs
            Y[..., 9],                      # [7] two in three slots
            Y[..., 10],                     # [8] singular late exam
            Y[..., 11],                     # [9] two exams, large gap
            Y[..., 12],                     # [10] avg_max
            Y[..., 13],                     # [11] lateness
        ], dim=-1)

        # Slice the specific 8 metrics used for the utility calculation
        # Ensure these indices match the order expected by REF_POINT and INDEX_WEIGHTS
        reduced_Y = torch.stack([
            combined[..., 4],  # triple (combined index 4)
            combined[..., 5],  # three in four (combined index 5)
            combined[..., 6],  # b2b (combined index 6)
            combined[..., 7],  # 2 in 3 (combined index 7)
            combined[..., 8],  # singular late (combined index 8)
            combined[..., 9],  # 2 exams large gap (combined index 9)
            combined[...,10],  # avg_max (combined index 10)
            combined[...,11],  # lateness (combined index 11)
        ], dim=-1)
        print(f"[custom_utility] Reduced Y shape: {reduced_Y.shape}")

    elif input_dim == 8:
        # --- Case 2: Input already has 8 composite metrics ---
        print("[custom_utility] Input has 8 dims, using directly.")
        reduced_Y = Y # Assume the input is already the correct 8 metrics

    else:
        # --- Case 3: Unexpected input dimension ---
        raise ValueError(f"Expected 14 or 8 metrics in Y, but got {input_dim}")

    # --- Apply weights and normalization (common logic) ---
    # Ensure shapes match (_ref and _weights should be length 8)
    if reduced_Y.shape[-1] != 8:
         # This should ideally not happen if the logic above is correct
         raise RuntimeError(f"Internal error: reduced_Y dimension is {reduced_Y.shape[-1]}, expected 8.")

    ref_t = _ref.to(device=Y.device, dtype=Y.dtype)     # shape (8,)
    wts_t = _weights.to(device=Y.device, dtype=Y.dtype) # shape (8,)

    # Normalize by reference point (element-wise division)
    # Add small epsilon to avoid division by zero if ref_point has zeros
    normalized = reduced_Y / (ref_t + 1e-9)       # shape (..., 8)

    # Apply weights (element-wise multiplication)
    weighted = normalized * wts_t                 # shape (..., 8)

    # Sum weighted, normalized objectives and negate (higher utility is better)
    utility_val = -weighted.sum(dim=-1)           # shape (...)
    print(f"[custom_utility] Output utility shape: {utility_val.shape}")

    return utility_val