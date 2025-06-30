import os
import sys
import glob
import re

import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import sample

from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.acquisition.objective import LearnedObjective
from botorch.sampling import SobolQMCNormalSampler
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_model
# Determine project root for local imports
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.sequencing import sequencing
from post_processing.post_processing import run_pp
from block_assignment.layercake import run_layer_cake
from config.settings import LICENSES, SAVE_PATH, PARAM_NAMES
from optimization.CArBO import CostAwareSchedulingOptimizer


#----------------------------------------------------------------
# Helpers
#----------------------------------------------------------------
# Regex for parsing tensor strings
tensor_re = re.compile(r"tensor\(\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
def parse_tensor_str(x):
    """Convert 'tensor(...)' strings to floats"""
    try:
        if isinstance(x, str) and x.startswith("tensor"):
            m = tensor_re.search(x)
            if m:
                return float(m.group(1))
        return x
    except:
        return 0.0

#----------------------------------------------------------------
# Preference data utilities
#----------------------------------------------------------------
def validate_preference_data(pref_X, pref_comp):
    """Ensure no NaNs/Infs and valid comparison indices"""
    if torch.isnan(pref_X).any() or torch.isinf(pref_X).any():
        print("[Error] NaN/Inf values in pref_X")
        return False
    if (pref_comp >= pref_X.size(0)).any() or (pref_comp < 0).any():
        print("[Error] Invalid indices in pref_comp")
        return False
    if pref_comp.size(0) < 3:
        print("[Warning] Very few preference pairs - model may be unstable")
        return False
    return True


def get_user_preferences(results, X=None, tol=1e-8, max_pairs=25):
    """Generate up to max_pairs preference comparisons based on score differences"""
    scores = [y.sum().item() for y, _ in results]
    q = len(scores)
    potential = []  # (pair, label, diff)
    for i in range(q):
        for j in range(i+1, q):
            # skip identical inputs
            if X is not None and torch.allclose(X[i], X[j], atol=tol):
                continue
            diff = scores[i] - scores[j]
            if abs(diff) < tol:
                continue  # skip ties
            label = 0 if diff >= 0 else 1
            potential.append(((i, j), label, abs(diff)))
    # prioritize largest differences
    potential.sort(key=lambda x: x[2], reverse=True)
    selected = potential[:max_pairs]
    prefs = {pair: label for (pair, label, _) in selected}
    print(f"Generated {len(prefs)} preferences from {len(potential)} potential pairs")
    return prefs


def limit_existing_preferences(pref_X, pref_comp, max_pairs=80):
    """Downsample existing preference pairs to max_pairs"""
    if pref_comp.size(0) <= max_pairs:
        return pref_X, pref_comp
    print(f"Reducing preference pairs from {pref_comp.size(0)} to {max_pairs}")
    device = pref_X.device
    idx = torch.randperm(pref_comp.size(0), device=device)[:max_pairs]
    sel_comp = pref_comp[idx]
    used = torch.unique(sel_comp.flatten())
    mapping = {old.item(): new for new, old in enumerate(used)}
    sel_X = pref_X[used]
    remap = torch.zeros_like(sel_comp, device=device)
    for i in range(sel_comp.size(0)):
        for j in range(2):
            remap[i, j] = mapping[int(sel_comp[i, j].item())]
    return sel_X, remap


def check_for_duplicate_rows(X: torch.Tensor, name="X"):
    """Warn if any duplicate rows in X"""
    uniques, counts = torch.unique(X, dim=0, return_counts=True)
    dup = counts > 1
    if dup.any():
        rows = uniques[dup]
        print(f"[Warning] Found {rows.size(0)} duplicate rows in {name}")
    else:
        print(f"No duplicates in {name}.")


def find_degenerate_pairs(pref_X: torch.Tensor, pref_comp: torch.LongTensor):
    """Return list of indices where compared points are identical"""
    deg = []
    for k, (i, j) in enumerate(pref_comp.tolist()):
        if torch.allclose(pref_X[i], pref_X[j], atol=1e-8):
            deg.append((k, i, j))
    return deg

#----------------------------------------------------------------
# Main optimization loop
#----------------------------------------------------------------
def run_preference_ei_parallel(
    ahh,
    objective_fn,
    post_processing_fn,
    block_assignment_fn,
    licenses,
    blocks_dir,
    X_init,
    Y_init,
    total_budget=3600,
    cost_ratio=3,
    n_iterations=50,
    q=4,
    candidate_pool_size=100,
    num_fantasies=64,
    seed=0,
):
    torch.manual_seed(seed)
    device = X_init.device
    optimizer = ahh(
        blocks_dir=blocks_dir,
        total_budget=total_budget,
        cost_ratio=cost_ratio,
        objective_fn=objective_fn,
        block_assignment=block_assignment_fn,
        post_processing=post_processing_fn,
        licenses=licenses,
    )
    lb, ub = optimizer.bounds[:, 0].to(device), optimizer.bounds[:, 1].to(device)
    dim = optimizer.bounds.size(0)

    X, Y = X_init.to(device), Y_init.to(device)
    utilities = []     
    # Initial preference setup
    print("-> Creating initial preference data...")
    initial = [(Y[i], None) for i in range(Y.size(0))]
    prefs = get_user_preferences(initial, X=X, max_pairs=25)
    if not prefs:
        # fallback minimal pair
        scores = [y.sum().item() for y in initial]
        best, worst = max(range(len(scores)), key=lambda i: scores[i]), min(range(len(scores)), key=lambda i: scores[i])
        prefs = {(best, worst): 0}
    pref_X, pref_comp = [], []
    for (i, j), label in prefs.items():
        pref_X.extend([X[i].unsqueeze(0), X[j].unsqueeze(0)])
        pair = [2*len(pref_comp), 2*len(pref_comp)+1]
        if label == 1:
            pair.reverse()
        pref_comp.append(pair)
    pref_X = torch.cat(pref_X, dim=0)
    pref_comp = torch.tensor(pref_comp, dtype=torch.long, device=device)
    pref_X += 1e-4 * torch.randn_like(pref_X)

    check_for_duplicate_rows(X)
    check_for_duplicate_rows(pref_X, name="pref_X (initial)")
    validate_preference_data(pref_X, pref_comp)
    deg = find_degenerate_pairs(pref_X, pref_comp)
    if deg:
        print(f"[Error] {len(deg)} degenerate comparisons")
    else:
        print("No degenerate comparisons.")

    # Iterations
    for it in range(n_iterations):
        print(f"\n=== Iteration {it+1}/{n_iterations} ===")
        optimizer.X, optimizer.Y = X, Y
        if pref_comp.size(0) > 80:
            pref_X, pref_comp = limit_existing_preferences(pref_X, pref_comp)
            pref_X += 1e-4 * torch.randn_like(pref_X)

        # Fit preference GP
        cpu_X, cpu_comp = pref_X.cpu(), pref_comp.cpu()
        model = PairwiseGP(cpu_X, cpu_comp, input_transform=Normalize(d=dim)).to(device)
        model.train()
        mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model).to(device)
        fit_gpytorch_model(mll)
        u_obj = LearnedObjective(model)
        # Acquisition
        sampler = SobolQMCNormalSampler(num_fantasies)
        acqf = qExpectedImprovement(model=model, best_f=0.0, sampler=sampler, objective=u_obj).to(device)

        # Candidate sampling
        blocks = list(optimizer.available_configs)
        chosen = sample(blocks, min(len(blocks), candidate_pool_size))
        ei_vals = []
        for idx, cfg in enumerate(chosen):
            cfg_t = torch.tensor(cfg, dtype=torch.double, device=device)
            lb_fix, ub_fix = lb.clone(), ub.clone()
            lb_fix[:cfg_t.numel()] = cfg_t
            ub_fix[:cfg_t.numel()] = cfg_t
            cand, val = optimize_acqf(acq_function=acqf,bounds=torch.stack([lb_fix, ub_fix]),q=1,num_restarts=5,raw_samples=20)
            ei_vals.append((cand.squeeze(0), val.item()))
            if idx % 10 == 0 or idx == len(chosen)-1:
                print(f"   Evaluated EI on block {idx+1}/{len(chosen)} (acq={val:.4f})")
        ei_vals.sort(key=lambda x: x[1], reverse=True)
        candidates = torch.stack([c for c, _ in ei_vals[:q]], dim=0)

        # Handle NaNs
        nan_mask = torch.isnan(candidates).any(dim=1)
        if nan_mask.any():
            fallback = optimizer._sample_existing(int(nan_mask.sum()), device=device)
            candidates[nan_mask] = fallback

        # Parallel evaluation
        print("-> Launching parallel evaluations...")
        with ThreadPoolExecutor(max_workers=q) as exec:
            futures = {exec.submit(optimizer.evaluate_parameters, candidates[i]): i for i in range(candidates.size(0))}
            res = []
            for fut in as_completed(futures):
                i = futures[fut]
                y, cost = fut.result()
                res.append((i, y, cost))
                print(f"   Candidate {i} evaluated: y={y.tolist()}, cost={cost:.4f}")
        res.sort(key=lambda x: x[0])
        evals = [(y, c) for _, y, c in res]
        for idx, (y_val, _) in enumerate(evals):
            x_cand = candidates[idx]
            # get posterior on CPU
            #post = model.posterior(x_cand.unsqueeze(0).cpu())
            post = model.posterior(candidates[idx].unsqueeze(0).cpu())
            util = post.mean.squeeze(-1).item()
            utilities.append(util)

        # Append data
        pre_n = X.size(0)
        for x_cand, (y_val, _) in zip(candidates, evals):
            X = torch.cat([X, x_cand.unsqueeze(0)], dim=0)
            Y = torch.cat([Y, y_val.unsqueeze(0)], dim=0) if y_val.dim()==1 else torch.cat([Y, y_val], dim=0)
        print(f"-> Appended {q} new points")

        # New preferences
        new = get_user_preferences(evals, X=candidates, max_pairs=5)
        for (i, j), label in new.items():
            xi, xj = candidates[i].unsqueeze(0), candidates[j].unsqueeze(0)
            pref_X = torch.cat([pref_X, xi, xj], dim=0)
            pair = [pref_X.size(0)-2, pref_X.size(0)-1]
            if label == 1:
                pair.reverse()
            pref_comp = torch.cat([pref_comp, torch.tensor([pair], device=device)], dim=0)
        if new:
            pref_X += 1e-4 * torch.randn_like(pref_X)
            check_for_duplicate_rows(pref_X, name=f"pref_X (iter {it+1})")
        print(f"[Iter {it+1}] prefs added: {len(new)}; total pairs: {pref_comp.size(0)}")

    return X, Y, utilities 




