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
from optimization.EIUU import parse_tensor_str, run_preference_ei_parallel
import argparse, json

parser = argparse.ArgumentParser()
parser.add_argument("--prefs",       type=str,   help="path to JSON of (winner,loser) pairs")
parser.add_argument("--n_iterations", type=int,  default=50)
args = parser.parse_args()

# load prefs if provided
if args.prefs:
    with open(args.prefs) as f:
        prefs = json.load(f)
else:
    prefs = []
# === Load your initial X_init, Y_init from CSVs ===
csv_paths = glob.glob(os.path.join(SAVE_PATH, "metrics/*.csv"))
if not csv_paths:
    print('csv_paths, ' , csv_paths)
    raise RuntimeError(f"No CSV files found in {SAVE_PATH!r}")
d = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
print('d. ', d)
for col in PARAM_NAMES:
    d[col] = d[col].apply(parse_tensor_str).astype(float)

X_np = d[PARAM_NAMES].to_numpy()
d['list'] = d.apply(lambda row: [
    row["triple in 24h (no gaps)"] + row["triple in same day (no gaps)"],
    row["three in four slots"],
    row["evening/morning b2b"] + row["other b2b"],
    row["two in three slots"],
    row["singular late exam"],
    row["two exams, large gap"],
    row["avg_max"],
    row["lateness"]
], axis=1)
Y_np = np.vstack(d['list'].values)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

X_init = torch.tensor(X_np, dtype=torch.double, device=device)
Y_init = torch.tensor(Y_np, dtype=torch.double, device=device)

print(f"Loaded {X_init.size(0)} initial data points with {X_init.size(1)} parameters")
print(f"Y values have shape {Y_init.shape}")

# === Run preference-based EI-parallel optimization ===
X_final, Y_final , utilities= run_preference_ei_parallel(
    ahh=CostAwareSchedulingOptimizer,
    objective_fn=sequencing,
    post_processing_fn=run_pp,
    block_assignment_fn=run_layer_cake,
    licenses=LICENSES,
    blocks_dir=SAVE_PATH + '/blocks/',
    X_init=X_init,
    Y_init=Y_init,
    n_iterations=50,
    q=4,
    candidate_pool_size=100,
    num_fantasies=64,
    seed=42,
)
#utilities.to_csv(SAVE_PATH + '/EIUUResults.csv')
print(f"\nOptimization complete! Final dataset has {X_final.size(0)} points.")


#param_names = df['parameter'].tolist()
# if you have names for each entry in Y:
metric_names = [
    "triples",
    "three_in_four",
    "back to backs",
    "two_in_three",
    "singular_late",
    "two_large_gap",
    "avg_max",
    "lateness",
]

params_df = pd.DataFrame(X_final.cpu().numpy(), columns=PARAM_NAMES)
metrics_df = pd.DataFrame(Y_final.cpu().numpy(), columns=metric_names)
results = pd.concat([params_df, metrics_df], axis=1)
results["utility"] = utilities

# write to disk
out_path = os.path.join(SAVE_PATH, "results_with_utility.csv")
results.to_csv(out_path, index=False)
print(f"Wrote {len(results)} rows to {out_path}")