import glob
import os
from pathlib import Path
import multiprocessing as mp

import pandas as pd

from config.settings import LICENSES, SAVE_PATH, OPTIM_MODE, REF_POINT, ASSIGNMENT_TYPE
from models.sequencing import sequencing
from block_assignment.run_blocks import run_multi_blocks
from block_assignment.layercake import run_layer_cake
from block_assignment.block_assignment_IP_min_conflict import run_min_IP
from optimization.CArBO import run_cost_aware_optimization
from optimization.random_draw import RandomBaselineOptimizer 
from post_processing.post_processing import run_pp



def save_results(results, optim_mode, filename="opt_summary.csv"):
    Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

    if isinstance(results, tuple):
        samples_df, metrics_df = results
        samples_df.to_csv(Path(SAVE_PATH) / f"{optim_mode}_samples_{filename}", index=False)
        metrics_df.to_csv(Path(SAVE_PATH) / f"{optim_mode}_metrics_{filename}", index=False)
    elif isinstance(results, pd.DataFrame):
        results.to_csv(Path(SAVE_PATH) / f"{optim_mode}_{filename}", index=False)
    else:
        print(f"Warning: Unexpected result type ({type(results)}) for saving.")


if __name__ == "__main__":
    # Ensure consistent multiprocessing start method
    print('IN MAIN')
    try:
        mp.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"Warning: Could not set start method 'spawn': {e}. Using default.")

    # Diagnostics
    print("OPTIM_MODE selected:", OPTIM_MODE)
    print("REF_POINT:", REF_POINT)

    blocks_dir =  SAVE_PATH + "/blocks/"
    good_schedules = [
        f for f in glob.glob(os.path.join(blocks_dir, "*size_cutoff*.csv"))
        if not os.path.basename(f).startswith("bad_")
    ]

    # If no existing "good" schedule files, generate them
    if not good_schedules: # CHANGE THIS JUST OFR ONE EXPERIMENT 
        print("RUN_MULTI_BLOCKS")
        run_multi_blocks(
            overlap=0.67,
            k=20000,
            tradeoff=(1000, 0.5),
            timelimit = 1 * 3600,           # 2 hours
            size_cutoffs=[ 300],
            k_values=[20000],
            frontloading_values=[2,  5],
            num_blocks_values=[22, 23],
        )
    
    assignment = run_min_IP
    if ASSIGNMENT_TYPE == 'cake' :
        assignment = run_layer_cake


    # Run Cost-Aware Bayesian Optimization
    if OPTIM_MODE == 'carbo': 
        print('CARBO')
        results = run_cost_aware_optimization(
            objective_fn=sequencing,
            post_processing = run_pp, 
            block_assignment= assignment ,
            blocks_dir=blocks_dir,
            total_budget=5 * 48 * 3600,  
            cost_ratio=15,             # block assignments 20x more expensive
            n_iterations=400,
            licenses=LICENSES
        )
    else: 
        print("RANDOM")
        m = RandomBaselineOptimizer(
        # exactly the same args CArBO would get:
        objective_fn=sequencing,
        post_processing=run_pp,
        block_assignment=assignment,
        blocks_dir=blocks_dir,
        total_budget=5 * 48 * 3600,   # e.g. 5 days in seconds
        cost_ratio=3,                # block assignments 3× more expensive
        licenses=LICENSES,
        # then your random‐baseline–specific arg:
        max_workers=5)
        results = m.run_random_baseline(n_iterations=400)

    # Save and report
    save_results(results, OPTIM_MODE, filename=f"{OPTIM_MODE}_results.csv")
    print(f"Saved optimization results for mode: {OPTIM_MODE}")



