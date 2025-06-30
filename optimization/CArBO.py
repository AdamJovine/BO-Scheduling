# Standard library
import gc
import glob
import os
import random
import re
import time
from datetime import datetime
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# System utilities
import psutil

# Third-party
import numpy as np
import pandas as pd
from types import SimpleNamespace

import torch
from torch.quasirandom import SobolEngine
from botorch.models import MultiTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.sampling import MCSampler
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models import MultiTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement
from botorch.models.multitask import KroneckerMultiTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.sampling import MCSampler
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, standardize, unnormalize
from botorch.models.transforms import Standardize, Normalize
from gpytorch.mlls import ExactMarginalLogLikelihood

# Local application imports
from config.settings import SEMESTER, REF_POINT, SAVE_PATH, ASSIGNMENT_TYPE, SEQ_BOUNDS, EMPTY_BLOCKS, BLOCK_BOUNDS,NUM_SLOTS , PP_BOUNDS, POST_PROCESSING, BA_TIME ,SEQ_TIME , PP_TIME ,NORM, load_exam_data , run_name, LICENSES #, comparison
from block_assignment.layercake import run_layer_cake, save_block_assignment
from block_assignment.block_assignment_IP_min_conflict import run_min_IP 
from block_assignment.run_blocks import process_block_assignments
from optimization.Parego import build_parego_ei
from optimization.Thompson import CostCooledTS
from optimization.DirchEIUU import EIUU
from optimization.costModel import MyTorchCostModel 
from block_assignment.helpers import name_block_assignment 
from post_processing.post_processing import run_pp
from metrics.evaluate import evaluate_schedule
from globals.build_global_sets import (
    normalize_and_merge,
    compute_co_enrollment,
    compute_student_unique,
    compute_slot_structures,
    compute_early_slot_penalties
)
from sqlalchemy import create_engine, text

exam_df, exam_sizes = load_exam_data(SEMESTER)
#engine = create_engine("/home/asj53/BOScheduling/results/schedules.db", echo=False)


class CostAwareSchedulingOptimizer: 
    """
    CArBO implementation for two-stage exam scheduling optimization.
    Block assignment is expensive; sequencing is cheap.

    Note: If you see a /bin/bash: libtinfo.so.6 warning,
    consider adjusting your environment's LD_LIBRARY_PATH
    to prioritize system libraries over conda's.
    """
    def __init__(self,
                 blocks_dir = SAVE_PATH + '/blocks/', 
                 total_budget=7200,
                 cost_ratio=3,
                 objective_fn=None,
                 block_assignment = None, 
                 post_processing = None , 
                 licenses=None):
        #self.
        self.blocks_dir = blocks_dir
        self.total_budget = total_budget
        self.cost_ratio = cost_ratio
        self.spent_budget = 0
        self.objective_fn = objective_fn
        self.block_assignment_fn = block_assignment
        self.post_processing = post_processing
        self.licenses = licenses or []
        self.current_licenses = []
        self.X = None
        self.Y = None
        self.costs = None
        self.bad_configs = set()

        # Load blocks and configs
        self.available_blocks = self._get_available_blocks()
        self.available_configs = {(
            b['size_cutoff'], b['frontloading'], b['num_blocks']
        ) for b in self.available_blocks}

        self.setup_parameter_bounds()

    def setup_parameter_bounds(self):
        self.block_params =BLOCK_BOUNDS
        self.seq_params = SEQ_BOUNDS
        self.pp_params = PP_BOUNDS
        self.param_names = list(self.block_params) + list(self.seq_params) + list(self.pp_params)

        bounds = []
        for vals in self.block_params.values():
            bounds.append([float(min(vals)), float(max(vals))])
        for lb, ub in self.seq_params.values():
            bounds.append([lb, ub])
        for lb, ub in self.pp_params.values():
            bounds.append([lb, ub])
        self.bounds = torch.tensor(bounds, dtype=torch.double)
    def _generate_block_assignment(self, block_config, license):
        """
        Run layercake to produce a new block CSV for this triple,
        save it, and register it in available_blocks/configs.
        """

        print("GENERATING BLOCK ASSIGNMENT : " ,block_config )
        size_cutoff, frontloading, num_blocks = block_config
        param_id = name_block_assignment(size_cutoff, frontloading, 20000,num_blocks)
        #param_id = f"size_cutoff{size_cutoff}frontloading{frontloading}k_val20000num_blocks{num_blocks}"
        metrics, ba = self.block_assignment_fn(    param_id,
                size_cutoff,
                frontloading,
                20000,         
                num_blocks,
                license,       
            )
        
        # 3) save to disk
        filepath = self.blocks_dir +param_id + '.csv'
        #print("BACKUP SAVE ")
        #metrics['block_assignment_df'].to_csv(filepath + '.csv')
        # 4) register it for future reuse
        new_block = {
            'size_cutoff':    size_cutoff,
            'frontloading':   frontloading,
            'num_blocks':     num_blocks,
            'path':           filepath 
        }

        print(f"Generated new block file at {filepath}")
        
        return filepath, ba
    def _get_block_info(self, block_config):
        """
        Given a tuple (size_cutoff, frontloading, num_blocks),
        return (is_preloaded, base_cost, block_path).
        """
        is_preloaded = block_config in self.available_configs
        base_cost    = 400 if is_preloaded else self.cost_ratio * 400
        match = next(
            (b for b in self.available_blocks
             if (b['size_cutoff'], b['frontloading'], b['num_blocks']) == block_config),
            None
        )
        # alt method 
        #alt = SAVE_PATH + '/blocks/' + name_block_assignment(block_config[0], block_config[1] , 20000, block_config[2])
        block_path = match['path'] if match else None
        #if not '.csv' in block_path: 
        #    block_path = block_path + '.csv'
        print('_get_block_info block path ', block_path )
        #print('alt ' , alt)
        return is_preloaded, base_cost, block_path
    
    def _get_available_blocks(self):
        print('blocks_dir : ', self.blocks_dir)
        files = glob.glob(os.path.join(self.blocks_dir, '*.csv'))
        good = [f for f in files if not os.path.basename(f).startswith('bad')]
        blocks = []
        for path in good:
            fn = os.path.basename(path)
            m1 = re.search(r'size_cutoff(\d+)', fn)
            m2 = re.search(r'frontloading(\d+)', fn)
            m3 = re.search(r'num_blocks(\d+)', fn)
            if int(m3.group(1))+ len(EMPTY_BLOCKS)<=NUM_SLOTS : 
                if m1 and m2 and m3:
                    blocks.append({
                        'size_cutoff': int(m1.group(1)),
                        'frontloading': int(m2.group(1)),
                        'num_blocks': int(m3.group(1)),
                        'path': path
                    })
            else: 
                print('rejected, ' , path)
        if not blocks:
            print('WARNING: no valid block files found')
        print("AVAIL " , blocks)
        return blocks

    def evaluate_parameters(self, params):
        # 1) Split raw tensor params into block‐params, seq‐params, post‐proc params
        #try:
        bp, sp, pp = self._split_parameters(params)
        block_config = tuple(bp)

        # 2) Find or generate the block assignment CSV (and base_cost)
        is_preloaded, base_cost, block_path = self._get_block_info(block_config)
        lic = self._select_license()

        start = time.time()
        ba = self._fetch_or_generate_block_assignment(
            block_config, is_preloaded, block_path, lic
        )
        if ba is None: # FIX THIS 
            # early‐exit for “bad” block triples
            print('in the ba bad return')
            print(' BAD PARAMS ' , params)
            bad_y, bad_cost = self._make_bad_return(start)
            return bad_y, bad_cost

        # 3) Normalize & merge, then build “global_sets” once
        ba_adj, by_student_block = normalize_and_merge(ba, exam_df, block_config)
        global_sets = self._compute_global_sets(ba_adj, by_student_block, block_config)

        # 4) Build block‐exam dataframe and sizes
        block_exam_df, block_to_size = self._build_block_exam_df(ba_adj, exam_sizes)

        # 5) Unpack seq‐params and build param_dict
        param_dict = self._build_param_dict(sp, block_exam_df, block_to_size)

        # 6) Call the core objective function
        schedule_dict, name = self.objective_fn(
            param_dict, global_sets, lic, block_path
        )
        if schedule_dict is None :#or set(schedule_dict.keys()) != comparison:
            print('in the seq bad return ')
            print(' BAD PARAMS ' , params)
            bad_y, bad_cost = self._make_bad_return(start)
            return bad_y, bad_cost
        # 7) (Optional) Post‐processing step
        #if POST_PROCESSING:
        schedule = self.post_processing(
            lic,
            ba,
            schedule_dict,
            chunk=25,
            size_cutoff=bp[0],
            big_cutoff=sp[6],
            
            pp_params = pp, 
            sched_name=name,
        )
        if schedule is None : 
            print('in the pp bad return :(')
            print(' BAD PARAMS ' , params)
            bad_y, bad_cost = self._make_bad_return(start)
            return bad_y, bad_cost

        # 8) Evaluate the final schedule and save metrics
        metrics = evaluate_schedule(
            schedule,
            exam_sizes,
            params,
            global_sets,
            block_path,
            name, 
        )
       
        # 9) Wrap up “y” and compute cost
        y, cost = self._finalize_return(metrics, start, base_cost)
        return y, cost
        #except: 
    # ──────────────────────────────────────────────────────────────────────
    # Private helpers below
    # ──────────────────────────────────────────────────────────────────────

    def _split_parameters(self, params):
        """
        Splits raw torch.Tensor params into:
        - bp: a list of ints (block‐params),
        - sp: a numpy array of floats (sequence‐params),
        - pp: a numpy array of floats (post‐proc params).
        """
        n_block = len(self.block_params)
        n_seq = len(self.seq_params)
        raw = params.cpu().numpy()

        bp = [int(x) for x in raw[:n_block]]
        sp = raw[n_block : n_block + n_seq]
        pp = raw[n_block + n_seq :]
        print(f"Evaluating parameters → bp: {bp}, sp: {sp}, pp: {pp}")
        return bp, sp, pp

    def _select_license(self):
        """
        Chooses a random unused license from self.licenses (if any),
        ensuring we don’t reuse the same license back‐to_back until all have
        been consumed.
        """
        if not getattr(self, "licenses", []):
            return None

        unused = [lic for lic in self.licenses if lic not in self.current_licenses]
        if not unused:
            self.current_licenses.clear()
            unused = list(self.licenses)

        lic = random.choice(unused)
        self.current_licenses.append(lic)
        print(f"Selected license: {lic}")
        return lic

    def _fetch_or_generate_block_assignment(self, block_config, is_preloaded, block_path, lic):
        """
        If is_preloaded is False: generate a new CSV via _generate_block_assignment
        then call process_block_assignments. If it’s flagged bad, return None.
        Otherwise, return a DataFrame loaded from block_path.
        """
        if is_preloaded:
            
            ba = pd.read_csv(block_path)
            print(f"Loaded pre‐existing block assignment from: {block_path!r}")
            return ba

        print("NOT PRELOADED → Generating new CSV for block_config:", block_config)
        # Generate block assignment CSV (the helper returns path + DataFrame)
        new_path, ba = self._generate_block_assignment(block_config,lic)
        ok = process_block_assignments(new_path)

        if (not ok ):
            # Mark as bad so we never try again
            self.bad_configs.add(block_config)
            print(f"Block {block_config} flagged bad — skipping objective.")
            return None
        size_cutoff, frontloading, num_blocks= block_config
        new_block = {
            'size_cutoff':    size_cutoff,
            'frontloading':   frontloading,
            'num_blocks':     num_blocks,
            'path':           new_path
        }
        self.available_blocks.append(new_block)
        self.available_configs.add(block_config)
        print(f"Generated and validated block assignment: {new_path!r}")
        return ba

    def _make_bad_return(self, start_time):
        """
        If a block triple is “bad,” returns a large‐inf objective and cost.
        """
        return torch.tensor(REF_POINT),time.time() - start_time

    def _compute_global_sets(self, ba_adj, by_student_block, block_config):
        """
        From the adjusted block assignment (ba_adj) and by_student_block,
        compute co‐enrollment, unique block counts, slot structures, early slots, etc.
        Returns a single global_sets dict to pass to the objective.
        """
        print("Building co‐enrollment counts …")
        block_config = str(block_config) # YOU MUST DO THIS 
        pairwise, triple, quadruple = compute_co_enrollment(by_student_block, block_config)

        print("Building student‐unique block counts …")
        student_unique_block, student_unique_block_pairs = compute_student_unique(
            by_student_block, num_slots=NUM_SLOTS, block_config = block_config)

        print("Building slot structures …") 
        slot_structures = compute_slot_structures(list(range(1, NUM_SLOTS + 1)))

        print("Computing early‐slot penalties …")
        first_list = compute_early_slot_penalties(num_slots=NUM_SLOTS)

        global_sets = {
            **slot_structures,
            "first_list": first_list,
            "student_unique_block": student_unique_block,
            "student_unique_block_pairs": student_unique_block_pairs,
            "pairwise": pairwise,
            "triple": triple,
            "quadruple": quadruple,
            "block_assignment": ba_adj,
        }
        return global_sets

    def _build_block_exam_df(self, ba_adj, exam_sizes):
        """
        Creates the merged DataFrame that maps each 'Exam Group' → 'Exam Block',
        then groups to sum sizes per block. Returns (block_exam_df, block_to_size).
        """
        block_exam_df = (
            ba_adj[["Exam Group", "Exam Block"]]
            .merge(exam_sizes, left_on="Exam Group", right_on="exam")
        )
        block_to_size = block_exam_df.groupby("Exam Block")["size"].sum()
        return block_exam_df, block_to_size

    def _build_param_dict(self, sp, block_exam_df, block_to_size):
        """
        From sequence‐params sp, compute all derived fields (big_blocks,
        early_slots, etc.) and return the final param_dict.
        """
        # Unpack sequence‐params in a single tuple:
        (
            alpha,
            gamma,
            delta,
            vega,
            theta,
            large_block_size,
            large_exam_weight,
            large_block_weight,
            large_size_1,
            large_cutoff_freedom,
        ) = tuple(float(x) for x in sp)

        # Identify all “big” blocks
        big_blocks = block_exam_df[block_exam_df["size"] >= large_size_1]["Exam Block"].unique()

        # early slots depend on len(big_blocks), large_cutoff_freedom, and EMPTY_BLOCKS
        early_slots_1 = list(range(1, min(len(big_blocks) + int(large_cutoff_freedom) + 1 + len(EMPTY_BLOCKS), NUM_SLOTS)))
        early_slots_2 = list(range(1, NUM_SLOTS + 1))

        param_dict = {
            "alpha": alpha,
            "beta": alpha,  # original code used alpha for both
            "gamma1": gamma,
            "gamma2": gamma,
            "delta": delta,
            "vega": vega,
            "theta": theta,
            "lambda_large1": large_exam_weight,
            "lambda_large2": large_exam_weight,
            "lambda_big": large_block_weight,
            "early_slots_1": early_slots_1,
            "early_slots_2": early_slots_2,
        }

        # “large_blocks_1”: blocks strictly > large_size_1
        param_dict["large_blocks_1"] = block_to_size[block_to_size > large_size_1].index.tolist()

        # “large_blocks_2”: blocks > large_size_1 AND ≤ large_size_1 (this yields an empty list)
        param_dict["large_blocks_2"] = block_to_size[
            (block_to_size > large_size_1) & (block_to_size <= large_size_1)
        ].index.tolist()

        # All big_blocks for convenience
        param_dict["big_blocks"] = list(big_blocks)

        print("Built param_dict:", param_dict)
        return param_dict


        

    def _finalize_return(self, metrics, start_time, base_cost):
        """
        Builds torch‐tensor y, clips NaNs to a large number, squeezes dims,
        and returns (y, cost) where cost = max(elapsed, base_cost).
        """
        elapsed = time.time() - start_time
        cost = max(elapsed, base_cost)
        result = (metrics["triple in 24h (no gaps)"] + metrics["triple in same day (no gaps)"],
            metrics["three in four slots"],
            metrics["evening/morning b2b"] + metrics["other b2b"],
            metrics["two in three slots"],
            metrics["singular late exam"],
            metrics["two exams, large gap"],
            metrics["avg_max"],
            metrics["lateness"])
        # Convert result into a torch.Tensor and clamp infinities
        y = torch.tensor(result, dtype=torch.double)
        y = torch.nan_to_num(y, nan=1e6, posinf=1e6, neginf=1e6).squeeze(-1)
        y = -y  # original code multiplied by -1

        print(f"Objective value: {result}, elapsed {elapsed:.2f}s, total cost: {cost:.2f}")
        return y, cost

    def predict_costs(self, X):
        """
        Predict evaluation cost: 1.0 if block config preloaded, else cost_ratio,
        reusing the same logic as evaluate_parameters.
        """
        # squeeze out q=1 axis if present
        if X.dim() == 3 and X.size(1) == 1:
            X = X.squeeze(1)  # now shape = [batch, dim]

        n_block = len(self.block_params)
        costs = torch.empty(X.size(0), dtype=torch.double, device=X.device)

        for i, row in enumerate(X):
            # build the same block_config tuple
            block_config = tuple(int(row[j].item()) for j in range(n_block))
            #print('block _config ' , block_config)
            
            _, base_cost, _ = self._get_block_info(block_config)
            #print('base_cost ' , base_cost)
            costs[i] = base_cost
       
        return costs

    def fit_model(self):
        """
        Fit independent GPs with input & output transforms, dropping any
        rows whose Y still contains NaN/Inf.
        """
        # 1) drop any rows with non-finite Y
        mask = torch.isfinite(self.Y).all(dim=-1)
        if not mask.all():
            removed = (~mask).sum().item()
            print(f"⚠️  Dropping {removed} rows with NaN/Inf in Y before GP fit")
            self.X = self.X[mask]
            self.Y = self.Y[mask]

        models = []
        for i in range(self.Y.shape[-1]):
            y_i = self.Y[:, i:i+1]
            # a tiny nugget so no two outputs are exactly identical
            y_i = y_i + 1e-6 * torch.randn_like(y_i)

            gp = SingleTaskGP(
                train_X=self.X,
                train_Y=y_i,
                input_transform=Normalize(d=self.X.size(-1)),
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            models.append(gp)

        return ModelListGP(*models)
    def cost_cooled_mtEHVI(
        self,
        alpha: float,
        mc_samples: int = 8,        # even 4 can work if you’re desperate
        baseline_limit: int = 20,   # max Pareto points to keep
    ):
        """
        Fast cost-cooled qEHVI via a small‐rank MultiTaskGP.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('in cost_cooled_mtEHVI ')
        # 1) training data
        X_train   = self.X.to(device)                              # [n, d]
        Y_obj     = self.Y.to(device)                              # [n, M]
        cost_train = self.predict_costs(X_train).unsqueeze(-1)     # [n, 1]
        Y_all     = torch.cat([Y_obj, cost_train], dim=1)          # [n, M+1]
        print('Y_all ' , Y_all)
        # 2) fit a minimal ICM MultiTaskGP (rank=1)
        gp = KroneckerMultiTaskGP(
            train_X=X_train,                         # [n, d]
            train_Y=Y_all,                           # [n, M+1]  (all objectives + cost)  
            input_transform=Normalize(d=X_train.size(-1)),
            outcome_transform=Standardize(m=Y_all.size(-1)),
        ).to(device)

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()
        print("✓ Fitted KroneckerMultiTaskGP on objectives + cost")
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()
        print("✓ fast MultiTaskGP fitted")

        # 3) tiny Pareto baseline
        mask    = self._pareto_mask(Y_obj.cpu().numpy())
        X_base_all = self.X[mask].to(device)
        if X_base_all.size(0) > baseline_limit:
            idx    = torch.randperm(X_base_all.size(0), device=device)[:baseline_limit]
            X_base = X_base_all[idx]
        else:
            X_base = X_base_all

        # 4) build qNoisyEHVI with QMC (very few samples)
        M       = Y_obj.size(1)
        ref_pt  = torch.tensor(REF_POINT, device=device)
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([mc_samples]), 
            collapse_batch_dims=True,
        )
        ehvi = (
            qNoisyExpectedHypervolumeImprovement(
                model=gp,
                ref_point=ref_pt,
                X_baseline=X_base,
                prune_baseline=False,       # skip expensive pruning
                sampler=sampler,
            )
            .subset_output(tasks=list(range(M)))
            .to(device)
        )
        print(f"✓ built fast qNoisyEHVI (mc_samples={mc_samples})")

        # 5) cost-cooled wrapper
        def cooled(X_norm: torch.Tensor) -> torch.Tensor:
            """
            X_norm: [b,q,d] or [q,d] in normalized space → [b,q] scores
            """
            Xn = X_norm.to(device)
            if Xn.dim() == 2:
                Xn = Xn.unsqueeze(0)   # → [1,q,d]

            with torch.no_grad():
                ehv = ehvi(Xn)                          # [b,q]
                post = gp.posterior(Xn)                # Posterior over M+1 tasks
                cost_mean = post.mean[..., -1]         # [b,q]
                batch_cost = cost_mean.max(dim=-1).values  # [b]
                return ehv / batch_cost.pow(alpha).unsqueeze(-1)

        print("→ returning fast cost-cooled EHVI wrapper")
        return cooled
    def cost_cooled_parEGO(self, alpha: float):
        """
        ParEGO acquisition with optional Chebyshev scalarization on normalized objectives,
        plus cost-cooling on the resulting qEI. Set `NORM=True` to normalize objectives before scalarization.
        Safely handles normalization by including the reference point when sample size is low
        and clamping standard deviation to avoid NaNs.
        """
        # Device selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n--- Starting cost_cooled_parEGO on device {device} (norm={NORM}) ---")
        print(f"Received alpha: {alpha}")

        # 1) Prepare Y (raw or normalized)
        Y_raw  = self.Y.to(device)                              # shape [n, M]
        ref_pt = torch.tensor(REF_POINT, device=device).unsqueeze(0)  # [1, M]

        if False:
            # if too few points, include ref_pt to ensure non-zero variance
            if Y_raw.size(0) < 2:
                Y_stats = torch.cat([Y_raw, ref_pt], dim=0)      # [n+1, M]
            else:
                Y_stats = Y_raw                                 # [n, M]

            # compute mean & std (unbiased=False avoids ddof issues)
            means = Y_stats.mean(dim=0, keepdim=True)          # [1, M]
            stds  = Y_stats.std(dim=0, unbiased=False, keepdim=True)
            stds  = stds.clamp(min=1e-6)                       # avoid zero-division

            # normalize both Y and reference point
            print('means , ' , means )
            print('stds, ' , stds)
            Y_proc   = (Y_raw - means) / stds                  # [n, M]
            print('Y_proc', Y_proc)
            ref_proc = (ref_pt  - means) / stds                # [1, M]
            print(f"Normalized Y shape: {tuple(Y_proc.shape)}")
        else:
            Y_proc   = Y_raw
            ref_proc = ref_pt
            print(f"Using raw Y shape: {tuple(Y_proc.shape)}")

        # 2) Draw a random ParEGO weight vector
        num_obj = Y_proc.size(1)
        weights = torch.rand(num_obj, device=device)
        
        weights = weights / weights.sum()
        print(f"ParEGO weights: {weights.tolist()}")

        # 3) Chebyshev scalarization
        Y_shifted  = Y_proc - ref_proc                      # [n, M]
        
        Y_weighted = weights.unsqueeze(0) * Y_shifted
        print('Y_weighted ' , Y_weighted )
        y_scalar   = Y_weighted.max(dim=1).values.unsqueeze(-1)  # [n, 1]
        print(f"Scalarized Y shape: {tuple(y_scalar.shape)}")

        # 4) Fit GP on (X, y_scalar)
        X_train = self.X.to(device)
        gp = SingleTaskGP(
            train_X=X_train,
            train_Y=y_scalar,
            input_transform=Normalize(d=X_train.size(-1)),
            outcome_transform=Standardize(m=1),
        ).to(device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        print("Single-output GP fitted on scalarized data.")

        # 5) Build q-Expected Improvement acquisition
        best_f  = y_scalar.max().item()
        acq_ei  = qExpectedImprovement(model=gp, best_f=best_f).to(device)
        print("qExpectedImprovement acquisition built.")

        # 6) Return cost-cooled acquisition function
        def cooled(X_norm: torch.Tensor) -> torch.Tensor:
            """
            Wrapper around qEI that applies cost-cooling: EI / cost^alpha.
            Expects X_norm in [b, q, d] or [q, d].
            """
            print("\n--- Enter cooled ParEGO acquisition wrapper ---")
            X_norm = X_norm.to(device)
            print(f"Input X_norm shape: {tuple(X_norm.shape)}")

            # ensure [b, q, d]
            if X_norm.dim() == 2:
                X_norm = X_norm.unsqueeze(0)

            # 6a) Evaluate qEI
            print("Evaluating qEI...")
            ei_vals = acq_ei(X_norm)                           # [b, q]
            print(f"Raw EI values shape: {tuple(ei_vals.shape)}")
            print('ei , ' , ei_vals)
            # 6b) Unnormalize to real X
            b, q, d = X_norm.shape
            bounds  = self.bounds.to(device)
            X_unn   = unnormalize(X_norm.view(-1, d), bounds.T).to(device)
            print(f"Unnormalized X shape: {tuple(X_unn.shape)}")
            print('X_unn : ' , X_unn)
            # 6c) Predict costs and compute batch cost
            print("Predicting costs on GPU…")
            costs      = self.predict_costs(X_unn).view(b, q)  # [b, q]
            batch_cost = costs.max(dim=1).values               # [b]
            print(f"Predicted costs shape: {tuple(costs.shape)}")
            print('costs , ' ,  costs)
            # 6d) Apply cost-cooling: EI / cost^alpha
            #cooled_vals = ei_vals / batch_cost.pow(alpha).unsqueeze(-1)
            cooled_vals = ei_vals / batch_cost.pow(alpha)   # both (b,)

            print(f"Cooled EI shape: {tuple(cooled_vals.shape)}")
            print('COOLED , VALS ' , cooled_vals )
            print("--- Exit cooled ParEGO wrapper ---")
            return cooled_vals

        print("--- Finished building cost_cooled_parEGO wrapper ---")
        return cooled



    def cost_cooled_ehvi(self, alpha: float):

        # 0) Select device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n--- Starting cost_cooled_ehvi on device {device} ---")
        print(f"Received alpha: {alpha}")
        print(f"Using REF_POINT: {REF_POINT}")

        # 1) Fit GP and move to device
        model = self.fit_model().to(device)
        model.likelihood.to(device)
        print("GP model fitted and moved to device.")

        # 2) Pareto‐mask and filter bad configs
        Y = self.Y.to(device)
        mask = self._pareto_mask(Y.cpu().numpy())
        print(f"Total evaluations: {Y.size(0)}")
        print(f"Pareto mask keeps {mask.sum()} points (before removing bad configs)")

        n_block = len(self.block_params)
        good = []
        for i, keep in enumerate(mask):
            if not keep:
                continue
            bc = tuple(int(v.item()) for v in self.X[i, :n_block])
            if bc in self.bad_configs:
                print(f"  Skipping bad config at index {i}: {bc}")
                continue
            good.append((self.X[i], self.Y[i]))

        if not good:
            raise RuntimeError("No valid Pareto points to build baseline!")

        X_base, Y_base = zip(*good)
        X_base = torch.stack(X_base).double().to(device)
        Y_base = torch.stack(Y_base).double().to(device)
        print(f"Collected {X_base.size(0)} good baseline points")

        # 3) Deduplicate
        seen = set()
        unique = []
        for x_row, y_row in zip(X_base, Y_base):
            key = tuple(x_row.tolist())
            if key not in seen:
                seen.add(key)
                unique.append((x_row, y_row))
        X_base, Y_base = zip(*unique)
        X_base = torch.stack(X_base).to(device)
        Y_base = torch.stack(Y_base).to(device)
        print(f"Deduplicated → {X_base.size(0)} baseline points")

        # 4) Subsample to at most 50
        if X_base.size(0) > 50:
            idx = torch.randperm(X_base.size(0), device=device)[:50]
            X_base = X_base[idx]
            Y_base = Y_base[idx]
            print(f"Subsampled baseline to 50 points")

        # 5) Jitter to break collinearity
        X_base_jittered = X_base + 1e-1 * torch.randn_like(X_base, device=device)
        print(f"Jittered baseline → shape {X_base_jittered.shape}")

        # 6) Build streamlined qNEHVI on GPU
        print("Building qNoisyExpectedHypervolumeImprovement acquisition function...")
        sampler = MCSampler(sample_shape=torch.Size([16]))  # fewer MC samples for speed
        acq = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=REF_POINT,
            X_baseline=X_base_jittered,
            prune_baseline=True,
            sampler=sampler,
        ).to(device)
        print("Acquisition function built successfully.")

        # 7) Return cooled wrapper
        def cooled(X_norm: torch.Tensor) -> torch.Tensor:
            print("\n--- Enter cooled acquisition wrapper ---")
            X_norm = X_norm.to(device)
            print(f"Input X_norm shape: {tuple(X_norm.shape)}")

            if X_norm.dim() == 2:
                X_norm = X_norm.unsqueeze(0)  # shape [b=1, q, d]

            # evaluate acquisition
            print("Evaluating qNEHVI...")
            vals = acq(X_norm)  # [b, q]
            print(f"Raw acquisition values shape: {tuple(vals.shape)}")

            # unnormalize to get real parameters
            b, q, d = X_norm.shape
            bounds = self.bounds.to(device=device)
            X_unn = unnormalize(X_norm.view(-1, d), bounds.T).to(device)
            print(f"Unnormalized X shape: {tuple(X_unn.shape)}")

            # predict costs
            print("Predicting costs on GPU…")
            costs = self.predict_costs(X_unn).view(b, q).to(device)
            print(f"Predicted costs shape: {tuple(costs.shape)}")

            # cooled score
            batch_cost = costs.max(dim=1).values  # [b]
            out = vals / batch_cost.pow(alpha).unsqueeze(-1)
            print(f"Final cooled scores shape: {tuple(out.shape)}")
            print("--- Exit cooled acquisition wrapper ---")

            return out

        print("--- Finished building cost_cooled_ehvi wrapper ---")
        return cooled
    def _sample_existing(self, batch_size, device):
        """
        Low-cost sampling: pick random existing block configs,
        then Sobol-sample only the sequencing dims.
        """
        print(f"--- In _sample_existing (batch_size={batch_size}) ---")
        print(f"_sample_existing: self.bounds shape: {self.bounds.shape}")
        print(f"_sample_existing: self.bounds.size(0): {self.bounds.size(0)}") # Check size here
        n_block = len(self.block_params)
        # Check calculation of n_seq
        if self.bounds.size(0) < n_block:
             print(f"ERROR: bounds size ({self.bounds.size(0)}) is less than n_block ({n_block})!")
             raise ValueError("Bounds size is insufficient for block parameters")
        n_seq   = self.bounds.size(0) - n_block
        print(f"_sample_existing: n_block={n_block}, n_seq={n_seq}")
        if n_seq <= 0:
             print(f"ERROR: n_seq is non-positive ({n_seq}). Cannot create SobolEngine.")
             raise ValueError("Number of sequence parameters must be positive")

        # sobol dimension should be n_seq
        sobol  = SobolEngine(dimension=n_seq, scramble=True)
        print(f"_sample_existing: Created SobolEngine with dimension={n_seq}")
        blocks = random.choices(list(self.available_configs), k=batch_size)
        seqs   = sobol.draw(batch_size).to(device=device, dtype=torch.double)

        samples = []
        for cfg, seq in zip(blocks, seqs):
            # Ensure indices are within bounds for self.bounds
            if n_block + n_seq > self.bounds.size(0):
                 print(f"ERROR: Parameter index calculation out of bounds! {n_block+n_seq} vs {self.bounds.size(0)}")
                 raise IndexError("Parameter index calculation out of bounds")

            vals = list(cfg) + [
                self.bounds[n_block + j, 0] + (self.bounds[n_block + j, 1] - self.bounds[n_block + j, 0]) * seq[j]
                for j in range(n_seq)
            ]
            samples.append(torch.tensor(vals, dtype=torch.double, device=device))

        print("--- Exiting _sample_existing ---")
        return torch.stack(samples)


    def _sample_new(self, batch_size, device):
        """
        High-cost sampling: full Sobol draw over all dims.
        """
        print(f"--- In _sample_new (batch_size={batch_size}) ---")
        print(f"_sample_new: self.bounds shape: {self.bounds.shape}")
        print(f"_sample_new: self.bounds.size(0): {self.bounds.size(0)}") # Check size here
        dimension_for_sobol = self.bounds.size(0) # Capture before use
        print(f"_sample_new: Dimension for SobolEngine: {dimension_for_sobol}")
        print(f"_sample_new: Type of dimension_for_sobol: {type(dimension_for_sobol)}")

        # Check if dimension is valid before creating SobolEngine
        if dimension_for_sobol <= 0:
            print(f"ERROR: Dimension for SobolEngine is non-positive ({dimension_for_sobol}).")
            raise ValueError("Dimension for SobolEngine must be positive")

        # This is the line causing the error based on the traceback
        try:
            sobol = SobolEngine(dimension=dimension_for_sobol, scramble=True)
            print(f"_sample_new: Created SobolEngine with dimension={dimension_for_sobol}")
        except Exception as e:
            print(f"ERROR: Failed to create SobolEngine with dimension {dimension_for_sobol}: {e}")
            raise # Re-raise the exception

        unit  = sobol.draw(batch_size).to(device=device, dtype=torch.double)
        # unnormalize into real‐parameter space
        unn_bounds = self.bounds.T.to(device=device, dtype=torch.double)
        print(f"_sample_new: Bounds transpose shape for unnormalize: {unn_bounds.shape}")
        try:
            unnormalized_samples = unnormalize(unit, unn_bounds)
            print(f"_sample_new: Unnormalized samples shape: {unnormalized_samples.shape}")
        except Exception as e:
             print(f"ERROR: Failed to unnormalize samples: {e}")
             raise # Re-raise

        print("--- Exiting _sample_new ---")
        return unnormalized_samples

    def optimize_eiuu(
        self,
        n_iterations: int = 50,
        q: int = 1,
        M: int = 16,    # number of sampled utilities per EIUU eval
        k: int = 8,     # number of low/high candidates to score
    ):
        print(f"\n=== Starting optimize_eiuu: n_iter={n_iterations}, q={q}, M={M}, k={k} ===")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dim = self.bounds.size(0)

        # normalize / unnormalize lambdas
        lb = self.bounds[:, 0].to(device).double()
        ub = self.bounds[:, 1].to(device).double()
        rng = (ub - lb).clamp(min=1.0)
        normalize = lambda X: (X - lb) / rng
        unnormalize = lambda X: lb + X * rng

        # initialize history
        self.X = torch.empty((0, dim), dtype=torch.double, device=device)
        self.Y = torch.empty((0, len(REF_POINT)), dtype=torch.double, device=device)
        self.spent_budget = 0.0

        # try loading past runs
        use_loaded = self.get_historical('20250624_041000', q, device)

        running = {}
        eval_counter = 0
        last_status = time.time()

        def _dispatch(x):
            nonlocal eval_counter
            eval_counter += 1
            print(f"[DISPATCH] #{eval_counter} x shape={tuple(x.shape)}")
            fut = pool.submit(self.evaluate_parameters, x)
            running[fut] = x

        # start thread pool
        max_workers = max(q, len(self.licenses), 1)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:

            # --- initial seeding/proposals ---
            if use_loaded and self.X.size(0) >= 2:
                print("[INIT] Fitting multi-output GP on loaded data...")
                Xn = normalize(self.X)
                # build independent GPs for each objective
                gps = []
                for i in range(self.Y.size(1)):
                    gp_i = SingleTaskGP(
                        Xn,
                        self.Y[:, i : i + 1],
                        input_transform=Normalize(d=dim),
                        outcome_transform=Standardize(m=1),
                    ).to(device)
                    gps.append(gp_i)
                mogp = ModelListGP(*gps).to(device)
                mll = SumMarginalLogLikelihood(mogp.likelihood, mogp)
                

                fit_gpytorch_mll(mll, options={"maxiter": 50})
                print("[INIT] GP fit complete.")

                # build EIUU acquisition
                eiuu = EIUU(model=mogp, M=M).to(device)

                # sample candidate batches
                low = self._sample_existing(k, device=device).double()
                high = self._sample_new(k, device=device).double()
                print("Sampled low/high candidates:", low.shape, high.shape)

                with torch.no_grad():
                    sl = eiuu(normalize(low).unsqueeze(1)).squeeze(-1)
                    sh = eiuu(normalize(high).unsqueeze(1)).squeeze(-1)
                print("Scores: sl.max()=", sl.max().item(), "sh.max()=", sh.max().item())

                combined = torch.cat([sl, sh])
                cands = torch.cat([low, high], dim=0)
                topk_vals, topk_idx = combined.topk(q)
                print("Top-k EIUU vals:", topk_vals)
                for idx in topk_idx:
                    _dispatch(cands[idx])

            else:
                print("[SEED] No historical data – sampling existing points")
                for i in range(max_workers):
                    x0 = (
                        self._sample_existing(1, device=device)
                        .squeeze(0)
                        .double()
                    )
                    _dispatch(x0)

            # --- main loop ---
            n = 0
            while running:
                done, _ = wait(running, timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in list(done):
                    x_old = running.pop(fut)
                    y_old, cost_old = fut.result()
                    # record
                    y_t = torch.as_tensor(y_old, dtype=torch.double, device=device).unsqueeze(0)
                    self.X = torch.cat([self.X, x_old.unsqueeze(0)], dim=0)
                    self.Y = torch.cat([self.Y, y_t], dim=0)
                    self.spent_budget += float(cost_old)
                    print(f"[UPDATE] n_samples={self.X.size(0)}, spent_budget={self.spent_budget:.2f}")

                    if n >= n_iterations:
                        print("[TERMINATE] reached max iterations.")
                        for p in running:
                            p.cancel()
                        running.clear()
                        break

                    # propose next
                    if n >= 2:
                        print("[PROPOSE] fitting GP + EIUU...")
                        Xn = normalize(self.X)
                        # multi-output GP
                        gps = []
                        for i in range(self.Y.size(1)):
                            gps.append(
                                SingleTaskGP(
                                    Xn,
                                    self.Y[:, i : i + 1],
                                    input_transform=Normalize(d=dim),
                                    outcome_transform=Standardize(m=1),
                                ).to(device)
                            )
                        mogp = ModelListGP(*gps).to(device)
                        mll = SumMarginalLogLikelihood(mogp.likelihood, mogp)
                        fit_gpytorch_mll(mll, options={"maxiter": 20})
                        eiuu = EIUU(model=mogp, M=M).to(device)

                        low = self._sample_existing(k, device=device).double()
                        high = self._sample_new(k, device=device).double()
                        with torch.no_grad():
                            sl = eiuu(normalize(low).unsqueeze(1)).squeeze(-1)
                            sh = eiuu(normalize(high).unsqueeze(1)).squeeze(-1)
                        print(f"  sl.max()={sl.max():.4f}, sh.max()={sh.max():.4f}")

                        # pick best of low vs high
                        combined = torch.cat([sl, sh])
                        cands = torch.cat([low, high], dim=0)
                        topk_vals, topk_idx = combined.topk(1)
                        x_next = cands[topk_idx[0]]
                    else:
                        x_next = self._sample_existing(1, device=device).squeeze(0).double()

                    _dispatch(x_next)
                    n += 1

                if time.time() - last_status > 10:
                    print(f"[STATUS] pending={len(running)}, spent_budget={self.spent_budget:.2f}")
                    last_status = time.time()

        # pareto filter on the raw multi‐objective Y  
        mask = self._pareto_mask(self.Y.cpu().numpy())
        print(f"[DONE] Returning {mask.sum()} Pareto pts.")
        return self.X[mask], self.Y[mask]

    def optimize_g(
        self,
        n_iterations: int = 50,
        q: int = 1,
        alpha: float = 1.0,
        k: int = 8,  # how many low/high candidates to compare each round
    ):
        device = torch.device("cpu")
        max_workers = max(q, len(self.licenses), 1)
        dim = self.bounds.size(0)
        num_obj = len(REF_POINT)

        # Prepare history
        self.bounds = self.bounds.to(device=device, dtype=torch.double)
        lb, ub = self.bounds[:, 0].clone(), self.bounds[:, 1].clone()
        range_ = ub - lb
        range_[range_ == 0] = 1.0
        normalise = lambda x: (x - lb) / range_
        self.X = torch.empty((0, dim), dtype=torch.double, device=device)
        self.Y = torch.empty((0, num_obj), dtype=torch.double, device=device)
        self.spent_budget = 0.0

        running = {}          # Future -> x tensor
        submission_times = {} # Future -> datetime
        eval_counter = 0

        def _get_mem():
            p = psutil.Process(os.getpid())
            m = p.memory_info()
            return m.rss / 1024**2, p.memory_percent()

        def _print_active(msg_prefix: str = ""):
            now = datetime.now().strftime("%H:%M:%S")
            rss, pct = _get_mem()
            print(f"{msg_prefix}[{now}] ACTIVE={len(running)} | RSS={rss:.1f}MB ({pct:.1f}%)")
            for i, (fut, x) in enumerate(running.items(), start=1):
                dur = (time.time() - submission_times[fut].timestamp())
                x_short = x.tolist()[:3] + ["..."] if len(x) > 3 else x.tolist()
                print(f"   {i}. x={x_short} | running {dur:.1f}s")

        def _dispatch(x, pool):
            nonlocal eval_counter
            eval_counter += 1
            ts = datetime.now()
            rss, pct = _get_mem()
            print(f"📤 [{ts:%H:%M:%S}] Dispatch #{eval_counter} | "
                f"x={x.tolist()[:5]}… | RSS={rss:.1f}MB ({pct:.1f}%)")
            fut = pool.submit(self.evaluate_parameters, x)
            running[fut] = x
            submission_times[fut] = ts

        # ───────── executor and kick-off ─────────
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # seed all workers with cheap evals
            for i in range(max_workers):
                seed = self._sample_existing(1, device=device).squeeze(0)
                print(f"🌱 Seeding worker {i+1}/{max_workers}")
                _dispatch(seed, pool)

            loop_it = 0
            last_status = time.time()

            # ───────── main loop with single timed wait ─────────
            while running:
                loop_it += 1

                # wait up to 1s for any completion, return all done futures
                done, _ = wait(running.keys(), timeout=1.0, return_when=FIRST_COMPLETED)

                # process each completed future immediately
                for fut in list(done):
                    x_fin = running.pop(fut)
                    sub_ts = submission_times.pop(fut)
                    dur = (datetime.now() - sub_ts).total_seconds()
                    print('before result ')
                    y_fin, cost_fin = fut.result()

                    # record history
                    if isinstance(y_fin, tuple):
                        print('y_fin is a tuple')
                        y_fin = torch.tensor(y_fin)
                    if isinstance(y_fin, torch.Tensor) and y_fin.dim() == 1:
                        print('y_fin is a Tensor')
                        y_fin = y_fin.unsqueeze(0)
                    self.X = torch.cat([self.X, x_fin.unsqueeze(0)], dim=0)
                    self.Y = torch.cat([self.Y, y_fin], dim=0)
                    self.spent_budget += float(cost_fin)
                    n_eval = self.X.size(0)

                    print(f"✅ [{datetime.now():%H:%M:%S}] Completed | "
                        f"x={x_fin.tolist()[:3]}… | Dur={dur:.1f}s | Cost={cost_fin:.2f} | Y : {y_fin}"
                        f"Progress={n_eval}/{n_iterations}")

                    # stopping criteria
                    if n_eval >= n_iterations or self.spent_budget >= self.total_budget:
                        print(f"🛑 [{datetime.now():%H:%M:%S}] Stopping; cancelling remaining.")
                        for pending in running:
                            pending.cancel()
                        running.clear()
                        break

                    # immediate replacement
                    if n_eval >= 2:
                        #acq = self.cost_cooled_parEGO(alpha=alpha)
                        # acq = self.cost_cooled_ehvi(alpha=alpha)
                        acq = self.cost_cooled_mtEHVI(alpha = alpha )
                        low = self._sample_existing(k, device=device)
                        print('low , ' , low)
                        high = self._sample_new(k, device=device)
                        print('high , ', high )
                        #with torch.no_grad():
                        #    ls = acq(normalise(low).unsqueeze(1))
                        #    hs = acq(normalise(high).unsqueeze(1))
                        #    
                        with torch.no_grad():
                            ls = acq(normalise(low).unsqueeze(1)) .squeeze(-1)   # (k,)
                            hs = acq(normalise(high).unsqueeze(1)).squeeze(-1)
                            print('ls , ' , ls )
                            print('hs , ' , hs )
                        idx_low  = ls.argmax()
                        idx_high = hs.argmax()
                        x_next = low[idx_low] if ls[idx_low] >= hs[idx_high] else high[idx_high]
                        #x_next = low[ls.argmax()] if ls.max() >= hs.max() else high[hs.argmax()]
                        print('x_next' , x_next)
                    else:
                        print('IN THE LESS THAN 2 ELSE ')
                        x_next = self._sample_new(1, device=device).squeeze(0)

                    _dispatch(x_next, pool)

                    if loop_it % 5 == 0:
                        gc.collect()

                # every 10 s print active processes
                if time.time() - last_status >= 10:
                    _print_active(msg_prefix="🔄 STATUS ")
                    last_status = time.time()

        # ───────── post-process ─────────
        mask = self._pareto_mask(self.Y.cpu().numpy())
        return self.X[mask], self.Y[mask]
    def get_historical(self, date,q, device ): 
        
        metrics_pattern = os.path.join(SAVE_PATH, 'metrics', date + '*.csv')
        csv_files = glob.glob(metrics_pattern)
        use_loaded = False
        
        if len(csv_files) > 5:
            # load all matching CSVs into a single DataFrame
            dfs = [pd.read_csv(f) for f in csv_files]
            data = pd.concat(dfs, ignore_index=True)
            print('data ')
            # assume first `dim` columns are the decision variables
            metrics = data.iloc[:, :14]
            metrics['triples'] = data['triple in 24h (no gaps)'] + data['triple in same day (no gaps)']
            metrics['b2b'] = data['evening/morning b2b'] + data['other b2b']
            print('Y ')
            print(metrics[[ 'triples', 'three in four slots', 'b2b' , 'two in three slots' , 'singular late exam' ,'two exams, large gap' , 'avg_max', 'lateness']])
            Y_loaded = torch.tensor(
                metrics[[ 'triples', 'three in four slots', 'b2b' , 'two in three slots' , 'singular late exam' ,'two exams, large gap' , 'avg_max', 'lateness']].values,
                dtype=torch.double, device = device 
            )
            print(' data.iloc[:, 14: ' ,  data.iloc[:, 14:])
            X_loaded = torch.tensor(data.iloc[:, 14:].values, dtype=torch.double ,device=device)
            # next columns correspond to the objective vector of length len(REF_POINT)
           
            self.X = X_loaded
            self.Y = Y_loaded
            # accumulate cost if present
            if 'cost' in data.columns:
                self.spent_budget = float(data['cost'].sum())
            use_loaded = True
            print(f"Loaded {self.X.size(0)} samples from metrics files.")

        return use_loaded
    def optimize_fid(
        self,
        n_iterations: int = 50,
        q: int = 5,
        alpha: float = 1.0,
    ):
        device = torch.device("cuda:0")
        dim = self.bounds.size(0)
        # normalize helpers
        self.bounds = self.bounds.to(device)

        # now everything derived from bounds is on CUDA
        lb, ub = self.bounds[:, 0], self.bounds[:, 1]
        rng = (ub - lb).clamp(min=1.0)
        normalize = lambda X: (X - lb) / rng
        unnormalize_fn = lambda X: lb + X * rng
        max_workers =max(q,len(self.licenses),1)
        # history
        #self.X = torch.empty((0,dim),dtype=torch.double)
        self.X = torch.empty((0, dim), dtype=torch.double, device=device)
        #self.Y = torch.empty((0,len(REF_POINT)),dtype=torch.double)
        self.Y = torch.empty((0, len(REF_POINT)), dtype=torch.double, device=device)
        self.spent_budget = 0.0
        # threading
        running, times = {}, {}
        #etrics_pattern = os.path.join(SAVE_PATH, 'metrics', '20250624_041000*.csv')
        #csv_files = glob.glob(metrics_pattern)
        max_workers =max(q,len(self.licenses),1)
        use_loaded = self.get_historical('20250624_041000', max_workers, device)

        eval_counter=0; last_status=time.time()
        def dispatch(x,pool):
            nonlocal eval_counter
            eval_counter+=1; ts=datetime.now()
            fut=pool.submit(self.evaluate_parameters,x)
            running[fut]=x; times[fut]=ts
        # main loop
        with ThreadPoolExecutor(max_workers) as pool:
            # seed
            if use_loaded and self.X.size(0) >= 2:
                # build acquisition function on loaded history
                print('use_loaded , ' )
                train_X_norm = normalize(self.X)
                acq_fn = build_parego_ei(
                    train_X_norm,
                    self.Y,
                    torch.stack([torch.zeros(dim), torch.ones(dim)], 1).T,
                    torch.tensor(REF_POINT, dtype=torch.double).unsqueeze(0),
                    q=q,
                    num_restarts=5,
                    raw_samples=128,
                )
                print('acq_fn , ' , acq_fn )
                # sample and score candidates
                low = self._sample_existing(100, device)
                high = self._sample_new(100, device)

                with torch.no_grad():
                    ls = acq_fn(normalize(low).unsqueeze(1)).view(-1) * self.cost_ratio
                    hs = acq_fn(normalize(high).unsqueeze(1)).view(-1)
                combined = torch.cat([ls, hs])
                print('combined , ' , combined)
                candidates = torch.cat([low, high], dim=0)
                print('cand , ' , candidates)
                topk_vals, topk_idx = combined.topk(q)
                for idx in topk_idx:
                    dispatch(candidates[idx], pool)
            else:
                # fallback to original cheap-seeding
                for _ in range(max(q, len(self.licenses), 1)):
                    dispatch(self._sample_existing(1, device).squeeze(0), pool)
            n_eval = 0 
            while running:
                
                done,_=wait(running.keys(),timeout=1.0,return_when=FIRST_COMPLETED)
                for fut in list(done):
                    print('N_EVAL : ' , n_eval )
                    x_fin=running.pop(fut); times.pop(fut)
                    print('x_fin ' , x_fin)
                    y_fin,cost_fin=fut.result()
                    y_fin = torch.as_tensor(y_fin).unsqueeze(0).to(device)
                    self.X=torch.cat([self.X,x_fin.unsqueeze(0)],0)
                    
                    self.Y=torch.cat([self.Y,y_fin],0)
                    self.spent_budget+=cost_fin
                    #n_eval=self.X.size(0)
                    #if n_eval>=n_iterations:# or self.spent_budget>=self.total_budget:
                    #    print("AAA A AA ")
                    #    for f in running: f.cancel()
                    #    running.clear(); break
                    #if n_eval >= 2:
                    #in this 
                    # build acquisition (vanilla EI)
                    train_X_norm = normalize(self.X)
                    acq_fn = build_parego_ei(
                        train_X_norm,
                        self.Y,
                        torch.stack([torch.zeros(dim), torch.ones(dim)], 1).T,
                        torch.tensor(REF_POINT, dtype=torch.double).unsqueeze(0),
                        q=q,
                        num_restarts=5,
                        raw_samples=128,
                    )

                    # sample 5 cheap and 5 expensive candidates
                    k = 5
                    low = self._sample_existing(10* k, device)   # (5, d)
                    high = self._sample_new(k, device)       # (5, d)
                    print(f"Sampling low-cost candidates: {low.shape}")
                    print(f"Sampling high-cost candidates: {high.shape}")

                    # score them
                    with torch.no_grad():
                        ls = acq_fn(normalize(low).unsqueeze(1)).view(-1)   # (5,)
                        hs = acq_fn(normalize(high).unsqueeze(1)).view(-1)  # (5,)
                    print(f"Acq values low: {ls.tolist()}")
                    print(f"Acq values high: {hs.tolist()}")

                    print( 'acq ls type, ', type(ls)) 
                    print( 'acq hs type, ' ,type(hs)) 
                    # pick best from each set
                    low_val, low_idx = ls.max(dim=0)
                    high_val, high_idx = hs.max(dim=0)
                    best_low = low[low_idx]
                    best_high = high[high_idx]
                    print('low_val ' , low_val)
                    print('high_val ' , high_val )
                    print('best_low , ' , best_low )
                    print('best_high. , ' , best_high)
                    # cost‐aware choose
                    if low_val * self.cost_ratio >= high_val:
                        x_next = best_low
                        print(f"Chose low-cost candidate (idx={low_idx}, EI={low_val:.4f}).")
                        #else:
                        #    x_next = best_high
                        #    print(f"Chose high-cost candidate (idx={high_idx}, EI={high_val:.4f}).")
                    else:
                        x_next = best_low#self._sample_existing(1,device).squeeze(0)
                    dispatch(x_next,pool)
                    n_eval+=1 
                    if time.time()-last_status>10:
                        last_status=time.time()
                        print(f"STATUS active={len(running)}")
                    gc.collect()
        mask=self._pareto_mask(self.Y.numpy())
        return self.X[mask], self.Y[mask]
    def optimize_thompson(
        self,
        n_iterations: int = 50,
        q: int = 1,
        alpha: float = 1.0,
        k: int = 8,  # number of low/high candidates
    ):
        print(f"\n=== Starting optimize_thompson: n_iter={n_iterations}, q={q}, alpha={alpha}, k={k} ===")
        # 1) Setup device and Sobol engine
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dim = self.bounds.size(0)

        # 2) Bounds normalization helpers
        lb = self.bounds[:, 0].to(device).double()
        ub = self.bounds[:, 1].to(device).double()
        rng = (ub - lb).clamp(min=1.0)
        normalize = lambda X: (X - lb) / rng
        unnormalize = lambda X: lb + X * rng

        # 3) History initialization or loading saved samples
        self.X = torch.empty((0, dim), dtype=torch.double, device=device)
        self.Y = torch.empty((0, len(REF_POINT)), dtype=torch.double, device=device)
        self.spent_budget = 0.0
        use_loaded = False


        use_loaded = self.get_historical('20250624_041000', q, device)

        running = {}
        eval_counter = 0
        last_status = time.time()
        ref_pt = torch.tensor(REF_POINT, dtype=torch.double, device=device).unsqueeze(0)

        def _dispatch(x):
            nonlocal eval_counter
            eval_counter += 1
            print(f"[DISPATCH] #{eval_counter} x shape={tuple(x.shape)}")
            fut = pool.submit(self.evaluate_parameters, x)
            running[fut] = x

        # 5) Thread pool and initial proposals
        max_workers = max(q, len(self.licenses), 1)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
           
            if use_loaded and self.X.size(0) >= 2:
                print("[INIT] Building GP on loaded data...")
                Xn = normalize(self.X)
                print("Xn:", Xn)
                # scalarize Y
                w = torch.rand(self.Y.size(1), device=device)
                w /= w.sum()
                Ys = ((self.Y - ref_pt) * w).max(dim=-1).values.unsqueeze(-1)
                # fit GP
                gp = SingleTaskGP(
                    Xn, Ys,
                    input_transform=Normalize(d=dim),
                    outcome_transform=Standardize(m=1),
                ).to(device)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                print("Fitting GP on loaded data...")
                fit_gpytorch_mll(mll, options={"maxiter": 100})
                print("GP fit complete.")
                # build Thompson acquisition
                acq = CostCooledTS(model=gp, alpha=alpha).to(device)
                print("Acquisition (CostCooledTS) instantiated.")

                # sample candidates
                low  = self._sample_existing(k, device=device).to(device).double()
                high = self._sample_new(k, device=device).to(device).double()
                print("Sampled candidates: low.shape=", low.shape, "high.shape=", high.shape)

                with torch.no_grad():
                    sl = acq(normalize(low).unsqueeze(1)).squeeze(-1) * self.cost_ratio
                    sh = acq(normalize(high).unsqueeze(1)).squeeze(-1)
                print("Scores: sl=", sl, "sh=", sh)

                combined = torch.cat([sl, sh])
                candidates = torch.cat([low, high], dim=0)
                topk_vals, topk_idx = combined.topk(q)
                print("Top-k values:", topk_vals)
                print("Top-k indices:", topk_idx)
                for idx in topk_idx:
                    _dispatch(candidates[idx])

            else:
                print("[SEED] Falling back to seeding.")
                for i in range(max_workers):
                    x0 = self._sample_existing(1, device=device).squeeze(0).to(device).double()
                    print(f"[SEED] Worker {i+1}/{max_workers} x0 shape={tuple(x0.shape)}")
                    _dispatch(x0)

            # 6) Main optimization loop
            n = 0 
            while running:
                
                done, _ = wait(running, timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in list(done):
                    print('n  ', n)
                    x_old = running.pop(fut)
                    y_old, cost_old = fut.result()
                    print(f"[RESULT] Received y={y_old}, cost={cost_old:.2f}")

                    # update history
                    y_tensor = torch.as_tensor(y_old, dtype=torch.double, device=device).unsqueeze(0)
                    self.X = torch.cat([self.X, x_old.unsqueeze(0)], dim=0)
                    self.Y = torch.cat([self.Y, y_tensor], dim=0)
                    self.spent_budget += float(cost_old)
                    #n = self.X.size(0)
                    print(f"[UPDATE] n_samples={n}, spent_budget={self.spent_budget:.2f}")

                    # termination
                    if n >= n_iterations :#:or self.spent_budget >= self.total_budget:
                        print("[TERMINATE] Budget or iterations exhausted—stopping.")
                        for p in running: p.cancel()
                        running.clear()
                        break

                    # propose new
                    if n >= 2:
                        print("[PROPOSE] Building GP and CostCooledTS acquisition...")
                        # Normalize + scalarize
                        Xn = normalize(self.X)
                        w = torch.rand(self.Y.size(1), device=device)
                        w /= w.sum()
                        Ys = ((self.Y - ref_pt) * w).max(dim=-1).values.unsqueeze(-1)
                        print(f"         Xn shape={Xn.shape}, Ys shape={Ys.shape}")

                        # Fit GP
                        gp = SingleTaskGP(
                            Xn, Ys,
                            input_transform=Normalize(d=dim),
                            outcome_transform=Standardize(m=1),
                        ).to(device)
                        print("         Fitting GP...")
                        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                        fit_gpytorch_mll(mll, options={"maxiter": 20})
                        print("         GP fit complete.")

                        # Build acquisition
                        acq = CostCooledTS(model=gp, alpha=alpha).to(device)
                        print("         Acquisition built.")

                        # Draw candidates
                        low  = self._sample_existing(k, device=device).to(device).double()
                        high = self._sample_new(k, device=device).to(device).double()
                        print(f"         low.shape={low.shape}, high.shape={high.shape}")

                        # Score
                        with torch.no_grad():
                            sl = acq(normalize(low).unsqueeze(1)).squeeze(-1)
                            sh = acq(normalize(high).unsqueeze(1)).squeeze(-1)
                        print(f"         Scores: sl.max()={sl.max():.4f}, sh.max()={sh.max():.4f}")

                        # ** filled-in selection **
                        if sl.max() * self.cost_ratio >= sh.max():
                            best_idx = sl.argmax().item()
                            x_next = low[best_idx]
                            print(f"         Selected LOW[{best_idx}] with score={sl[best_idx]:.4f}")
                        else:
                            best_idx = sh.argmax().item()
                            x_next = high[best_idx]
                            print(f"         Selected HIGH[{best_idx}] with score={sh[best_idx]:.4f}")
                    else:
                        x_next = self._sample_existing(1, device=device).squeeze(0).to(device).double()
                        print("         Not enough data yet—sampling existing.")

                    _dispatch(x_next)
                    n+=1 

                # periodic status
                if time.time() - last_status > 10:
                    print(f"[STATUS] pending={len(running)}, spent_budget={self.spent_budget:.2f}")
                    last_status = time.time()

        mask = self._pareto_mask(self.Y.cpu().numpy())
        print(f"[DONE] Returning {mask.sum()} Pareto-optimal points.")
        return self.X[mask], self.Y[mask]


def run_cost_aware_optimization(
    objective_fn,
    post_processing , 
    block_assignment , 
    licenses,
    blocks_dir=SAVE_PATH + 'blocks/',
    total_budget=3600,
    cost_ratio=3,
    n_iterations=100,
):
    """
    Run the CArBO optimizer and return a DataFrame of the pareto-optimal
    (parameters + objective values).
    """
    print("CARBO ")
    print("COST RATIO " , cost_ratio)
    process_block_assignments() # clear out any bad block assignments 
    # 1) Initialize and run
    optimizer = CostAwareSchedulingOptimizer(
        blocks_dir=blocks_dir,
        total_budget=total_budget,
        licenses=licenses,
        cost_ratio=cost_ratio,
        objective_fn=objective_fn, 
        block_assignment= block_assignment , 
        post_processing = post_processing 
    )
    if 'thom' in run_name: 
        print("RUNNING THOMPSON OPT")
        X_pareto, Y_pareto = optimizer.optimize_thompson(n_iterations=n_iterations, q = 5, k = 1000)
    if 'eiuu' in run_name: 
        print('RUNNING EIUU') 
        X_pareto, Y_pareto = optimizer.optimize_eiuu(n_iterations=n_iterations, q = 5 ) 
    else: 
        print("RUNNING PARETO OPT")
        X_pareto, Y_pareto = optimizer.optimize_fid(n_iterations=n_iterations, q = 5 ) 
    # 2) Build a DataFrame
    #    - one column per parameter
    #    - one column per objective dimension
    param_names = optimizer.param_names
    df = pd.DataFrame(
        X_pareto.cpu().numpy(),
        columns=param_names
    )
    # if multi-dim objectives, add columns "obj_0", "obj_1", ...
    Y_np = Y_pareto.cpu().numpy()
    if Y_np.ndim == 1:
        df['objective'] = Y_np
    else:
        for i in range(Y_np.shape[1]):
            df[f'objective_{i}'] = Y_np[:, i]

    # 3) Print a quick summary
    print(f"\nOptimization complete! Pareto front has {len(df)} points.")
    print(df.head())

    return df 

