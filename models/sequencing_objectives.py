import gurobipy as gp
import pandas as pd
import traceback
import numpy as np
from pathlib import Path
import os 
from models.sequencing import build_scheduler_model, solve_model

from config.settings import SEMESTER , NUM_SLOTS ,SAVE_PATH, GLOBAL_TIMESTAMP, POST_PROCESSING, EMPTY_BLOCKS, get_name , load_exam_data
from post_processing.post_processing import run_pp

import datetime

# Shared data (load once)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def exam_schedule_objective_random(params, license_env, good=True):
    """
    A random-but-structured surrogate objective for CArBO testing.
    When good==True, returns lower (better) metrics; when False, returns worse metrics.
    
    Args:
        params: tuple containing
            block_file (str),
            alpha (float), gamma (float), delta (float),
            vega (float), theta (float),
            large_block_size (float),
            large_exam_weight (float),
            large_block_weight (float),
            large_size_1 (float),
            large_size_2 (float)
        license_env: unused placeholder for license info
        good (bool): if True, sample metrics from “good” (low) ranges;
                     if False, sample from “bad” (high) ranges.
    
    Returns:
        tuple of 8 floats:
            (triple24 + triple_same,
             three_in_four,
             combined_b2b,
             two_in_three,
             singular_late,
             large_gap,
             avg_max_gap,
             lateness)
    """
    block_file, alpha, gamma, delta, vega, theta, Ls, Lew, Lbw, Ls1, Ls2 = params

    # load block assignment to ensure file exists
    try:
        _ = pd.read_csv(block_file)
    except Exception as e:
        print(f"Error reading block file: {e}")
        # return a very bad constant if file load fails
        return (1000, 1000, 1000, 1000, 1000, 1000, 1000.0, 100000.0)

    # choose ranges based on quality flag
    if good:
        # “good” schedules: low counts and low lateness
        tri24    = np.random.randint(0, 3)
        trisame  = np.random.randint(0, 3)
        three4   = np.random.randint(0, 2)
        emb2b    = np.random.randint(0, 2)
        otherb2b = np.random.randint(0, 2)
        two3     = np.random.randint(0, 3)
        singlate = np.random.randint(0, 2)
        two_gap  = np.random.randint(0, 2)
        avg_max  = float(np.random.uniform(0, 5))
        lateness = float(np.random.uniform(0, 1000))
    else:
        # “bad” schedules: higher counts and higher lateness
        tri24    = np.random.randint(5, 15)
        trisame  = np.random.randint(5, 15)
        three4   = np.random.randint(3, 10)
        emb2b    = np.random.randint(3, 10)
        otherb2b = np.random.randint(3, 10)
        two3     = np.random.randint(10, 50)
        singlate = np.random.randint(5, 20)
        two_gap  = np.random.randint(5, 20)
        avg_max  = float(np.random.uniform(10, 30))
        lateness = float(np.random.uniform(20000, 60000))

    # assemble the 8 objectives
    obj0 = tri24 + trisame               # triple24 + triple_same
    obj1 = three4                        # three in four slots
    obj2 = emb2b + otherb2b             # combined b2b
    obj3 = two3                          # two in three slots
    obj4 = singlate                      # singular late exam
    obj5 = two_gap                       # two exams, large gap
    obj6 = avg_max                       # average max gap
    obj7 = lateness                      # total lateness

    return (obj0, obj1, obj2, obj3, obj4, obj5, obj6, obj7)