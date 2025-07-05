import numpy as np
import pandas as pd
import math
import os
from matplotlib import pyplot as plt
from itertools import product

# from config.settings import SAVE_PATH, DATA_PATH, UI_PATH, SEMESTER
# from globals.build_global_sets import normalize_and_merge
UI_PATH = "/Users/adamshafikjovine/Documents/BOScheduling/metrics/plots/"
SAVE_PATH = "/Users/adamshafikjovine/Documents/BOScheduling/results/sp25"
SEMESTER = "sp25"
DATA_PATH = "/Users/adamshafikjovine/Documents/BOScheduling/data/" + SEMESTER
times = [""]


def slots_to_time(slots):
    d = {}
    if "fa" in SEMESTER:
        d = {
            1: "Dec 13, 9am",
            2: "Dec 13, 2pm",
            3: "Dec 13, 7pm",
            4: "Dec 14, 9am",
            5: "Dec 14, 2pm",
            6: "Dec 14, 7pm",
            7: "Dec 15, 9am",
            8: "Dec 15, 2pm",
            9: "Dec 15, 7pm",
            10: "Dec 16, 9am",
            11: "Dec 16, 2pm",
            12: "Dec 16, 7pm",
            13: "Dec 17, 9am",
            14: "Dec 17, 2pm",
            15: "Dec 17, 7pm",
            16: "Dec 18, 9am",
            17: "Dec 18, 2pm",
            18: "Dec 18, 7pm",
            19: "Dec 19, 9am",
            20: "Dec 19, 2pm",
            21: "Dec 19, 7pm",
            22: "Dec 20, 9am",
            23: "Dec 20, 2pm",
            24: "Dec 20, 7pm",
            25: "Dec 21, 9am",
            26: "Dec 21, 2pm",
            27: "Dec 21, 7pm",
        }
    else:
        d = {
            1: "May 11, 9am",
            2: "May 11, 2pm",
            3: "May 11, 7pm",
            4: "May 12, 9am",
            5: "May 12, 2pm",
            6: "May 12, 7pm",
            7: "May 13, 9am",
            8: "May 13, 2pm",
            9: "May 13, 7pm",
            10: "May 14, 9am",
            11: "May 14, 2pm",
            12: "May 14, 7mm",
            13: "May 15, 9am",
            14: "May 15, 2pm",
            15: "May 15, 7pm",
            16: "May 16, 9am",
            17: "May 16, 2pm",
            18: "May 16, 7pm",
            19: "May 17, 9am",
            20: "May 17, 2pm",
            21: "May 17, 7pm",
            22: "May 18, 9am",
            23: "May 18, 2pm",
            24: "May 18, 7pm",
            25: "May 19, 9am",
            26: "May 19, 2pm",
            27: "May 19, 7pm",
        }

    return [d[s] for s in slots]


# Create the chart
def get_plot(schedule_name, name):
    sched = pd.read_csv(SAVE_PATH + "/schedules/" + schedule_name)
    exam_sizes = pd.read_csv(DATA_PATH + "/exam_sizes.csv")
    slots = np.unique(sched["slot"].values)

    num_slots1 = len(slots)
    num_slots2 = int(max(slots))
    h = np.zeros(num_slots2)
    h1 = np.zeros(num_slots2)
    h2 = np.zeros(num_slots2)
    h3 = np.zeros(num_slots2)
    h4 = np.zeros(num_slots2)
    for s in slots:
        s = int(s)
        exams = sched[sched["slot"] == s]["exam"].tolist()
        exams_over_400 = sched[(sched["slot"] == s) & (sched["size"] >= 400)][
            "exam"
        ].tolist()
        exams_in_300_400 = sched[
            (sched["slot"] == s) & (sched["size"] >= 300) & (sched["size"] < 400)
        ]["exam"].tolist()
        exams_in_200_300 = sched[
            (sched["slot"] == s) & (sched["size"] >= 200) & (sched["size"] < 300)
        ]["exam"].tolist()
        exams_in_100_200 = sched[
            (sched["slot"] == s) & (sched["size"] >= 100) & (sched["size"] < 200)
        ]["exam"].tolist()
        sizes_over_400 = exam_sizes[exam_sizes["exam"].isin(exams_over_400)][
            "size"
        ].sum()
        sizes_in_300_400 = exam_sizes[exam_sizes["exam"].isin(exams_in_300_400)][
            "size"
        ].sum()
        sizes_in_200_300 = exam_sizes[exam_sizes["exam"].isin(exams_in_200_300)][
            "size"
        ].sum()
        sizes_in_100_200 = exam_sizes[exam_sizes["exam"].isin(exams_in_100_200)][
            "size"
        ].sum()
        sizes = exam_sizes[exam_sizes["exam"].isin(exams)]["size"].sum()
        h[s - 1] = sizes
        h1[s - 1] = sizes_over_400
        h2[s - 1] = sizes_in_300_400
        h3[s - 1] = sizes_in_200_300
        h4[s - 1] = sizes_in_100_200

    plt.style.use("classic")
    plt.figure(figsize=(18, 12))

    # plt.bar(x=slots, height=[max(h)]*num_slots1, color='red', alpha=0.4, width = 1, align = 'center')
    plt.bar(
        x=range(1, num_slots2 + 1),
        height=h1,
        align="center",
        width=1,
        color="tab:red",
        label="Exams w/ over 400 students",
    )
    plt.bar(
        x=range(1, num_slots2 + 1),
        height=h2,
        align="center",
        width=1,
        bottom=h1,
        color="tab:orange",
        label="Exams w/ over 300 but less than 400 students",
    )
    plt.bar(
        x=range(1, num_slots2 + 1),
        height=h3,
        align="center",
        width=1,
        bottom=h1 + h2,
        color="gold",
        label="Exams w/ over 200 but less than 300 students",
    )
    plt.bar(
        x=range(1, num_slots2 + 1),
        height=h4,
        align="center",
        width=1,
        bottom=h1 + h2 + h3,
        color="pink",
        label="Exams w/ over 100 but less than 200 students",
    )

    plt.bar(
        x=range(1, num_slots2 + 1),
        height=h - h1 - h2 - h3 - h4,
        align="center",
        bottom=h1 + h2 + h3 + h4,
        width=1,
        color="tab:purple",
        label="Other Exams",
    )

    plt.xlabel("Times", fontsize=20)
    plt.xticks(
        np.arange(1, num_slots2 + 1),
        slots_to_time(np.arange(1, num_slots2 + 1)),
        rotation=90,
        fontsize=16,
    )
    plt.yticks(fontsize=16)
    plt.ylabel("Number of students", fontsize=20)
    plt.title("Number of students taking an exam in each time slot", fontsize=25)
    plt.legend(loc="best", fontsize=14)
    plt.savefig(UI_PATH + name + ".png")

    # plt.show()


def last_day(sched_name, save_name):
    # goop['Exam Block'] =
    # sched, by_student_block = normalize_and_merge(goop,)
    sched = pd.read_csv(SAVE_PATH + "/schedules/" + sched_name)
    print(sched)
    enrl_df = pd.read_csv(DATA_PATH + "/enrl.csv")
    enrl_df = enrl_df.merge(sched, left_on="Exam Key", right_on="Exam Group")
    by_student_block = (
        enrl_df.groupby("anon-netid")["slot"].apply(list).reset_index(name="slots")
    )  # create_by_student_slot_df(exam_df, schedule_dict)
    by_student_block["last_block"] = (
        by_student_block["slots"].apply(lambda x: max(x)).copy()
    )
    last_block_counts = by_student_block["last_block"].value_counts().reset_index()
    last_block_counts.columns = ["last_block", "occurrences"]

    last_block_counts = last_block_counts.sort_values(by="last_block").reset_index(
        drop=True
    )
    print("last_block_counts", last_block_counts)

    slots = np.unique(sched["slot"].values)
    # Ensure num_slots2 is an integer for range function
    num_slots2 = int(max(slots)) if len(slots) > 0 else 0

    print("slot , ", slots)
    h = np.zeros(num_slots2)

    # Convert last_block_counts to a dictionary for efficient lookup
    counts_dict = last_block_counts.set_index("last_block")["occurrences"].to_dict()

    for s in range(1, num_slots2 + 1):  # Iterate through all possible slot numbers
        # Get the occurrence count from the dictionary, defaulting to 0 if not found
        h[s - 1] = counts_dict.get(float(s), 0)

    plt.style.use("classic")
    plt.figure(figsize=(18, 12))
    plt.bar(x=range(1, num_slots2 + 1), height=h, align="center", width=1, color="pink")

    plt.xlabel("Times", fontsize=20)
    # Ensure the ticks cover all possible slots up to num_slots2
    plt.xticks(
        np.arange(1, num_slots2 + 1),
        slots_to_time(np.arange(1, num_slots2 + 1)),
        rotation=90,
        fontsize=16,
    )
    plt.yticks(fontsize=16)
    plt.ylabel("Number of students", fontsize=20)
    plt.title(
        "Number of students taking their last exam in each time slot", fontsize=25
    )
    plt.savefig(UI_PATH + save_name + "_dist.png")
    # plt.show()


# import os
# import glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from datetime import datetime
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# mport glob
# mport os
# mport itertools
# from datetime import datetime
# import re
# import matplotlib.dates as mdates
# import torch
# import sys

# exact hypervolume from pygmo
# import pygmo as pg

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# import matplotlib.dates as mdates  # Add this import

current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (e.g., /home/asj53/BOScheduling)
project_root = os.path.dirname(current_dir)

# if project_root not in sys.path:
#    sys.path.insert(0, project_root)
# --- End of path modification ---

# Now the import should work because 'BOScheduling' is in sys.path
# from utils.helpers import custom_utility


# import re
# from datetime import datetime
# import os, sys

# insert the package root (one folder up) at the front of sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def pareto_frontier(df, x_col, y_col):
    # Retrieve the relevant data points
    data = df[[x_col, y_col]].values
    # Sort indices based on the x-axis values (ascending)
    sorted_idx = np.argsort(data[:, 0])
    frontier_indices = []
    current_best = np.inf  # initialize as infinity
    # For each point in order of increasing x, update the frontier
    for i in sorted_idx:
        # Since lower y is better, if a point has a y-value lower than current_best,
        # then it is on the frontier.
        if data[i, 1] < current_best:
            frontier_indices.append(i)
            current_best = data[i, 1]
    return np.array(frontier_indices)


def plot_pairwise():
    # 1. Read all CSV files with the "paretoMOUCB" prefix from the directory.
    csv_path = "/home/asj53/BOScheduling/results/metrics"
    model = "pareto_MOBO"
    pattern = os.path.join(csv_path, "*" + model + "*.csv")

    # Use valid timestamps
    start_ts = datetime.strptime("20250416_230925", "%Y%m%d_%H%M%S")
    end_ts = datetime.strptime("20250417_040649", "%Y%m%d_%H%M%S")
    all_files = sorted(glob.glob(pattern))
    csv_files = []

    for f in all_files:
        basename = os.path.basename(f)
        # Match timestamps at the start of the filename: e.g., 20250415_144801
        match = re.match(r"^(\d{8}_\d{6})", basename)
        if match:
            timestamp_str = match.group(1)
            try:
                file_ts = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                if start_ts <= file_ts <= end_ts:
                    csv_files.append(f)
            except Exception as e:
                print(f"Skipping file due to timestamp parse error: {f}, error: {e}")
        else:
            print(f"Skipping file due to no timestamp at beginning: {f}")

    dataframes = []
    for file in csv_files:
        print("Reading file:", file)  # Optional: print file names for verification.
        df_temp = pd.read_csv(file)
        dataframes.append(df_temp)

    # Combine all data into one DataFrame
    df = pd.concat(dataframes, ignore_index=True)

    # 3. Create output directory for the plots
    output_dir = "pareto_plots"
    os.makedirs(output_dir, exist_ok=True)

    # 4. Generate pairwise scatter plots with the Pareto frontier highlighted.
    # Get a list of column names (metrics)
    columns = df.columns.tolist()

    # Loop over all unique pairs of numeric columns.
    for col_x, col_y in itertools.combinations(columns, 2):
        if pd.api.types.is_numeric_dtype(df[col_x]) and pd.api.types.is_numeric_dtype(
            df[col_y]
        ):
            plt.figure()
            # Plot all data points
            plt.scatter(df[col_x], df[col_y], alpha=0.5, label="Data points")

            # Compute the indices of Pareto-optimal points for the pair of metrics.
            frontier_indices = pareto_frontier(df, col_x, col_y)
            frontier_points = df.iloc[frontier_indices].sort_values(by=col_x)

            # Overlay the Pareto frontier.
            plt.plot(
                frontier_points[col_x],
                frontier_points[col_y],
                color="red",
                linewidth=2,
                marker="o",
                label="Pareto frontier",
            )

            plt.xlabel(col_x)
            plt.ylabel(col_y)
            plt.title(f"{col_x} vs {col_y} with Pareto Frontier")
            plt.legend()

            # Sanitize the column names for use in the file name.
            fname = (
                f"{model}plot_{col_x}_{col_y}.png".replace(" ", "_")
                .replace(",", "")
                .replace("/", "_")
            )
            plt.savefig(os.path.join(output_dir, fname))
            plt.close()

    print("Plots created in the folder:", output_dir)


def hypervolume():
    """
    Scan metric CSVs in a time window, compute the non-dominated front hypervolume progression,
    and plot raw, normalized, log10, percent-gain, and Pareto-set size over time.
    """
    # ---- Configuration ----
    base_dir = "/home/asj53/BOScheduling/results/metrics"
    model = "carbo"
    pattern = os.path.join(base_dir, f"*{model}*.csv")
    # start_ts   = datetime.strptime("20250419_001725", "%Y%m%d_%H%M%S") #MOUCB
    # end_ts     = datetime.strptime("20250420_193422", "%Y%m%d_%H%M%S")
    # start_ts   = datetime.strptime("20250419_162925", "%Y%m%d_%H%M%S") #EHVI
    # end_ts     = datetime.strptime("20250419_164458", "%Y%m%d_%H%M%S")
    start_ts = datetime.strptime("20250517_003558", "%Y%m%d_%H%M%S")  # MOBO
    end_ts = datetime.strptime("20250518_220741", "%Y%m%d_%H%M%S")
    # start_ts   = datetime.strptime("20250415_114955", "%Y%m%d_%H%M%S") #RANDOM
    # end_ts     = datetime.strptime("20250415_204003", "%Y%m%d_%H%M%S")
    ref_point = np.array(
        [1, 10, 30, 30, 100, 100, 1000, 2000, 2000, 5000, 2000, 2000, 20, 60000],
        dtype=float,
    )
    objectives = [
        "conflicts",
        "quints",
        "quads",
        "four in five slots",
        "triple in 24h (no gaps)",
        "triple in same day (no gaps)",
        "three in four slots",
        "evening/morning b2b",
        "other b2b",
        "two in three slots",
        "singular late exam",
        "two exams, large gap",
        "avg_max",
        "lateness",
    ]

    # ---- Gather CSV files once per timestamp ----
    seen = set()
    timed_csv = []
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        m = re.match(r"^(\d{8}_\d{6})", fname)
        if not m:
            print(f"Skipping (no ts): {path}")
            continue
        ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
        if not (start_ts <= ts <= end_ts) or ts in seen:
            continue
        seen.add(ts)
        timed_csv.append((ts, path))

    # ---- Compute HV progression ----
    records = []
    full_vol = np.prod(ref_point)
    prev_norm = None
    all_pts = np.empty((0, len(objectives)), dtype=float)

    for ts, path in timed_csv:
        df = pd.read_csv(path)
        cols = [c for c in objectives if c in df.columns]
        pts = df[cols].dropna().values
        if pts.size == 0:
            continue

        # accumulate and dedupe
        all_pts = np.vstack((all_pts, pts))
        all_pts = np.unique(all_pts, axis=0)

        # filter non-dominated front
        front = np.array(
            [
                p
                for p in all_pts
                if not any(
                    (np.all(q <= p) and np.any(q < p))
                    for q in all_pts
                    if not np.array_equal(q, p)
                )
            ]
        )
        if front.size == 0:
            continue

        # adjust reference point if necessary
        max_f = front.max(axis=0)
        mask = ref_point[: front.shape[1]] <= max_f
        if mask.any():
            ref_point[mask] = max_f[mask] * 1.1

        # compute hypervolume
        hv_calc = pg.hypervolume(front.tolist())
        raw_hv = hv_calc.compute(ref_point[: front.shape[1]].tolist())
        norm_hv = raw_hv / full_vol
        log_hv = np.log10(norm_hv + 1e-30)
        pct_gain = (
            0.0 if prev_norm in (None, 0) else (norm_hv - prev_norm) / prev_norm * 100
        )
        prev_norm = norm_hv

        records.append(
            {
                "timestamp": ts,
                "raw_hv": raw_hv,
                "norm_hv": norm_hv,
                "log10_hv": log_hv,
                "pct_gain": pct_gain,
                "pareto_size": front.shape[0],
            }
        )

        print(
            f"{ts} → rawHV={raw_hv:.3e}, normHV={norm_hv:.3e}, "
            f"logHV={log_hv:.3f}, Δ%={pct_gain:.2f}%, |PF|={front.shape[0]}"
        )

    # ---- Build DataFrame and save ----
    print("RESULTS  : ", records)

    results = pd.DataFrame(records).sort_values("timestamp")
    results.to_csv(f"hypervol_progression_{model}.csv", index=False)

    # ---- Plot progression ----
    fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
    x = pd.to_datetime(results["timestamp"])

    axes[0].plot(x, results["raw_hv"], "-o")
    axes[0].set_title("Raw Hypervolume")
    axes[1].plot(x, results["norm_hv"], "-o")
    axes[1].set_title("Normalized Hypervolume")
    axes[2].plot(x, results["log10_hv"], "-o")
    axes[2].set_title("log₁₀(Normalized HV)")
    axes[3].plot(x, results["pct_gain"], "-o")
    axes[3].set_title("Δ% Normalized HV")
    axes[4].plot(x, results["pareto_size"], "-o")
    axes[4].set_title("Pareto‑set Size")

    for ax in axes:
        ax.grid(True)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    plt.tight_layout()
    plt.savefig(f"hypervol_progression_{model}.png")
    plt.show()


# hypervolume()


def combined_graph():
    """
    Reads all 'hypervolume_progression_*.csv' files, normalizes column names,
    assigns iteration numbers, and plots normalized hypervolume vs. iteration
    for each model.
    """
    # Find all hypervolume CSVs
    csv_files = glob.glob("hypervol_progression_*.csv")
    if not csv_files:
        raise FileNotFoundError(
            "No 'hypervo_progression_*.csv' files found in the current directory."
        )

    hv_data = {}
    for path in csv_files:
        df = pd.read_csv(path)
        model = (
            os.path.basename(path)
            .replace("hypervolume_progression_", "")
            .replace(".csv", "")
        )

        # Determine normalized HV column name
        if "norm_hv" in df.columns:
            col = "norm_hv"
        elif "normalized_hypervolume" in df.columns:
            col = "normalized_hypervolume"
        else:
            raise KeyError(
                f"No normalized hypervolume column found in {path}. Available columns: {df.columns.tolist()}"
            )

        # Sort by timestamp if exists, else by index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        # Assign iteration
        df["iteration"] = np.arange(1, len(df) + 1)
        hv_data[model] = df
        print(hv_data)
    # Plot

    plt.figure(figsize=(10, 6))
    for model, df in sorted(hv_data.items()):
        plt.plot(df["iteration"], df[col], "-o", label=model)

    plt.xlabel("Iteration")
    plt.ylabel("Normalized Hypervolume")
    plt.title("Normalized Hypervolume Progression by Iteration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"comparison.png")
    plt.show()


def plot_best_custom_utility():
    """
    Scan metric CSVs within a specified time window, calculate the best-so-far
    custom utility score at each timestamp, and plot its progression.
    """
    # === CONFIGURATION ===
    csv_path = "/home/asj53/BOScheduling/results/metrics"  # Adjust if needed
    model = "EHVI"  # Specify the model identifier for file filtering
    pattern = os.path.join(csv_path, f"*{model}*.csv")

    # --- Timestamp Filtering Configuration ---
    # Define the time window for analysis
    # Example: Use the same timestamps as in the hypervolume example, adjust as needed
    # start_ts = datetime.strptime("20250417_141725", "%Y%m%d_%H%M%S") # Example: MOUCB start
    # end_ts   = datetime.strptime("20250420_193422", "%Y%m%d_%H%M%S") # Example: MOUCB end
    # Or use a different range relevant to the 'pairwise' model data
    # start_ts = datetime.strptime("20250419_130000", "%Y%m%d_%H%M%S") # PAIRWISE
    # end_ts   = datetime.strptime("20251231_235959", "%Y%m%d_%H%M%S") # Example end (wide range)
    # start_ts   = datetime.strptime("20250419_162925", "%Y%m%d_%H%M%S") #EHVI
    # end_ts     = datetime.strptime("20250419_164458", "%Y%m%d_%H%M%S")
    # start_ts   = datetime.strptime("20250418_213436", "%Y%m%d_%H%M%S") #MOBO
    # end_ts     = datetime.strptime("20250418_214003", "%Y%m%d_%H%M%S")
    start_ts = datetime.strptime("20250418_003436", "%Y%m%d_%H%M%S")  #
    end_ts = datetime.strptime("20250418_234003", "%Y%m%d_%H%M%S")

    objective_cols = [
        "conflicts",
        "quints",
        "quads",
        "four in five slots",
        "triple in 24h (no gaps)",
        "triple in same day (no gaps)",
        "three in four slots",
        "evening/morning b2b",
        "other b2b",
        "two in three slots",
        "singular late exam",
        "two exams, large gap",
        "avg_max",
        "lateness",
    ]
    output_dir = "utility_plots"  # Directory to save the plot
    os.makedirs(output_dir, exist_ok=True)

    # --- Gather and filter files by timestamp (like hypervolume) ---
    seen_ts = set()
    timed_csv_files = []
    all_found_files = sorted(glob.glob(pattern))  # Get all matching files first

    print(f"Scanning for files matching pattern: {pattern}")
    print(f"Filtering between {start_ts} and {end_ts}")

    for path in all_found_files:
        fname = os.path.basename(path)
        # Use regex to match timestamp at the beginning of the filename
        match = re.match(r"^(\d{8}_\d{6})", fname)
        if not match:
            # print(f"Skipping (no timestamp match): {fname}")
            continue

        timestamp_str = match.group(1)
        try:
            ts = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError as e:
            print(f"Skipping (timestamp parse error): {fname}, error: {e}")
            continue

        # Apply the time window filter and check for duplicate timestamps
        if not (start_ts <= ts <= end_ts):
            # print(f"Skipping (out of time range): {fname} ({ts})")
            continue
        if ts in seen_ts:
            # print(f"Skipping (duplicate timestamp): {fname} ({ts})")
            continue

        seen_ts.add(ts)
        timed_csv_files.append((ts, path))
        # print(f"Included file: {fname} ({ts})") # Optional: Verbose logging

    # Sort the filtered files by timestamp
    timed_csv_files.sort()

    if not timed_csv_files:
        print(
            f"No CSV files found for model '{model}' within the specified time range."
        )
        return

    print(f"Found and filtered {len(timed_csv_files)} files for processing.")

    # --- Tracking best utility ---
    timestamps = []
    best_utils = []
    best_so_far = float("-inf")  # Initialize with negative infinity for maximization

    for i, (ts, path) in enumerate(timed_csv_files):
        print(f"Processing file {i+1}/{len(timed_csv_files)}: {os.path.basename(path)}")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  Error reading CSV {path}: {e}. Skipping.")
            continue

        # Check if all required objective columns are present
        missing_cols = [col for col in objective_cols if col not in df.columns]
        if missing_cols:
            print(
                f"  Skipping file {os.path.basename(path)} due to missing columns: {missing_cols}"
            )
            continue

        # Ensure data is numeric and handle potential NaNs
        try:
            # Select only the objective columns and drop rows with any NaNs in these columns
            valid_rows = df[objective_cols].dropna()
            if valid_rows.empty:
                print(
                    f"  Skipping file {os.path.basename(path)} as it has no valid (non-NaN) objective rows."
                )
                continue
            Y = torch.tensor(valid_rows[objective_cols].values, dtype=torch.float)
        except Exception as e:
            print(
                f"  Error converting data to tensor in {os.path.basename(path)}: {e}. Skipping."
            )
            continue

        # Apply custom utility function
        if Y.numel() == 0:  # Check if tensor is empty after potential filtering
            print(
                f"  Skipping file {os.path.basename(path)} as objective tensor is empty."
            )
            continue

        try:
            utilities = custom_utility(Y)
            if utilities.numel() == 0:
                print(
                    f"  Skipping file {os.path.basename(path)} as utility calculation resulted in empty tensor."
                )
                continue
            max_util = utilities.max().item()
        except Exception as e:
            print(
                f"  Error calculating utility for {os.path.basename(path)}: {e}. Skipping."
            )
            continue

        # Track best-so-far utility
        # Note: We track the best utility found *up to and including* this file's timestamp
        # If multiple files have the same timestamp, the last one processed will set the value for that timestamp
        best_so_far = max(best_so_far, max_util)

        # Store timestamp and the best utility *seen so far*
        timestamps.append(ts)
        best_utils.append(best_so_far)
        # print(f"  Timestamp: {ts}, Max Utility in file: {max_util:.4f}, Best Utility So Far: {best_so_far:.4f}") # Optional logging

    if not timestamps:
        print("No valid utility data could be processed.")
        return

    # --- Plotting ---
    plt.figure(figsize=(12, 7))  # Slightly larger figure
    # Use plot_date for proper time axis handling initially, then format
    plt.plot_date(
        mdates.date2num(timestamps),
        best_utils,
        linestyle="-",
        marker="o",
        markersize=4,
        label=f"Best Utility ('{model}')",
    )

    plt.title(
        f"Best Custom Utility Progression ('{model}')\n({start_ts.strftime('%Y-%m-%d %H:%M')} to {end_ts.strftime('%Y-%m-%d %H:%M')})"
    )
    plt.xlabel("Timestamp")
    plt.ylabel("Best Utility Found (Higher is Better)")
    plt.grid(True, which="major", linestyle="--", linewidth="0.5")
    plt.legend()

    # Improve Ticks and Formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))
    plt.gca().xaxis.set_major_locator(
        plt.MaxNLocator(10)
    )  # Limit number of major ticks
    plt.gcf().autofmt_xdate()  # Auto-rotate date labels

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"best_trip_custom_utility_{model}.png")
    plt.savefig(plot_filename)
    print(f"Plot saved to: {plot_filename}")
    plt.show()


def plot_best_custom_utility_by_iteration():
    """
    Compare custom utility progression across EHVI, EUBO, and pairwise models.
    For EUBO and pairwise, only include CSVs with timestamps after 20250420_000000.
    For EHVI, include all files.
    """

    csv_path = "/home/asj53/BOScheduling/results/metrics"
    models_to_compare = ["EHVI", "EUBO", "pairwise"]
    max_iterations = 30
    output_dir = "utility_plots"
    os.makedirs(output_dir, exist_ok=True)

    objective_cols = [
        "conflicts",
        "quints",
        "quads",
        "four in five slots",
        "triple in 24h (no gaps)",
        "triple in same day (no gaps)",
        "three in four slots",
        "evening/morning b2b",
        "other b2b",
        "two in three slots",
        "singular late exam",
        "two exams, large gap",
        "avg_max",
        "lateness",
    ]

    min_timestamp = datetime.strptime("20250420_000000", "%Y%m%d_%H%M%S")
    all_results = {}

    print(
        f"\n--- Custom Utility Comparison (Timestamp Filter: only EUBO & pairwise) ---"
    )

    for model in models_to_compare:
        print(f"\nProcessing model: {model}")
        pattern = os.path.join(csv_path, f"*{model}*.csv")
        all_files = sorted(glob.glob(pattern))

        # Sort and optionally filter files by timestamp
        def extract_ts(f):
            match = re.search(r"(\d{8}_\d{6})", os.path.basename(f))
            if match:
                return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
            return datetime.min

        if model in ["EUBO", "pairwise"]:
            filtered_files = [f for f in all_files if extract_ts(f) > min_timestamp]
        else:
            filtered_files = all_files  # EHVI gets all files

        filtered_files = sorted(filtered_files, key=extract_ts)[:max_iterations]

        if not filtered_files:
            print(f"  No valid files for model '{model}'.")
            all_results[model] = {"iters": [], "best_utils": []}
            continue

        best_so_far = float("-inf")
        best_utils = []
        iters = []

        for i, path in enumerate(filtered_files):
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"    Error reading {os.path.basename(path)}: {e}")
                continue

            missing_cols = [col for col in objective_cols if col not in df.columns]
            if missing_cols:
                print(f"    Skipping {os.path.basename(path)}: missing {missing_cols}")
                continue

            try:
                valid_rows = df[objective_cols].dropna()
                if valid_rows.empty:
                    continue
                Y = torch.tensor(valid_rows.values, dtype=torch.float)
            except Exception as e:
                print(f"    Error processing {os.path.basename(path)}: {e}")
                continue

            try:
                utilities = custom_utility(Y)
                max_util = (
                    utilities.item() if utilities.ndim == 0 else utilities.max().item()
                )
            except Exception as e:
                print(f"    Utility calc error for {os.path.basename(path)}: {e}")
                continue

            best_so_far = max(best_so_far, max_util)
            iters.append(i)
            best_utils.append(best_so_far)

        all_results[model] = {"iters": iters, "best_utils": best_utils}
        print(
            f"  {model}: {len(best_utils)} iterations processed, final best = {best_so_far:.4f}"
        )

    # --- PLOT ---
    print("\n--- Plotting ---")
    plt.figure(figsize=(12, 6))
    colors = {"EHVI": "red", "EUBO": "blue", "pairwise": "purple"}

    for model, data in all_results.items():
        if not data["iters"]:
            continue
        plt.plot(
            data["iters"],
            data["best_utils"],
            label=model,
            color=colors[model],
            marker="o",
        )

    plt.title("Best Custom Utility Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Best Utility Found So Far")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(title="Model")
    plt.xticks(ticks=list(range(max_iterations)))
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    save_path = os.path.join(
        output_dir, "best_custom_utility_comparison_by_iteration.png"
    )
    plt.savefig(save_path)
    print(f"\nPlot saved to: {save_path}")
    plt.show()


def find_better():

    # === CONFIGURATION ===
    """average maximum is: 14.259388379204893
    conflicts: 1
    quints: 0
    quads: 3
    triple in 24h (no gaps): 53
    triple in same day (no gaps): 27
    four in five slots: 16
    three in four slots: 574
    evening/morning b2b: 389
    other b2b: 1674
    two in three slots: 3262
    singular late exam count:  543
    two exams, large gap:  448

    conflicts,quints,quads,four in five slots,triple in 24h (no gaps),triple in same day (no gaps),three in four slots,evening/morning b2b,other b2b,two in three slots,singular late exam,"two exams, large gap",avg_max,lateness
    1,0,2,5,29,40,310,460,1045,3685,1008,709,16.684892966360856,44904

    """
    directory = "/home/asj53/BOScheduling/results/metrics/"
    reference = [1, 0, 3, 23, 53, 27, 574, 389, 1674, 3262, 543, 448, 14.3]
    n_metrics = len(reference)
    threshold = 11  # minimum number of metrics where candidate must be strictly lower

    # === RESULTS ===
    results = []

    # === ITERATE THROUGH CSV FILES ===
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            try:
                df = pd.read_csv(filepath)
                if df.shape[0] > 0:
                    row = df.iloc[0].tolist()[:n_metrics]  # Only take first 12 columns
                    count_lower = sum([a <= b for a, b in zip(row, reference)])
                    if count_lower >= threshold:
                        results.append(
                            {
                                "file": filename,
                                "num_lower_metrics": count_lower,
                                "values": row,
                            }
                        )
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    # === DISPLAY RESULTS ===
    results = sorted(
        results, key=lambda x: -x["num_lower_metrics"]
    )  # Sort by best dominance
    for res in results:
        print(
            f"{res['file']} | Lower in {res['num_lower_metrics']} metrics | Values: {res['values']}"
        )


# find_better()

# print(pd.read_csv('/home/asj53/BOScheduling/results/schedules/20250418_213833_pareto_MOBOparam_026.73param_130.84param_23.27param_34.47param_43.46param_51623.15param_680.02param_750.92param_8316.97param_9164.24.csv'))


def pca():

    import os
    import pandas as pd
    import numpy as np

    # from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # === PATH TO YOUR METRICS DIRECTORY ===
    metrics_path = "/home/asj53/BOScheduling/results/metrics"

    # === CATEGORY LABELS AND COLORS ===
    label_keywords = {
        "pareto_MOBO": "MOBO",
        "pareto_EHVI": "EHVI",
        "pareto_MOUCB": "MOUCB",
    }
    colors = {
        "MOBO": "blue",
        "EHVI": "orange",
        "MOUCB": "green",
    }

    # === LOAD AND CATEGORIZE CSV FILES ===
    all_data = []
    labels = []

    for filename in os.listdir(metrics_path):
        if not filename.endswith(".csv"):
            continue

        filepath = os.path.join(metrics_path, filename)
        df = pd.read_csv(filepath)

        # Drop 'lateness' if it's in the last column
        if df.columns[-1].lower().strip() == "lateness":
            df = df.iloc[:, :-1]

        # Check which category the file belongs to
        matched = False
        for keyword, label in label_keywords.items():
            if keyword in filename:
                all_data.append(df.values)
                labels.extend([label] * len(df))
                matched = True
                break

        if not matched:
            print(f"Skipped: {filename}")

    # === STACK ALL ROWS TO FORM THE DATA MATRIX ===
    if len(all_data) == 0:
        raise ValueError("No data found. Please check the path and filenames.")

    X = np.vstack(all_data)

    # === RUN PCA ===
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # === PLOT RESULTS ===
    plt.figure(figsize=(10, 6))
    for label in label_keywords.values():
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(
            X_pca[idx, 0], X_pca[idx, 1], label=label, color=colors[label], alpha=0.7
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of Schedule Metrics by Optimization Strategy")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.savefig("PCA.png")


import glob
import os

# Directory where ycaour CSVs are generated
directory = "/Users/adamshafikjovine/Documents/BOScheduling/results/sp25/schedules"
pattern = os.path.join(directory, "*.csv")

# Find all CSVs
all_paths = glob.glob(pattern)

# Filter out any you don’t want
filtered_paths = [p for p in all_paths if not p.endswith("_temp.csv")]

# Now just grab the base-names
filtered_files = [os.path.basename(p) for p in filtered_paths]

print(filtered_files)
for name, path in zip(filtered_files, filtered_paths):
    try:
        print("f", name)
        # pass full path to get_plot/last_day, but use 'name' for labeling/output
        get_plot(name, name)
        last_day(name, name)
    except Exception as e:
        print(f"Error processing {name}: {e}")
        continue
