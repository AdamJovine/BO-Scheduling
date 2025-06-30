import streamlit as st
import os
import glob
import datetime
import random

import numpy as np
import pandas as pd
import torch

from config.settings import SAVE_PATH, UI_PATH, PARAM_NAMES, LICENSES
from optimization.EIUU import run_preference_ei_parallel, parse_tensor_str
from optimization.CArBO import CostAwareSchedulingOptimizer
from models.sequencing import sequencing
from post_processing.post_processing import run_pp
from block_assignment.layercake import run_layer_cake
from UI.pages.schedule_plots import get_plot, last_day

st.set_page_config(layout="wide")
st.title("COMPARE SCHEDULES")

# ---- Discover schedule files ----
schedule_files = [
    os.path.basename(p)
    for p in glob.glob(os.path.join(SAVE_PATH, "metrics", "20250618*.csv"))
]

# ---- Load & cache schedule metrics ----
@st.cache_data
def load_schedule_data(config_files, last_run):
    data = []
    for fname in config_files:
        name = os.path.splitext(fname)[0]
        df = pd.read_csv(os.path.join(SAVE_PATH, "metrics", fname))
        df = df[df.columns[:14]]
        m = df.iloc[0].copy()
        m['reschedules'] = m['triple in 24h (no gaps)'] + m['triple in same day (no gaps)']
        m['back_to_back'] = m['evening/morning b2b'] + m['other b2b']
        data.append({"name": name, "metrics": m.to_dict()})
    return data

# ---- Data I/O for optimizer ----
def load_current_data():
    paths = glob.glob(os.path.join(SAVE_PATH, "metrics", "*.csv"))
    if not paths:
        raise RuntimeError("No metrics CSVs found!")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    for col in PARAM_NAMES:
        df[col] = df[col].apply(parse_tensor_str).astype(float)
    X_np = df[PARAM_NAMES].to_numpy()
    df['list'] = df.apply(
        lambda r: [
            r["triple in 24h (no gaps)"] + r["triple in same day (no gaps)"],
            r["three in four slots"],
            r["evening/morning b2b"] + r["other b2b"],
            r["two in three slots"],
            r["singular late exam"],
            r["two exams, large gap"],
            r["avg_max"],
            r["lateness"]
        ], axis=1
    )
    Y_np = np.vstack(df['list'].values)
    dev = torch.device("cpu")
    X = torch.tensor(X_np, dtype=torch.double, device=dev)
    Y = torch.tensor(Y_np, dtype=torch.double, device=dev)
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return X, Y, names


def build_tensors_from_prefs(X_init, names, prefs):
    idx = {n: i for i, n in enumerate(names)}
    X_list, comp = [], []
    for k, (w, l) in enumerate(prefs):
        i, j = idx[w], idx[l]
        X_list += [X_init[i].unsqueeze(0), X_init[j].unsqueeze(0)]
        comp.append([2 * k, 2 * k + 1])
    if X_list:
        return torch.cat(X_list, 0), torch.tensor(comp, dtype=torch.long, device=X_init.device)
    return torch.empty((0, X_init.size(1)), device=X_init.device), torch.empty((0, 2), dtype=torch.long, device=X_init.device)


def save_current_data(Xf, Yf, utils):
    metric_names = [
        "triple_in_24h", "three_in_four",
        "back to backs", "two_in_three",
        "singular_late", "two_large_gap", "avg_max", "lateness",
    ]
    pd.concat([
        pd.DataFrame(Xf.cpu().numpy(), columns=PARAM_NAMES),
        pd.DataFrame(Yf.cpu().numpy(), columns=metric_names),
    ], axis=1).assign(utility=utils) \
      .to_csv(os.path.join(SAVE_PATH, "results_with_utility.csv"), index=False)

import subprocess
import json

def run_one_iteration():
    # 1) save preferences so the batch job can see them
    prefs_path = os.path.join(SAVE_PATH, "pending_prefs.json")
    with open(prefs_path, "w") as f:
        json.dump(st.session_state.prefs, f)

    # 2) build a little SLURM submission script
    slurm_txt = f"""#!/bin/bash
#SBATCH -J aEIUU_1iter
#SBATCH -o /home/asj53/aEIUU_%j.out
#SBATCH -e /home/asj53/aEIUU_%j.err
#SBATCH --partition=frazier
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH -t 23:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/asj53/BOScheduling/optimization

set -x
source ~/.bashrc
conda activate research_env

# Pass in the prefs file and override n_iterations
python -u run_EIUU.py \\
    --prefs {prefs_path} \\
    --n_iterations 1
"""
    script_path = os.path.join(SAVE_PATH, "submit_one_iter.slurm")
    with open(script_path, "w") as f:
        f.write(slurm_txt)

    # 3) actually submit it
    res = subprocess.run(
        ["sbatch", script_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0:
        st.success(f"Submitted SLURM job:\n{res.stdout.strip()}")
    else:
        st.error(f"Error submitting job:\n{res.stderr.strip()}")

    # 4) reset state
    st.session_state.prefs = []
    st.session_state.last_run = datetime.datetime.utcnow()
    st.stop()


# ---- Session state defaults ----
for key, default in {
    "base_schedule": None,
    "challenger": None,
    "just_selected": False,
    "prefs": [],
    "last_run": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---- Load schedule_data & helper ----
schedule_data = load_schedule_data(schedule_files, st.session_state.last_run)
names = [d["name"] for d in schedule_data]

def show_schedule(col, sched):
    with col:
        st.subheader(sched["name"])
        txt = "".join(f"{k}: {v}<br>" for k, v in sched["metrics"].items())
        st.markdown(
            f"<div class='gray-box'><strong>Metrics</strong><br><br>{txt}</div>",
            unsafe_allow_html=True
        )
        for suf, cap in [("", "Schedule plot"), ("_dist", "# done plot")]:
            img = os.path.join(UI_PATH, f"{sched['name']}{suf}.png")
            if os.path.exists(img):
                st.image(img, caption=cap, use_container_width=True)
            else:
                if cap == 'Schedule plot':
                    get_plot(f"{sched['name']}.csv", sched['name'])
                else:
                    last_day(f"{sched['name']}.csv", sched['name'])
                st.image(img, caption=cap, use_container_width=True)

# ---- Hourly lockout ----
if st.session_state.last_run:
    next_time = st.session_state.last_run + datetime.timedelta(hours=1)
    if datetime.datetime.utcnow() < next_time:
        mins = int((next_time - datetime.datetime.utcnow()).total_seconds() // 60) + 1
        st.info(f"✅ Last run at {st.session_state.last_run:%H:%M UTC}")
        st.warning(f"Come back in ~**{mins}** min for new comparisons.")
        st.stop()

# ---- Pairwise comparison UI with on_click callback ----
def get_pair():
    #st.write("get_pair: entered")
    if st.session_state.base_schedule is None:
        #st.write("get_pair: initial mode")
        left = st.selectbox("First schedule", names, key="sel1")
        right = st.selectbox(
            "Second schedule", names,
            index=1 if len(names) > 1 else 0,
            key="sel2"
        )
        radio_key, button_key = "pref_init", "btn_init"
        #st.write(f"get_pair: left={left}, right={right}")
    else:
        #st.write("get_pair: ongoing mode")
        base = st.session_state.base_schedule
        pool = [n for n in names if n != base]
        #st.write(f"get_pair: base={base}, pool={pool}")
        if not pool:
            st.info("🏁 You’ve compared all schedules!")
            st.stop()
        if st.session_state.challenger not in pool:
            #st.write("get_pair: picking new challenger")
            st.session_state.challenger = random.choice(pool)
        left, right = base, st.session_state.challenger
        radio_key, button_key = "pref_ongoing", "btn_ongoing"
        if st.session_state.just_selected:
            #st.success(f"You picked **{base}**.")
            st.session_state.just_selected = False
        #st.write(f"get_pair: left={left}, right={right}")
    #st.write(f"get_pair: returning left={left}, right={right}, radio_key={radio_key}, button_key={button_key}")
    return left, right, radio_key, button_key

left, right, radio_key, button_key = get_pair()

# callback for submit button
def submit_pref(left, right, radio_key):
    choice = st.session_state.get(radio_key, "")
    st.write(f"submit_pref: choice={choice}, left={left}, right={right}")
    if choice:
        loser = right if choice == left else left
        st.session_state.prefs.append((choice, loser))
        st.write(f"submit_pref: appended ({choice}, {loser})")
        st.session_state.base_schedule = choice
        st.session_state.just_selected = True
        st.session_state.challenger = None

# display pair and submit button
c1, c2 = st.columns(2)
show_schedule(c1, next(d for d in schedule_data if d["name"] == left))
show_schedule(c2, next(d for d in schedule_data if d["name"] == right))
choice = st.radio("Which do you prefer?", ["", left, right], key=radio_key)
st.button("Submit Preference", key=button_key, on_click=submit_pref, args=(left, right, radio_key))

# ---- Batch-submit at bottom ----
if st.session_state.prefs:
    st.markdown("---")
    st.info(f"🗒️ Queued preferences: **{len(st.session_state.prefs)}**")
    if st.button("Submit preference list and run one iteration", key="batch"):
        run_one_iteration()