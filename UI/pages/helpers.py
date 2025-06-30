import os
import glob
import re
import pandas as pd
import streamlit as st
import datetime 
import subprocess
import json
from config.settings import SAVE_PATH, UI_PATH, NUM_SLOTS
from UI.pages.schedule_plots import get_plot, last_day

def run_one_iteration():
    # save preferences
    prefs_path = os.path.join(SAVE_PATH, "pending_prefs.json")
    with open(prefs_path, "w") as f:
        json.dump(st.session_state.prefs, f)
    # build SLURM script
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

python -u run_EIUU.py \
    --prefs {prefs_path} \
    --n_iterations 1
"""
    script_path = os.path.join(SAVE_PATH, "submit_one_iter.slurm")
    with open(script_path, "w") as f:
        f.write(slurm_txt)
    res = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if res.returncode == 0:
        st.success(f"Submitted SLURM job:\n{res.stdout.strip()}")
    else:
        st.error(f"Error submitting job:\n{res.stderr.strip()}")
    st.session_state.prefs = []
    st.session_state.last_run = datetime.datetime.utcnow()
    st.stop()


def get_schedule_files(date_prefix: str, metrics_dir: str = None) -> list[str]:
    """
    Discover schedule CSV filenames (not full paths) matching a given date prefix.

    Args:
        date_prefix: Date string prefix (e.g., '20250617').
        metrics_dir: Directory under SAVE_PATH for metrics. Defaults to SAVE_PATH/metrics.

    Returns:
        List of CSV filenames matching the prefix.
    """
    metrics_dir = metrics_dir or os.path.join(SAVE_PATH, 'metrics')
    pattern = os.path.join(metrics_dir, f"{date_prefix}*.csv")
    return [os.path.basename(p) for p in glob.glob(pattern)]


def extract_i_number(path: str) -> int:
    """
    Extract the first integer immediately following 'i' in a filename.

    Raises:
        ValueError if no such pattern is found.
    """
    fname = os.path.basename(path)
    m = re.search(r"i(\d+)", fname)
    if not m:
        raise ValueError(f"No 'i<digits>' segment found in {path!r}")
    return int(m.group(1))


@st.cache_data
def generate_plots_for_files(filenames: list[str], save_path: str = SAVE_PATH, ui_path: str = UI_PATH):
    """
    Ensure schedule and distribution plots exist for each metrics file.
    """
    for fname in filenames:
        try: 
            base, _ = os.path.splitext(fname)
            for func, suf in [(get_plot, ''), (last_day, '_dist')]:
                img_path = os.path.join(ui_path, f"{base}{suf}.png")
                if not os.path.exists(img_path):
                    func(fname, base)
        except: 
            print('MISSING, ' , fname )
@st.cache_data
def load_schedule_data_basic(filenames: list[str], save_path: str = SAVE_PATH) -> list[dict]:
    """
    Load schedule metrics and slot counts for a basic scheduling page.

    Returns list of dicts:
        name: display name,
        basename: file base name,
        metrics: first-row metrics (first 17 cols),
        columns: slot presence mapping,
        max_slot: highest slot number.
    """
    # path for cached aggregate
    cache_csv = os.path.join(save_path, 'cached_data.csv')
    param_cols = ['size_cutoff','reserved','num_blocks', 'large_block_size','large_exam_weight','large_block_weight','large_size_1','large_cutoff_freedom']
    metrics_cols = ['conflicts','quints','quads','four in five slots','three in four slots','two in three slots','singular late exam',"two exams, large gap",'avg_max']
    computed_cols = ['reschedules', 'back_to_back']
    # If cache exists, read and reconstruct
    if os.path.exists(cache_csv):
        data = []
        # slot columns prefixed with 'slot_'
        df_metrics = pd.read_csv(cache_csv)
        #print(df_metrics.columns)
        all_cols = set(df_metrics.columns) - {'name', 'basename', 'max_slot'}
        slot_cols = sorted([c for c in all_cols if c.startswith('slot_')])
        df_metric = df_metrics[ computed_cols + metrics_cols ] 
        first_row = df_metric.iloc[0]
       
        #df_metric['reschedules'] = df_metrics.loc['triple in 24h (no gaps)'] + df_metrics.loc['triple in same day (no gaps)']
        #df_metric['back_to_back'] = df_metrics.loc['evening/morning b2b'] + df_metrics.loc['other b2b']
        df_params = df_metrics[param_cols]
        
        for i, row in df_metrics.iterrows():
            #print('in loop ' , i )
            metrics = {k: row[k] for k in metrics_cols}
            params = {k: row[k] for k in param_cols }
            columns = {int(k.split('_')[1]): int(row[k]) for k in slot_cols}
            data.append({
                'name': row['name'],
                'basename': row['basename'],
                'metrics': metrics,
                'params' : params, 
                'columns': {str(slot): columns.get(slot, 0) for slot in range(1, NUM_SLOTS+1)},
                'max_slot': int(row['max_slot']),
            })
        return data

    # Otherwise, load fresh and build cache
    data = []
     #try: 
    for fname in filenames:
        base, _ = os.path.splitext(fname)
        #try: 
        idx = extract_i_number(base)
        display = f"Schedule {idx}"

        # metrics
        df_metrics = pd.read_csv(os.path.join(save_path, 'metrics', fname))
        print('df_metrics' , df_metrics)
        # large_block_size,large_exam_weight,large_block_weight,large_size_1,large_cutoff_freedom
        # conflicts,quints,quads,four in five slots,triple in 24h (no gaps),triple in same day (no gaps),three in four slots,evening/morning b2b,other b2b,two in three slots,singular late exam,"two exams, large gap",avg_max,lateness, 
        df_metric = df_metrics[metrics_cols]
        df_params = df_metrics[param_cols]
    
        df_metric['reschedules'] = df_metrics['triple in 24h (no gaps)'] + df_metrics['triple in same day (no gaps)']
        df_metric['back_to_back'] = df_metrics['evening/morning b2b'] + df_metrics['other b2b']
        first_row = df_metric.iloc[0]
        #first_row['reschedules'] = df_metrics['conflicts' , '']
        print('first _rw' , first_row)
        params  =df_params.iloc[0]
        # schedule slots
        df_sched = pd.read_csv(os.path.join(save_path, 'schedules', fname))
        slots = sorted(df_sched['slot'].unique())
        max_slot = max(slots, default=0)

        data.append({
            'name': display,
            'basename': base,
            'metrics': first_row.to_dict(),
            'params' : params.to_dict(), 
            'columns': {str(i): (1 if i in slots else 0) for i in range(1, NUM_SLOTS+1)},
            'max_slot': max_slot,
        })
        #except: 
        #   print('file bad  ' , fname)

    # Flatten and save to CSV for next runs
    rows = []
    # collect keys
    metric_keys = list(data[0]['metrics'].keys()) if data else []
    params_keys = list(data[0]['params'].keys()) if data else []
    print('param_keys' , params_keys)
    for s in data:
        row = {'name': s['name'], 'basename': s['basename'], 'max_slot': s['max_slot']}
        # metrics
        for k in metric_keys:
            row[k] = s['metrics'][k]
        # params
        for p in params_keys: 
            row[p] = s['params'][p]
        # slots
        for slot, val in s['columns'].items():
            row[f'slot_{slot}'] = val
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(cache_csv, index=False)

    return data
     #except: 
     #    print(' EEEE E ')
     #    return []


@st.cache_data
def load_schedule_data_compare(paths: list[str]) -> list[dict]:
    """
    Load and enrich schedule metrics for comparison pages.

    Returns list of dicts with basename, display_name, and metrics dict.
    """
    result = []
    for p in paths:
        fname = os.path.basename(p)
        base, _ = os.path.splitext(fname)
        idx = extract_i_number(p)
        display = f"Schedule {idx}"
        df = pd.read_csv(p)
        df = df[df.columns[:17]]
        m = df.iloc[0].copy()
        m['reschedules'] = m.get('triple in 24h (no gaps)', 0) + m.get('triple in same day (no gaps)', 0)
        m['back_to_back'] = m.get('evening/morning b2b', 0) + m.get('other b2b', 0)
        result.append({
            'basename': base,
            'display_name': display,
            'metrics': m.to_dict(),
        })
    return result


def apply_slot_exclusion(container: st.delta_generator.DeltaGenerator, data: list[dict], num_slots: int = NUM_SLOTS) -> list[dict]:
    """
    Render slot-exclusion checkboxes in the given container and filter schedules accordingly.

    Args:
        container: Streamlit container (e.g., a column or sidebar).
        data: list of schedule dicts with 'columns' mapping slot→0/1.
        num_slots: total number of slots to display.

    Returns:
        Filtered data list.
    """
    filtered = data.copy()
    container.subheader("Exclude Slots")
    cols = container.columns(num_slots, gap="small")
    exclude_flags = {}
    for i, col in enumerate(cols, start=1):
        flag = col.checkbox("", key=f"exclude_{i}")
        col.caption(str(i))
        exclude_flags[i] = flag
    for slot, flag in exclude_flags.items():
        if flag:
            filtered = [s for s in filtered if s['columns'].get(str(slot), 0) == 0]
    return filtered
def apply_numeric_sliders(
    container: st.delta_generator.DeltaGenerator,
    data: list[dict],
    metric_keys: list[str],
    param_keys : list[str],
) -> tuple[list[dict], dict]:
    """
    Render numeric sliders for *metrics* first and then, just below,
    for *parameters* found in the `params` dict of each schedule.

    Returns
    -------
    filtered_data : list[dict]
        Schedules that satisfy **all** chosen thresholds.
    thresholds : dict
        {metric_or_param_name: chosen_threshold}
    """
    # Work on a copy so original order isn’t disturbed
    filtered = data.copy()
    thresholds: dict[str, int | float] = {}

    # ---------- METRIC SLIDERS ----------
    container.subheader("Metric thresholds")
    for key in metric_keys:
        vals = [s["metrics"].get(key, 0) for s in filtered]
        if not vals:
            continue
        lo, hi = int(min(vals)), int(max(vals))
        # slider only makes sense if range > 0
        thresh = container.slider(key, lo, hi, hi) if lo < hi else lo
        thresholds[key] = thresh
        filtered = [s for s in filtered if int(s["metrics"].get(key, 0)) <= thresh]

    # ---------- PARAMETER SLIDERS ----------
    #if param_keys:
    container.subheader("Parameter thresholds")
    print('param_keyyy' , param_keys)
    for key in param_keys:
        vals = [s["params"].get(key, 0) for s in filtered]
        #print('vals' , vals)
        if not vals:
            
            continue
        lo, hi = min(vals), max(vals)
        # cast to int if both ends are integers; leave float otherwise
        is_int_range = isinstance(lo, int) and isinstance(hi, int)
        default = hi
        thresh = container.slider(key, lo, hi, hi) if lo < hi else lo#(
            #container.slider(key, int(lo), int(hi), int(default))
            #if is_int_range and lo < hi
            #else container.slider(key, lo, hi, default)
            #if lo < hi
            #else lo
            #)
        thresholds[key] = thresh
        # Accept parameter ≤ threshold
        filtered = [
            s for s in filtered if s["params"].get(key, 0) <= thresh
        ]

    return filtered, thresholds
def show_schedule_block(
    col: st.delta_generator.DeltaGenerator,
    sched: dict,
    ui_path: str = UI_PATH,
    incumbent: bool = False,
    metrics: bool = True
) -> None:
    """
    Render a schedule's metrics panel and plots in the given column.
    - If metrics=True: 3-column layout (metrics | schedule plot | #done plot)
    - If metrics=False: single column with both plots side by side via st.image(list)
    """
    title = sched.get('display_name', sched.get('name'))
    if incumbent:
        title += " — **Incumbent**"
    col.subheader(title)

    base = sched.get('basename', sched.get('name'))

    # Paths to the two images
    img1 = os.path.join(ui_path, f"{base}.png")
    img2 = os.path.join(ui_path, f"{base}_dist.png")

    # Ensure they're generated if missing
    if not os.path.exists(img1):
        get_plot(f"{base}.csv", base)
    if not os.path.exists(img2):
        last_day(f"{base}.csv", base)

    if metrics:
        # 3-column layout
        mcol, pcol, dcol = col.columns([1.5, 3, 3])

        # Metrics panel
        html = "".join(f"{k}: {v}<br>" for k, v in sched['metrics'].items())
        style = (
            'border:2px solid red; padding:10px; border-radius:5px;'
            if incumbent else
            'background-color:#f0f0f0; padding:10px; border-radius:5px;'
        )
        with mcol:
            st.markdown(f"<div style='{style}'><strong>Metrics</strong><br><br>{html}</div>",
                        unsafe_allow_html=True)

        # Schedule plot
        with pcol:
            if os.path.exists(img1):
                st.image(img1, caption="Schedule plot", use_container_width=True)
            else:
                st.markdown("*(Schedule plot unavailable)*")

        # # done plot
        with dcol:
            if os.path.exists(img2):
                st.image(img2, caption="# done plot", use_container_width=True)
            else:
                st.markdown("*(# done plot unavailable)*")

    else:
        # Just both plots, side by side
        imgs = []
        caps = []
        if os.path.exists(img1):
            imgs.append(img1)
            caps.append("Schedule plot")
        if os.path.exists(img2):
            imgs.append(img2)
            caps.append("# done plot")

        if imgs:
            # Streamlit will render a list of images in a row
            col.image(imgs, caption=caps, use_container_width=True)
        else:
            col.markdown("*(Plots unavailable)*")
