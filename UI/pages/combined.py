import streamlit as st
import threading, random
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

from config.settings import SAVE_PATH
from UI.pages.helpers import (
    get_schedule_files,
    load_schedule_data_basic,
    apply_numeric_sliders,
    show_schedule_block,
    run_one_iteration,
)

st.set_page_config(layout="wide")
st.title("COMPARE SCHEDULES")

def _background_optimize():
    run_one_iteration()
    st.session_state.optimizing = False

# ── WAIT MODE ─────────────────────────────────────────────────────────────────
if st.session_state.get("optimizing", False):
    # Refresh every second
    st_autorefresh(interval=1_000, key="timer")

    remaining = st.session_state.timer_end - datetime.now()
    secs = max(int(remaining.total_seconds()), 0)
    mins, sec = divmod(secs, 60)

    st.markdown("## Optimization in progress…")
    st.markdown(f"Please come back in **{mins:02d}:{sec:02d}** (mm:ss).")

    if st.button("Interrupt optimization"):
        st.session_state.optimizing = False

    st.stop()

# ── LOAD & FILTER ─────────────────────────────────────────────────────────────
date_prefix = "20250618"
filenames = get_schedule_files(date_prefix)
schedule_data = load_schedule_data_basic(filenames)

st.sidebar.header("Filter Schedules")
metric_keys = list(schedule_data[0]["metrics"].keys()) if schedule_data else []
filtered_schedules, thresholds = apply_numeric_sliders(
    container=st.sidebar,
    data=schedule_data,
    metric_keys=metric_keys,
)
if len(filtered_schedules) < 2:
    st.sidebar.warning("Need at least 2 schedules to compare. Adjust filters.")

# ── SESSION STATE SETUP ───────────────────────────────────────────────────────
names = [s["name"] for s in filtered_schedules]
for k, default in {"base": None, "challenger": None, "prefs": []}.items():
    st.session_state.setdefault(k, default)

# ── PAIRWISE COMPARISON ────────────────────────────────────────────────────────
if len(names) >= 2:
    if st.session_state.base not in names:
        st.session_state.base = names[0]
    pool = [n for n in names if n != st.session_state.base]
    if pool and st.session_state.challenger not in pool:
        st.session_state.challenger = pool[0]

    left = st.selectbox("First schedule", names, key="base")
    right_pool = [n for n in names if n != left]
    right = st.selectbox("Second schedule", right_pool, key="challenger")

    c1, c2 = st.columns(2)
    sl = next(s for s in filtered_schedules if s["name"] == left)
    sr = next(s for s in filtered_schedules if s["name"] == right)
    show_schedule_block(c1, sl, incumbent=True,  metrics=True)
    show_schedule_block(c2, sr, incumbent=False, metrics=True)

    st.radio("Which do you prefer?",
             options=[left, right, "No preference"],
             key="pref_choice")

    def _record_pref():
        choice = st.session_state.pref_choice
        if not choice:
            return
        if choice in (left, right):
            loser = right if choice == left else left
            wb = next(s["basename"] for s in filtered_schedules if s["name"] == choice)
            lb = next(s["basename"] for s in filtered_schedules if s["name"] == loser)
            st.session_state.prefs.append((wb, lb))
            new_base = choice
        else:
            new_base = left
        st.session_state.base = new_base
        rem = [n for n in names if n != new_base]
        st.session_state.challenger = random.choice(rem) if rem else None
        st.session_state.pref_choice = None

    st.button("Submit Preference", on_click=_record_pref)

# ── BATCH + START OPTIMIZATION ────────────────────────────────────────────────
if st.session_state.prefs:
    st.markdown("---")
    st.info(f"🗒️ Queued preferences: **{len(st.session_state.prefs)}**")
    if st.button("Submit preference list and run one iteration", key="batch"):
        st.session_state.optimizing = True
        st.session_state.timer_end = datetime.now() + timedelta(minutes=30)
        threading.Thread(target=_background_optimize, daemon=True).start()
        # immediately bail so next st_autorefresh sees optimizing=True:
        st.stop()
