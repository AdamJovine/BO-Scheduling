import streamlit as st
import os
import pandas as pd

from config.settings import SAVE_PATH, UI_PATH, NUM_SLOTS
from UI.pages.helpers import (
    get_schedule_files,
    generate_plots_for_files,
    load_schedule_data_basic,
    apply_slot_exclusion,
    apply_numeric_sliders,
    show_schedule_block,
)

# Page config
st.set_page_config(layout="wide")

# ---- Custom CSS ----
st.markdown(
    """
    <style>
        body, .stApp { background-color: #f0f2f6; color: black !important; }
        h1, h2, h3, h4, h5, h6, p, span, div, label { color: black !important; }
        .stButton > button {
            background-color: #e0e0e0 !important;
            color: black !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            font-size: 14px !important;
            vertical-align: middle;
        }
        div[data-baseweb="input"] input {
            color: black !important;
            background-color: white !important;
            border: 1px solid #999 !important;
        }
        div[role="listbox"] { background-color: white !important; color: black !important; }
        .stImage { display: block; margin: 20px auto 0; }
        header[data-testid="stHeader"] { background-color: #333333 !important; color: white !important; }
        .gray-box {
            background-color: #e0e0e0;
            padding: 10px;
            border-radius: 8px;
            font-size: 14px;
        }
        .grey-button .stButton > button {
            background-color: #6c757d !important;
            color: white !important;
            border: 2px solid #6c757d !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Unique schedule identifier ----
schedule_id = "20250624_185106"

# ---- One-time loading into session_state ----
if "schedule_data" not in st.session_state:
    # 1) find all metrics files
    files = get_schedule_files(schedule_id)
    # 2) generate any missing plots
    generate_plots_for_files(files)
    # 3) load metrics + slot data
    data = load_schedule_data_basic(files)
    # 4) merge max_slot into metrics dict
    for s in data:
        s["metrics"]["max_slot"] = s.pop("max_slot")
    # store
    st.session_state.schedule_files = files
    st.session_state.schedule_data = data

# retrieve from session_state
schedule_files = st.session_state.schedule_files
schedule_data  = st.session_state.schedule_data

# ---- Likes CSV ----
likes_path = os.path.join(SAVE_PATH, "like.csv")
if os.path.exists(likes_path):
    likes_df = pd.read_csv(likes_path)
else:
    likes_df = pd.DataFrame(columns=["exam"])

# metric names for sliders
metric_names = list(schedule_data[0]["metrics"].keys()) if schedule_data else []
param_names = list(schedule_data[0]["params"].keys()) if schedule_data else []

# ---- Layout: filters vs. results ----
left_col, right_col = st.columns([2, 4])
filtered = schedule_data.copy()

# ---- Apply filters ----
with left_col:
    st.subheader("FILTER SCHEDULES")
    # slot-exclusion UI + filtering
    filtered = apply_slot_exclusion(st, filtered, num_slots=NUM_SLOTS)
    print("PARAM NAMEESS" , param_names)
    # numeric sliders for all metrics (including max_slot)
    filtered, thresholds = apply_numeric_sliders(st, filtered, metric_names, param_names)

# ---- Display results ----
with right_col:
    if not filtered:
        st.warning("No schedules match the selected criteria.")
    for sched in filtered:
        # layout: download & pin buttons
        nth, col_pin, col_dl = st.columns([4, 1, 1])

        with col_dl:
            st.markdown('<div class="grey-button">', unsafe_allow_html=True)
            schedule_csv = os.path.join(SAVE_PATH, "schedules", f"{sched['basename']}.csv")
            with open(schedule_csv, "rb") as f:
                st.download_button(
                    label="Download schedule",
                    data=f,
                    file_name=f"{sched['basename']}.csv",
                    mime="text/csv",
                    key=f"dl_{sched['basename']}"
                )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_pin:
            if st.button("pin", key=f"more_{sched['basename']}"):
                if sched["name"] not in likes_df["exam"].values:
                    likes_df.loc[len(likes_df)] = sched["basename"]
                    likes_df.to_csv(likes_path, index=False)
                    st.success(f"Saved **{sched['name']}** to like.csv")
                else:
                    st.info(f"**{sched['name']}** already in like.csv")

        # show metrics & plots
        show_schedule_block(right_col, sched, incumbent=False, metrics=True)
