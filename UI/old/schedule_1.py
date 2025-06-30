import streamlit as st
from PIL import Image
import os
import glob
from config.settings import SAVE_PATH , UI_PATH
from UI.pages.schedule_plots import get_plot, last_day
import pandas as pd
st.set_page_config(layout="wide")

# ---- Style ----
st.markdown(
    """
    <style>
        body, .stApp {
            background-color: #f0f2f6;
            color: black !important;
        }

        h1, h2, h3, h4, h5, h6, p, span, div, label {
            color: black !important;
        }

        .stButton > button {
            background-color: black !important;
            color: white !important;
            border: 2px solid black !important;
        }

        div[data-baseweb="input"] input {
            color: black !important;
            background-color: white !important;
            border: 1px solid #999 !important;
        }

        div[role="listbox"] {
            background-color: white !important;
            color: black !important;
        }

        .stImage {
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-top: 20px;
        }

        header[data-testid="stHeader"] {
            background-color: #333333 !important;
            color: white !important;
        }

        .title {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
        }

        .subtitle {
            text-align: center;
            font-size: 24px;
            margin-bottom: 30px;
        }

        .gray-box {
            background-color: #e0e0e0;
            padding: 20px;
            border-radius: 10px;
            font-size: 18px;
        }

        .select-button {
            display: flex;
            justify-content: center;
            margin-top: 30px;
        }

        .more-info {
            text-align: right;
            font-size: 18px;
            margin-top: 30px;
            font-weight: bold;
        }

        .graph-title {
            text-align: center;
            font-weight: bold;
            margin-top: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Title ----
st.image("cornell-logo.png", width=150)
st.markdown("<div class='title'>FINAL EXAM SCHEDULER</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle' style='color: black;'>Schedule 1</div>", unsafe_allow_html=True)


# ---- Layout: Left | Center | Right ----
left, center, right = st.columns([1.5, 3, 1.5])
# Option B: dropdown of all .csv files in your schedules folder
schedule_files = [
    os.path.basename(p)
    for p in glob.glob(os.path.join(SAVE_PATH, "schedules", "*.csv"))
]
print('schedule_files ' , schedule_files )
#/home/asj53/BOScheduling/results/sp25/schedules/20250614_065847i1-30830ed058547eee431b8eed83a4feda.csv
# specify your old default (no leading space)
default_sched = "20250614_065847i1-30830ed058547eee431b8eed83a4feda.csv"
default_idx = schedule_files.index(default_sched) if default_sched in schedule_files else 0

# let the user pick (with your default pre-selected)
sched_name = st.selectbox(
    "Choose a schedule file",
    schedule_files,
    index=default_idx
).strip()  # just in case

# derive a plot base name (no ".csv")
base_name = os.path.splitext(sched_name)[0]

# now regenerate your charts for the newly selected schedule
get_plot(sched_name, base_name)
last_day(sched_name, base_name)

# load the two DataFrames from the chosen file
schedule_df = pd.read_csv(os.path.join(SAVE_PATH, "schedules", sched_name))
metrics_df  = pd.read_csv(os.path.join(SAVE_PATH, "metrics",   sched_name))

# … then the rest of your layout/panels stays the same, e.g.:

left, center, right = st.columns([1.5, 3, 1.5])
with left:
    m = metrics_df.iloc[0]
    items_html = "".join(f"{col}: {val}<br>" for col, val in m.items())
    st.markdown(
        f"""
        <div class='gray-box'>
            <strong>Schedule Details</strong><br><br>
            {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

with center:
    plot_path = os.path.join(UI_PATH, f"{base_name}.png")
    if os.path.exists(plot_path):
        st.image(plot_path, use_container_width=True)
    else:
        st.markdown("*(Schedule chart placeholder)*")

with right:
    st.markdown("<div class='gray-box' style='text-align:center'><strong>Stakeholder Information</strong></div>", unsafe_allow_html=True)

    # last-exam distribution
    st.markdown("<div class='graph-title'>Last exam distribution</div>", unsafe_allow_html=True)
    dist_path = os.path.join(UI_PATH, f"{base_name}_distribution.png")
    if os.path.exists(dist_path):
        st.image(dist_path, use_container_width=True)
    else:
        st.markdown("*(Last exam image placeholder)*")

    # gaps between exams
    st.markdown("<div class='graph-title'>Gaps between exams</div>", unsafe_allow_html=True)
    gaps_path = os.path.join(UI_PATH, f"{base_name}_gaps.png")
    if os.path.exists(gaps_path):
        st.image(gaps_path, use_container_width=True)
    else:
        st.markdown("*(Gaps graph placeholder)*")
