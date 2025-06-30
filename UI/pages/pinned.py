import streamlit as st
import os
import pandas as pd
from config.settings import SAVE_PATH, UI_PATH, NUM_SLOTS
from UI.pages.schedule_plots import get_plot, last_day

# Page config
st.set_page_config(layout="wide")

# ---- Header ----
st.image("cornell-logo.png", width=100)
st.title("PINNED SCHEDULES")

# ---- Load liked exams ----n
likes_path = os.path.join(SAVE_PATH, 'like.csv')
if os.path.exists(likes_path):
    likes_df = pd.read_csv(likes_path)
else:
    likes_df = pd.DataFrame(columns=['exam'])

pinned = likes_df['exam'].tolist()

if not pinned:
    st.info("No pinned schedules yet. Go to the main page to pin your favorites.")
    st.stop()

# ---- Ensure plots exist for pinned schedules ----
@st.cache_resource
def generate_plots_for_pins(exams):
    for base in exams:
        sched_file = f"{base}.csv"
        # Create if missing
        plot_path = os.path.join(UI_PATH, f"{base}.png")
        dist_path = os.path.join(UI_PATH, f"{base}_distribution.png")
        if not os.path.exists(plot_path) or not os.path.exists(dist_path):
            get_plot(sched_file, base)
            last_day(sched_file, base)

generate_plots_for_pins(pinned)

# ---- Display pinned schedules ----
for base in pinned:
    st.subheader(base)
    # Download
    sched_path = os.path.join(SAVE_PATH, 'schedules', f"{base}.csv")
    with st.container():
        col_dl, col_unpin = st.columns([1,1])
        with col_dl:
            if os.path.exists(sched_path):
                with open(sched_path, 'rb') as f:
                    st.download_button(
                        label="Download schedule",
                        data=f,
                        file_name=f"{base}.csv",
                        mime='text/csv'
                    )
            else:
                st.warning("Schedule file missing.")
        with col_unpin:
            if st.button("Unpin", key=f"unpin_{base}"):
                likes_df = likes_df[likes_df['exam'] != base]
                likes_df.to_csv(likes_path, index=False)
                st.success(f"Unpinned **{base}**")
                st.experimental_rerun()

    # Load metrics
    metrics_file = os.path.join(SAVE_PATH, 'metrics', f"{base}.csv")
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        df = df[df.columns[:17]]
        m = df.iloc[0].to_dict()
    else:
        m = {}

    # Layout metrics and plots
    mcol, pcol, dcol = st.columns([1.5, 3, 3])
    with mcol:
        html = "".join(f"{k}: {v}<br>" for k, v in m.items())
        st.markdown(f"<div class='gray-box'><strong>Metrics</strong><br><br>{html}</div>", unsafe_allow_html=True)
    img1 = os.path.join(UI_PATH, f"{base}.png")
    with pcol:
        if os.path.exists(img1):
            st.image(img1, caption="Schedule plot", use_container_width=True)
        else:
            st.markdown("*(Schedule image not available)*")
    img2 = os.path.join(UI_PATH, f"{base}_distribution.png")
    with dcol:
        if os.path.exists(img2):
            st.image(img2, caption="# of students done plot", use_container_width=True)
        else:
            st.markdown("*(Distribution image not available)*")
