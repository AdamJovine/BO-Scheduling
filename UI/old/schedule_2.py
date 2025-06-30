import streamlit as st
from PIL import Image
import os

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
st.markdown("<div class='subtitle' style='color: black;'>Schedule 2</div>", unsafe_allow_html=True)


# ---- Layout: Left | Center | Right ----
left, center, right = st.columns([1.5, 3, 1.5])

# ---- Left Panel: Schedule Details ----
with left:
    st.markdown("<div class='gray-box'><strong>Schedule Details</strong><br><br>"
                "Blocks: 24<br>"
                "Conflicts: 0<br>"
                "Quints: 0<br>"
                "Quads: 1<br>"
                "4-in-5: 3<br>"
                "3-in-24hr: 19<br>"
                "Same day triple: 6<br>"
                "3-in-4: 202<br>"
                "Eve/Mor B2B: 472<br>"
                "Other B2B: 814<br>"
                "2-in-3: 2841<br><br>"
                "<strong>Large exam cutoff:</strong><br>300"
                "</div>", unsafe_allow_html=True)

# ---- Center Panel: Main Bar Chart ----
with center:
    if os.path.exists("schedule_2_schedule.png"):
        st.image("schedule_1_schedule.png", use_container_width=True)
    else:
        st.markdown("*(Schedule chart placeholder)*")

# ---- Right Panel: Stakeholder Info ----
with right:
    st.markdown("<div class='gray-box' style='text-align:center'><strong>Stakeholder Information</strong></div>", unsafe_allow_html=True)
    st.markdown("<div class='graph-title'>Last exam distribution</div>", unsafe_allow_html=True)
    if os.path.exists("schedule_1_done.png"):
        st.image("schedule_1_done.png", use_container_width=True)
    else:
        st.markdown("*(Last exam image placeholder)*")

    st.markdown("<div class='graph-title'>Gaps between exams</div>", unsafe_allow_html=True)

    if os.path.exists("schedule_1_gaps.png"):
        st.image("schedule_1_gaps.png", use_container_width=True)
    else:
        st.markdown("*(Gaps graph placeholder)*")
