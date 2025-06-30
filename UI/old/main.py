import streamlit as st

st.set_page_config(page_title="Home page", page_icon="🌍", layout="wide")

# Custom CSS for styling only the first button
st.markdown(
    """
    <style>
        body, .stApp {
            background-color: #f0f2f6;  /* Light grayish-blue background */
        }
        h1, h2, h3, h4, h5, h6 {
            color: black !important;  /* Ensure all headers, labels, and text are black */
            text-align: center;
        }
        stButton > button {
            display: block;
            margin: auto; /* Centers buttons */
            width: 100%; /* Makes buttons more uniform */
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("cornell-logo.png", width=150)
st.title("FINAL EXAM SCHEDULING")

# Center the first button using a unique class
col1, col2, col3 = st.columns([1, 2, 1])  # Middle column is wider for centering

with col2:
    if st.button("View Generated Schedules (03/21 Request) →"):
        st.switch_page("pages/schedules.py")
st.markdown("</div>", unsafe_allow_html=True)  # Close the div

# Spacer for better separation
st.markdown("<br>", unsafe_allow_html=True)

# Create three columns for the other buttons (side by side)
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("Submit a Final Exam Request →"):
        st.switch_page("pages/request.py")

with col2:
    if st.button("Past Finals Schedules →"):
        st.switch_page("pages/old.py")

with col3:
    if st.button("Information about Schedule Metrics →"):
        st.switch_page("pages/info.py")
