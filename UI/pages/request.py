


import streamlit as st
import pandas as pd
import datetime

# Set the page title and layout
st.set_page_config(page_title="Final Exam Request", layout="wide")

# Custom CSS for styling fixes
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
        .stFileUploader label {
            color: black !important;  /* Sets file uploader text color to black */
            font-weight: bold;
        }
        .stCheckbox label {
            color: black !important;  /* Sets file uploader text color to black */
            font-weight: bold;
        }
        /* Logo container for proper positioning */
        .logo-container {
            display: flex;
            align-items: center;
            padding: 10px 0 10px 20px;  /* Adds spacing from top & left */
        }
        /* Adjust spacing for better layout */
        .block-container {
            padding: 3rem 5rem; /* Increases left/right spacing */

        }
        .stColumn {
            padding: 20px; /* Increases padding around columns */
        }
        .stButton > button {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: auto; /* Centers the button horizontally */
            width: 50%; /* Adjust width as needed */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Display logo in the top right
st.image("cornell-logo.png", width=150)

# Centered Title with added space
st.markdown("<h1 style='margin-bottom: 40px;'>FINAL EXAM SCHEDULING</h1>", unsafe_allow_html=True)

# Create three columns with more spacing
col1, col2, col3 = st.columns([1.2, 1.5, 1.2])  # Adjusted proportions for better spacing

# Section for uploading CSV in the first column
with col1:
    st.markdown("<h2>Upload CSV Files</h2>", unsafe_allow_html=True)

    uploaded_file1 = st.file_uploader("Course Information", type="csv", key="file1")
    if uploaded_file1 is not None:
        df1 = pd.read_csv(uploaded_file1)
        st.markdown("<h3>Data Preview</h3>", unsafe_allow_html=True)
        st.dataframe(df1)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)  # Spacer

    uploaded_file2 = st.file_uploader("Student Information", type="csv", key="file2")  # Now in black
    if uploaded_file2 is not None:
        df2 = pd.read_csv(uploaded_file2)
        st.markdown("<h3>Data Preview</h3>", unsafe_allow_html=True)
        st.dataframe(df2)

# Calendar section in the second column (for date range selection)
with col2:
    st.markdown("<h2>Select Date Range for Final Exams</h2>", unsafe_allow_html=True)

    # Using range argument for selecting a date range
    st.markdown("<p style='color: black; text-align: center;'>Select the range of dates for the exam period</p>", unsafe_allow_html=True)

    start_end_dates = st.date_input(
    "",  # Empty label to avoid duplication
    value=(datetime.date.today(), datetime.date.today() + datetime.timedelta(days=1)),
    min_value=datetime.date(2020, 1, 1),
    max_value=datetime.date(2030, 12, 31),
    key="date_range"
)

    # Display the selected date range
    start_date, end_date = start_end_dates
    st.markdown("<h3 style='color: black; text-align: center;'>Selected Dates:</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: black; text-align: center;'>{start_date} to {end_date}</p>", unsafe_allow_html=True)

    st.markdown("<br><br><br><br>", unsafe_allow_html=True)  # Adjust the number of <br> as needed
    if st.button("Submit Request"):
        st.switch_page("pages/wait.py")
# Blocked-off dates and times input section in the third column
with col3:
    st.markdown("<h2>Enter Blocked Off Dates and Times</h2>", unsafe_allow_html=True)

    # Text area for users to enter blocked-off dates/times
    user_input = st.text_area("")
    # Custom label in black

    st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)  # Spacer

    if user_input.strip():
        selected_options = [option.strip() for option in user_input.split(",")]
        st.markdown("<h3>You entered:</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='color: black;'>{', '.join(selected_options)}</p>",
            unsafe_allow_html=True
        )

    # New text input box for additional user input
    st.markdown("<h2>Additional Notes</h2>", unsafe_allow_html=True)
    additional_notes = st.text_input("")

    if additional_notes:
        st.markdown("<h3>Your Notes:</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='color: black;'>{additional_notes}</p>",
            unsafe_allow_html=True
        )
