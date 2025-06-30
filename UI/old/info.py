import streamlit as st
import pandas as pd

# Add custom styles to apply to the page
st.markdown(
    """
    <style>
        body, .stApp {
            background-color: #f0f2f6;  /* Light grayish-blue background */
            text-align: center;  /* Center-align all text */
        }

        /* Ensure all headers, labels, and text on the page are black and centered */
        h1, h2, h3, h4, h5, h6 {
            color: black !important;  /* Make sure all text elements are black */
            text-align: center !important;  /* Center-align text */
        }

        /* Make the Streamlit menu (top toolbar) background dark with white text */
        .stToolbar, .st-a11y {
            background-color: #333333 !important;  /* Dark menu background */
            color: white !important;  /* White text in the menu */
        }

        /* Style the login button: black background with white text */
        .stButton > button {
            background-color: black !important;  /* Black background */
            color: white !important;  /* White text */
            border: 2px solid black !important;  /* Black border */
        }

        /* Make text inside input fields white */
        div[data-baseweb="input"] input {
            color: white !important;  /* White text in input fields */
            background-color: #333333 !important;  /* Dark background for input fields */
            border: 1px solid #444444 !important;  /* Dark border for inputs */
        }
          label {
            color: black !important;  /* Change label text color to black */
            font-weight: bold;  /* Make the label text bold (optional) */
        }
        /* Make text in the select dropdown white */
        div[role="listbox"] {
            color: white !important;
            background-color: #333333 !important;
        }
            /* Center the logo */
        .logo-container {
            display: flex;
            justify-content: center;  /* Center horizontally */
            align-items: center;      /* Center vertically */
            padding: 20px;
        }

               /* Center the logo */
        .stImage {
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo - centered at the top
st.image("cornell-logo.png", width=150)  # Ensure the logo path is correct

st.title("FINAL EXAM SCHEDULING")
st.subheader("Information About Schedule Metrics")


# Define the table in HTML with inline CSS for styling
table_html = """
<style>
    .table-container {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    table {
        width: 120%;
        border-collapse: collapse;
        background-color: white;
    }
    th, td {
        border: 3px solid black;
        padding: 10px;
        text-align: left;
        color: black;
    }
    th {
        background-color: #b0b1b5;
    }
</style>

<div class="table-container">
    <table>
        <tr>
            <th>Metric</th>
            <th>Description</th>
        </tr>
        <tr>
            <td>Direct conflict (Conflict)</td>
            <td>A student with two exams scheduled at the same time slot.</td>
        </tr>
        <tr>
            <td>Back-to-back exams (B2B)</td>
            <td>A student enrolled in a pair of courses that are scheduled to happen in adjacent slots. This can either be 9am and 2pm exam, 2pm and 7pm exam, or 7pm and 9am exam. But no other courses scheduled immediately before or after.</td>
        </tr>
        <tr>
            <td>Two exams in 24 hours (2-in-24hr)</td>
            <td>A student with two exams in three consecutive slots, or equivalently, within a 24-hour period.</td>
        </tr>
        <tr>
            <td>Three exams in the same day (Same day triple)</td>
            <td>A student enrolled in a triple of courses where one is scheduled at 9am, 2pm, and 7pm on the same day. No courses immediately before or after.</td>
        </tr>
        <tr>
            <td>Three exams in 24 hours (3-in-24hr)</td>
            <td>A student enrolled in a triple of courses scheduled at either 2pm, 7pm, 9am the next day, or 7pm, 9am the next day, 2pm the next day. And no courses immediately before or after those courses.</td>
        </tr>
        <tr>
            <td>Two exams in 3 slots (2-in-3)</td>
            <td>A student enrolled in a pair of courses with only one exam slot between them and no other courses immediately before or after. This could be scheduled at either 9am, 7pm, or 2pm and 9am, or 7pm and 2pm.</td>
        </tr>
        <tr>
            <td>Three exams in 4 slots (3-in-4)</td>
            <td>A student enrolled in a triple of courses where they have a 2-in-3 and back-to-back next to each other. A triple of courses scheduled within 4 slots.</td>
        </tr>
    </table>
</div>
"""

# Display the table using Streamlit Markdown
st.markdown(table_html, unsafe_allow_html=True)
 # Adjust height as needed
