import streamlit as st
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
        .stButton > button {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: auto; /* Centers the button horizontally */
            width: 25%; /* Adjust width as needed */
        }
    </style>
    """,
    unsafe_allow_html=True
)

#st.image("cornell-logo.png", width=150)
st.title("FINAL EXAM SCHEDULING")
st.markdown(f"<p style='color: black; text-align: center;'>Cornell University</p>", unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)
st.title("THANK YOU!")
st.markdown(f"<div style='color: black; text-align: center;'>Schedules will be available in approximately:</div>", unsafe_allow_html=True)
st.title("~36 HRS")
st.markdown(f"<div style='color: black; text-align: center;'>An email will be sent as soon as schedules are available</div>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)
if st.button("Log Out"):
    st.switch_page("login.py")

st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown(f"<h6 style='color: black; text-align: center;'>Questions?</h6>", unsafe_allow_html=True)
st.markdown(f"<div style='color: black; text-align: center;'>Contact: exam_schedule_support@cornell.edu</div>", unsafe_allow_html=True)
