import streamlit as st

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

def login():
    st.title("FINAL EXAM SCHEDULING")
    st.subheader("Cornell University")
    st.subheader("Log In")

    user_id = st.text_input("Enter ID")
    password = st.text_input("Enter Password", type="password")

    # Dummy credentials
    valid_id = "Fabi"
    valid_password = "ORIE123"

    if st.button("Login"):
        if user_id == valid_id and password == valid_password:
            st.session_state["authenticated"] = True
            st.success("Login successful! Redirecting...")
            st.switch_page("pages/main.py")  # Redirects to the main page
        else:
            st.error("Invalid ID or Password")

if __name__ == "__main__":
    login()
