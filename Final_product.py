import time
import pandas as pd
import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sqlite3
import hashlib

# Set the page configuration
st.set_page_config(
    page_title="ChurnGuard Pro",
    page_icon="💳",  # You can replace this with the path to your custom icon
)

# Load the trained machine learning model (replace 'Pickle_RL_Model.pkl' with the actual model file path)
model = joblib.load('Pickle_RL_Model_1.pkl')

# Connect to SQLite database (create the database if it doesn't exist)
conn = sqlite3.connect('user_credentials_1.db')
cursor = conn.cursor()

# Create a table to store user credentials if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password_hash TEXT
    )
''')
conn.commit()


# Function to hash the password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Function to check user credentials in SQLite database
def authenticate_sqlite(username, password):
    cursor.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
    result = cursor.fetchone()
    if result is not None:
        stored_password_hash = result[0]
        input_password_hash = hash_password(password)
        return stored_password_hash == input_password_hash
    return False


# Function to add a new user to the SQLite database
def add_user_to_database(username, password):
    password_hash = hash_password(password)
    cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, password_hash))
    conn.commit()


# List of input fields in the same order as used for training the scaler
req_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
            'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Total_Trans_Amt', 'Total_Trans_Ct', 'Card_Category_Blue', 'Card_Category_Gold', 'Card_Category_Platinum',
            'Card_Category_Silver', 'Income_Category_$120K +', 'Income_Category_$40K - $60K',
            'Income_Category_$60K - $80K', 'Income_Category_$80K - $120K', 'Income_Category_Less than $40K', 'Gender_F',
            'Gender_M', 'Marital_Status_Divorced', 'Marital_Status_Married', 'Marital_Status_Single',
            'Education_Level_College', 'Education_Level_Doctorate', 'Education_Level_Graduate',
            'Education_Level_High School', 'Education_Level_Post-Graduate', 'Education_Level_Uneducated']


# Function to make predictions
def predict_churn(input_values, scaler):
    # Create a DataFrame with user inputs in the same order as req_cols
    user_input_df = pd.DataFrame([input_values], columns=req_cols)

    # Apply MinMax scaling to the user input data
    user_input_scaled = scaler.transform(user_input_df)

    # Make predictions using the loaded model
    prediction = model.predict(user_input_scaled)

    return prediction


# Set the background image
background_image = 'background_image.jfif'  # Replace with the path to your background image

st.image(background_image, use_column_width=True)

# Streamlit UI
st.title(':blue[Churn Guard Pro] 💳')

# Initialize session state with an empty dictionary
session_state = st.session_state

# Initialize session state with an empty dictionary
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# Load the scaler (replace 'scaler.pkl' with the actual scaler file path)
scaler = joblib.load('scaler_2.pkl')

# Login or Register selection
choice = st.sidebar.radio("Choose an action", ["Login", "Register"])

# Sign-in section
if choice == "Login":
    login_container = st.sidebar.empty()
    if st.session_state.current_user is None:
        sign_in_username = login_container.text_input('Username')
        login_container = st.sidebar.empty()
        sign_in_password = login_container.text_input('Password', type='password')
        login_container = st.sidebar.empty()
        if login_container.button('Sign In'):
            if authenticate_sqlite(sign_in_username, sign_in_password):
                st.session_state.current_user = sign_in_username
                st.sidebar.success(f'You are signed in as: {st.session_state.current_user}')
                sign_in_username = ''
                sign_in_password = ''
            else:
                st.sidebar.error('Authentication failed. Please check your credentials.')

# User registration section in the sidebar
if choice == "Register":
    st.sidebar.header("Register")
    register_username = st.sidebar.text_input('New Username')
    register_password = st.sidebar.text_input('New Password', type='password')
    register_confirm_password = st.sidebar.text_input('Confirm Password', type='password')

    if st.sidebar.button('Create Account'):
        if register_password == register_confirm_password:
            add_user_to_database(register_username, register_password)
            st.sidebar.success('Account created successfully. You can now sign in.')
            register_username = ''
            register_password = ''
            register_confirm_password = ''

if session_state.current_user:
    # Additional input fields displayed after a successful login
    st.subheader('Input customer data', divider='gray')

    # List of input fields
    numerical_inputs = [
        'Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
        'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Total_Trans_Ct', 'Total_Trans_Amt'
    ]

    category_inputs = [
        'Card_Category_Blue', 'Card_Category_Gold', 'Card_Category_Silver', 'Card_Category_Platinum',
        'Income_Category_$120K +', 'Income_Category_$40K - $60K', 'Income_Category_$60K - $80K',
        'Income_Category_$80K - $120K',
        'Income_Category_Less than $40K', 'Gender_F', 'Gender_M', 'Marital_Status_Divorced', 'Marital_Status_Married',
        'Marital_Status_Single',
        'Education_Level_College', 'Education_Level_Doctorate', 'Education_Level_Graduate',
        'Education_Level_High School',
        'Education_Level_Post-Graduate', 'Education_Level_Uneducated'
    ]

    # Initialize input_values as a dictionary of False values for all input fields
    input_values = {field: False for field in category_inputs}

    # Divide the input fields into two columns
    col1, spacer, col2 = st.columns([3.5, 0.2, 3.5])

    numerical_labels = {
        'Customer_Age': 'Customer Age',
        'Dependent_count': 'Number of Dependents',
        'Months_on_book': 'Relationship with bank in months',
        'Total_Relationship_Count': 'Total Relationship Count',
        'Months_Inactive_12_mon': 'Number of Months Inactive (Last 12 months)',
        'Contacts_Count_12_mon': 'Number of Contacts (Last 12 Months)',
        'Credit_Limit': 'Credit Limit of the credit card',
        'Total_Revolving_Bal': 'Total Revolving Balance',
        'Total_Trans_Ct': 'Total Transaction Count (Last 12 months)',
        'Total_Trans_Amt': 'Total Transaction Amount'

    }
    for field in numerical_inputs[:-2]:  # Exclude the last two numerical inputs
        input_values[field] = col1.number_input(numerical_labels[field], min_value=0)

    for field in numerical_inputs[-2:]:  # Include the last two numerical inputs
        input_values[field] = col2.number_input(numerical_labels[field], min_value=0)

    # Select Card_Category from a list and set the corresponding input value to True
    selected_card_category = col2.selectbox('Select Card Category', ['Blue',  'Silver', 'Gold', 'Platinum'])
    for category in ['Card_Category_Blue', 'Card_Category_Gold', 'Card_Category_Silver', 'Card_Category_Platinum']:
        input_values[category] = (category == f'Card_Category_{selected_card_category}')

    # Select Income_Category from a list and set the corresponding input value to True
    selected_income_category = col2.selectbox('Select Income Category',
                                              ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K',
                                               '$120K +'])
    for category in ['Income_Category_$120K +', 'Income_Category_$40K - $60K', 'Income_Category_$60K - $80K',
                     'Income_Category_$80K - $120K', 'Income_Category_Less than $40K']:
        input_values[category] = (category == f'Income_Category_{selected_income_category}')

    # Select Gender and set the corresponding input value to True
    selected_gender = col2.radio('Select Gender', ['Female', 'Male'])
    input_values['Gender_F'] = (selected_gender == 'Female')
    input_values['Gender_M'] = (selected_gender == 'Male')

    # Select Marital_Status and set the corresponding input value to True
    selected_marital_status = col2.radio('Select Marital Status', ['Divorced', 'Married', 'Single'])
    input_values['Marital_Status_Divorced'] = (selected_marital_status == 'Divorced')
    input_values['Marital_Status_Married'] = (selected_marital_status == 'Married')
    input_values['Marital_Status_Single'] = (selected_marital_status == 'Single')

    # Select Education_Level and set the corresponding input value to True
    selected_education_level = col2.radio('Select Education Level',
                                          ['College Graduate', 'Doctorate', 'Graduate', 'High School', 'Post-Graduate',
                                           'Uneducated'])
    for category in ['Education_Level_College', 'Education_Level_Doctorate', 'Education_Level_Graduate',
                     'Education_Level_High School', 'Education_Level_Post-Graduate', 'Education_Level_Uneducated']:
        input_values[category] = (category == f'Education_Level_{selected_education_level}')

    # Boolean to control whether to display the results
    display_results = False

    if st.button('Submit'):
        # Show loading spinner
        with st.spinner('Predicting...'):
            time.sleep(1)  # Simulate a 2-second pause
            display_results = True

    if display_results:
        # Make predictions based on the input values
        prediction = predict_churn(input_values, scaler)

        if prediction[0] == 0:
            st.success('Customer has a Less possibility to churn', icon="✅")
        else:
            st.error('  Customer has a High possibility to churn', icon="⚠️")

    if st.sidebar.button('Log Out'):
        session_state.current_user = None


# Close the database connection when the Streamlit app is stopped
@st.cache_resource
def on_session_state_exit():
    conn.close()
