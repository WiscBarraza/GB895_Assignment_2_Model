import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model and encoder
with open("churn_rf_healthy_meals_pt2.pkl", "rb") as f:
    model = pickle.load(f)

with open("churn_encoder_healthy_meals_pt2.pkl", "rb") as f:
    encoder = pickle.load(f)

categorical_cols = ['EDUCATION', 'INCOME_LEVEL', 'DEVICE_TYPE']

feature_columns = [
    'NUM_SESSIONS', 'GROSS_SESSION_LENGTH', 'ACTIVE_DAYS', 'ACTIVE_QUARTERS',
    'AGE', 'TECH_COMFORT_SCORE', 'AVG_SESSIONS_PER_QUARTER',
    'EDUCATION_High School', 'EDUCATION_Other', 'EDUCATION_Post-Graduate',
    'INCOME_LEVEL_Low', 'INCOME_LEVEL_Medium', 'INCOME_LEVEL_Very High',
    'DEVICE_TYPE_Mobile-only', 'DEVICE_TYPE_Multi-device'
]

st.set_page_config(page_title="Churn Risk Simulator", layout="centered")
st.title("Churn Risk Simulator")
st.write("Enter hypothetical customer values to see predicted churn/renewal probability.")

# Inputs 
num_sessions = st.slider("Num Sessions", min_value=0, max_value=262, value=27, step=1)
gross_session_length = st.slider("Gross Session Length", min_value=0, max_value=15601, value=830, step=1)
active_days = st.slider("Active Days", min_value=0, max_value=5, value=2, step=1)
active_quarters = st.slider("Active Quarters", min_value=0, max_value=4, value=2, step=1)
age = st.slider("Age", min_value=20, max_value=54, value=36, step=1)
tech_comfort_score = st.slider("Tech Comfort Score", min_value=1, max_value=5, value=3, step=1)
avg_sessions_per_quarter = st.slider("Avg Sessions per Quarter", min_value=1, max_value=122, value=18, step=1)

education = st.selectbox("Education", ["High School", "Graduate", "Post-Graduate", "Other"])
income_level = st.selectbox("Income Level", ["Low", "Medium", "High", "Very High"])
device_type = st.selectbox("Device Type", ["Desktop-only", "Mobile-only", "Multi-device"])

# Predict button
if st.button("Predict"):

    # Step 1: build numeric portion
    numeric_df = pd.DataFrame({
        'NUM_SESSIONS': [num_sessions],
        'GROSS_SESSION_LENGTH': [gross_session_length],
        'ACTIVE_DAYS': [active_days],
        'ACTIVE_QUARTERS': [active_quarters],
        'AGE': [age],
        'TECH_COMFORT_SCORE': [tech_comfort_score],
        'AVG_SESSIONS_PER_QUARTER': [avg_sessions_per_quarter],
    })

    # Step 2: build categorical portion, in the same column order used during fit
    raw_categorical = pd.DataFrame({
        'EDUCATION': [education],
        'INCOME_LEVEL': [income_level],
        'DEVICE_TYPE': [device_type],
    })[categorical_cols]

    # Step 3: encode categorical using the SAME fitted encoder from training
    encoded = encoder.transform(raw_categorical)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Step 4: combine numeric + encoded categorical
    input_df = pd.concat([numeric_df.reset_index(drop=True),
                           encoded_df.reset_index(drop=True)], axis=1)

    # Step 5: force column order to match training exactly
    input_df = input_df[feature_columns]

    # Step 6: predict
    probability_renewed = model.predict_proba(input_df)[0][1]
    churn_probability = 1 - probability_renewed

    st.subheader("Result")
    st.write(f"**Churn probability:** {churn_probability:.1%}")
    st.write(f"**Renewal probability:** {probability_renewed:.1%}")
