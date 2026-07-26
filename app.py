import gradio as gr
import numpy as np
import pandas as pd

categorical_cols = ['EDUCATION', 'INCOME_LEVEL', 'DEVICE_TYPE']

feature_columns = [
    'NUM_SESSIONS', 'GROSS_SESSION_LENGTH', 'ACTIVE_DAYS', 'ACTIVE_QUARTERS',
    'AGE', 'TECH_COMFORT_SCORE', 'AVG_SESSIONS_PER_QUARTER',
    'EDUCATION_High School', 'EDUCATION_Other', 'EDUCATION_Post-Graduate',
    'INCOME_LEVEL_Low', 'INCOME_LEVEL_Medium', 'INCOME_LEVEL_Very High',
    'DEVICE_TYPE_Mobile-only', 'DEVICE_TYPE_Multi-device'
]

def predict(num_sessions, gross_session_length, active_days, active_quarters,
            age, tech_comfort_score, avg_sessions_per_quarter,
            education, income_level, device_type):

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

    # Step 5: force column order to match training exactly (critical safety net)
    input_df = input_df[feature_columns]

    # Step 6: predict
    probability_renewed = model.predict_proba(input_df)[0][1]
    churn_probability = 1 - probability_renewed

    return f"Churn probability: {churn_probability:.1%}  |  Renewal probability: {probability_renewed:.1%}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(label="Num Sessions", minimum=0, maximum=262, value=27, step=1),
        gr.Slider(label="Gross Session Length", minimum=0, maximum=15601, value=830, step=1),
        gr.Slider(label="Active Days", minimum=0, maximum=5, value=2, step=1),
        gr.Slider(label="Active Quarters", minimum=0, maximum=4, value=2, step=1),
        gr.Slider(label="Age", minimum=20, maximum=54, value=36, step=1),
        gr.Slider(label="Tech Comfort Score", minimum=1, maximum=5, value=3, step=1),
        gr.Slider(label="Avg Sessions per Quarter", minimum=1, maximum=122, value=18, step=1),
        gr.Dropdown(label="Education", choices=["High School", "Bachelor's", "Post-Graduate", "Other"]),
        gr.Dropdown(label="Income Level", choices=["Low", "Medium", "High", "Very High"]),
        gr.Dropdown(label="Device Type", choices=["Desktop-only", "Mobile-only", "Multi-device"]),
    ],
    outputs="text",
    title="Churn Risk Simulator",
    description="Enter hypothetical customer values to see predicted churn/renewal probability."
)

demo.launch(debug=True)