
import streamlit as st
import pandas as pd
import joblib

#  PAGE CONFIG 
st.set_page_config(
    page_title="LoanWise | Loan Prediction",
    page_icon="💜",
    layout="wide"
)

#  LOAD MODEL 
loaded_artifact = joblib.load('loan_approval_model.joblib')

loaded_model = loaded_artifact['model']
feature_names = loaded_artifact['feature_names']


#  CUSTOM STYLING 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F7F5FC;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(120deg, #34245F, #7656B8);
    padding: 32px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
    font-weight: 800;
}

.hero p {
    color: #E8DFFF;
    font-size: 15px;
}

.section-title {
    color: #34245F;
    font-size: 21px;
    font-weight: 800;
    margin: 20px 0 12px 0;
}

[data-testid="stForm"] {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #E8E1F4;
}

div.stButton > button {
    background: #6846A5;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    background: #50337F;
    color: white;
}

.result-card {
    padding: 34px 28px;
    border-radius: 20px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 8px 24px rgba(52, 36, 95, 0.06);
}

.result-card.approved-card {
    background: linear-gradient(135deg, #EAF8F0 0%, #FFFFFF 78%);
    border: 1px solid #B9E5CC;
}

.result-card.rejected-card {
    background: linear-gradient(135deg, #FFF0F1 0%, #FFFFFF 78%);
    border: 1px solid #F0C3C8;
}

.result-icon {
    width: 66px;
    height: 66px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 14px auto;
    border-radius: 50%;
    font-size: 35px;
    font-weight: 800;
}

.approved-card .result-icon {
    color: #16845B;
    background: #D4F1E0;
}

.rejected-card .result-icon {
    color: #C44855;
    background: #FCE0E3;
}

.result-eyebrow {
    color: #766A88;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 7px;
}

.approved {
    color: #16845B;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 8px;
}

.rejected {
    color: #C44855;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 8px;
}

.result-description {
    color: #655D70;
    font-size: 15px;
    margin: 0;
}


/* Input fields */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #34245F !important;
    border: 1px solid #D8CBEA !important;
    border-radius: 9px !important;
}

/* Input labels */
.stNumberInput label,
.stSelectbox label {
    color: #34245F !important;
    font-weight: 600 !important;
}

/* Input focus */
.stNumberInput input:focus,
.stTextInput input:focus {
    border: 2px solid #7656B8 !important;
    box-shadow: 0 0 0 2px rgba(118, 86, 184, 0.15) !important;
}

/* Dropdown text */
.stSelectbox div[data-baseweb="select"] {
    color: #34245F !important;
}

/* Form background */
[data-testid="stForm"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E8E1F4 !important;
}


</style>
""", unsafe_allow_html=True)


# PREDICTION FUNCTION 
def predict_loan(applicant):

    applicant_df = pd.DataFrame([applicant])

    applicant_df['education'] = applicant_df['education'].map(
        loaded_artifact['education_mapping']
    )

    applicant_df['self_employed'] = applicant_df['self_employed'].map(
        loaded_artifact['self_employed_mapping']
    )

    applicant_df = applicant_df[feature_names]

    prediction = loaded_model.predict(applicant_df)

    result = loaded_artifact['target_mapping'][int(prediction[0])]

    return result


# HEADER 
st.markdown("""
<div class="hero">
    <h1>💜 LoanWise</h1>
    <p>Smart Loan Assessment powered by Machine Learning.</p>
    <p>Enter applicant information below to generate a loan prediction.</p>
</div>
""", unsafe_allow_html=True)


# APPLICANT FORM 
st.markdown(
    '<div class="section-title">👤 Applicant Information</div>',
    unsafe_allow_html=True
)

with st.form("loan_form"):

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Personal & Employment Details")

        no_of_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            step=1
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"]
        )

        income_annum = st.number_input(
            "Annual Income",
            min_value=0,
            step=100000
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0,
            step=100000
        )

        loan_term = st.number_input(
            "Loan Term (Years)",
            min_value=1,
            step=1
        )

    with col2:

        st.markdown("#### Credit & Asset Details")

        cibil_score = st.number_input(
            "CIBIL Score",
            min_value=300,
            max_value=900,
            value=700
        )

        residential_assets_value = st.number_input(
            "Residential Assets Value",
            min_value=0,
            step=100000
        )

        commercial_assets_value = st.number_input(
            "Commercial Assets Value",
            min_value=0,
            step=100000
        )

        luxury_assets_value = st.number_input(
            "Luxury Assets Value",
            min_value=0,
            step=100000
        )

        bank_asset_value = st.number_input(
            "Bank Asset Value",
            min_value=0,
            step=100000
        )

    st.markdown("---")

    predict = st.form_submit_button(
        "✨ Assess Loan Application"
    )


#  PREDICTION RESULT
if predict:

    applicant = {
        'no_of_dependents': no_of_dependents,
        'education': education,
        'self_employed': self_employed,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value
    }

    result = predict_loan(applicant)

    st.markdown(
        '<div class="section-title">📊 Assessment Result</div>',
        unsafe_allow_html=True
    )

    if result == "Approved":

        st.markdown("""
        <div class="result-card approved-card">
            <div class="result-icon">✓</div>
            <div class="result-eyebrow">Assessment complete</div>
            <div class="approved">Loan Approved</div>
            <p class="result-description">
                The model predicts that this application belongs
                to the approved category.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-card rejected-card">
            <div class="result-icon">!</div>
            <div class="result-eyebrow">Assessment complete</div>
            <div class="rejected">Loan Rejected</div>
            <p class="result-description">
                The model predicts that this application belongs
                to the rejected category.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "This is a machine-learning demonstration, not an actual "
        "lending decision. Predictions are based on patterns "
        "learned from the training dataset."
    )


#  FOOTER 
st.divider()

st.markdown(
    "<p style='text-align:center;color:#999;'>"
    "LoanWise | Machine Learning Loan Assessment"
    "</p>",
    unsafe_allow_html=True
)
