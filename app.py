# This is an streamlit app that will be used to predict if a person is eligible for a loan or not
import numpy as np
import streamlit as st
import pandas as pd
import os
import pickle

# ML Libraries
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from xgboost import XGBClassifier

import pickle

class Encoders:
    def __init__(self):
        self.education = self.load_encoder("education_encoder.pkl")
        self.gender = self.load_encoder("gender_encoder.pkl")
        self.home_ownership = self.load_encoder("home_ownership_encoder.pkl")
        self.loan_intent = self.load_encoder("loan_intent_encoder.pkl")
        self.previous_loans = self.load_encoder("previous_loans_encoder.pkl")

    def load_encoder(self, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

Encoders.load_encoder

#import the model
with open("xgb_classifier_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)


st.title("Loan Eligibility Predictor 🏦💸")

gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage", "Other"])
loan_intent = st.selectbox("Loan Intent", ["Education", "Medical", "Venture", "Personal", "Home Improvement", "Debt Consolidation"])
previous_loans = st.selectbox("Previous Loan Default", ["Yes", "No"])

income = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_rate = st.number_input("Interest Rate (%)", min_value=0.0, format="%.2f")
loan_term = st.number_input("Loan Term (in months)", min_value=0)
age = st.number_input("Age", min_value=18)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "person_income": income,
        "person_age": age,
        "loan_amnt": loan_amount,
        "loan_int_rate": loan_rate,
        "loan_term": loan_term,
    }])

    # Apply encoders
    input_data["person_gender"] = encoders.gender.transform([gender])
    input_data["person_education"] = encoders.education.transform([[education]])
    home_df = pd.DataFrame(encoders.home_ownership.transform([[home_ownership]]), columns=encoders.home_ownership.get_feature_names_out())
    loan_df = pd.DataFrame(encoders.loan_intent.transform([[loan_intent]]), columns=encoders.loan_intent.get_feature_names_out())
    input_data["previous_loan_defaults_on_file"] = encoders.previous_loans.transform([previous_loans])

    input_data = pd.concat([input_data.reset_index(drop=True), home_df, loan_df], axis=1)

    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("Eligible for Loan!")
    else:
        st.error("Not Eligible for Loan.")