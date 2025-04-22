# This is an streamlit app that will be used to predict if a person is eligible for a loan or not
import numpy as np
import streamlit as st
import pandas as pd
import os
import pickle
from pandas import DataFrame
import matplotlib.pyplot as plt
import random
import time

# ML Libraries
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Encoders
class Encoders:
    def __init__(self):
        self.gender: LabelEncoder = self.load_encoder("gender_encoder.pkl")
        self.education: OrdinalEncoder = self.load_encoder("education_encoder.pkl")
        self.home_ownership: OneHotEncoder = self.load_encoder("home_ownership_encoder.pkl")
        self.loan_intent: OneHotEncoder = self.load_encoder("loan_intent_encoder.pkl")
        self.previous_loans: LabelEncoder = self.load_encoder("previous_loans_encoder.pkl")

    def load_encoder(self, filename):
        """This function loads the encoders for the model

        Args:
            filename (_type_): the filename of the encider

        Returns:
            _type_: returns a loaded pickle object
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

#import the model
class XGB_Classifier(XGBClassifier):
    def __init__(self):
        self.xgb_model: XGBClassifier = self.load_model("xgb_classifier_model.pkl")
        self.encoders = Encoders()
        
    def load_model(self, filename):
        """This function is to load the model

        Args:
            filename (_type_): The filename of the model in .pkl format

        Returns:
            _type_: Pickle loaded object
        """
        with open(filename, "rb") as f:
            return pickle.load(f)
        
    def train(self, data: DataFrame):
        # Split data to train and test
        x = data.drop(columns=["loan_status"])
        y = data["loan_status"]
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # encoding the categorical variables
        train_gender_encoded = pd.DataFrame(self.encoders.gender.fit_transform(train_x["person_gender"]))
        train_gender_encoded.columns = ["gender"]

        train_home_ownership_encoded = pd.DataFrame(self.encoders.home_ownership.fit_transform(train_x[["person_home_ownership"]]))
        train_home_ownership_encoded.columns = self.encoders.home_ownership.get_feature_names_out(["person_home_ownership"])

        train_loan_intent_encoded = self.encoders.loan_intent.fit_transform(train_x[["loan_intent"]])
        train_loan_intent_encoded.columns = self.encoders.loan_intent.get_feature_names_out(["loan_intent"])

        train_prev_loans_encoded = pd.DataFrame(self.encoders.previous_loans.fit_transform(train_x["previous_loan_defaults_on_file"]))
        train_prev_loans_encoded.columns = ['previous_loan_default']

        train_education_encoded = pd.DataFrame(self.encoders.education.fit_transform(train_x[["person_education"]]))
        train_education_encoded.columns = ['education_level']
        
        # Flatten the y values
        train_y = train_y.values.ravel()
        
        #retrain the model
        self.xgb_model.fit(train_x, train_y)
        
    def predict(self, input_data: DataFrame):
        # Encode inputs using the loaded encoders

        gender_encoded = pd.DataFrame(self.encoders.gender.transform(input_data["person_gender"]))
        gender_encoded.columns = ["gender"]

        home_ownership_encoded = pd.DataFrame(self.encoders.home_ownership.transform(input_data[["person_home_ownership"]]))
        home_ownership_encoded.columns = self.encoders.home_ownership.get_feature_names_out(["person_home_ownership"])

        loan_intent_encoded = pd.DataFrame(self.encoders.loan_intent.transform(input_data[["loan_intent"]]))
        loan_intent_encoded.columns = self.encoders.loan_intent.get_feature_names_out(["loan_intent"])

        previous_loans_encoded = pd.DataFrame(self.encoders.previous_loans.transform(input_data["previous_loan_default"]))
        previous_loans_encoded.columns = ["previous_loan_default"]

        education_encoded = pd.DataFrame(self.encoders.education.transform(input_data[["person_education"]]))
        education_encoded.columns = ["education_level"]

        # Combine all required numeric + encoded features
        final_input = pd.concat([
            input_data[[
                "person_age", "person_income", "person_emp_exp",
                "loan_amnt", "loan_int_rate", "loan_percent_income",
                "cb_person_cred_hist_length", "credit_score"
            ]].reset_index(drop=True),
            gender_encoded,
            home_ownership_encoded.reset_index(drop=True),
            loan_intent_encoded.reset_index(drop=True),
            previous_loans_encoded.reset_index(drop=True),
            education_encoded.reset_index(drop=True)
        ], axis=1)

        # Predict with model
        return self.xgb_model.predict(final_input)


    def evaluation(self, data: DataFrame):
        x = data.drop(columns=["loan_status"])
        y = data["loan_status"]
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
        prediction = self.xgb_model.predict(test_x)
        return classification_report(test_y, prediction)
    
    def feature_importance(self, data: DataFrame):
        # Split data to train and test
        x = data.drop(columns=["loan_status"])
        y = data["loan_status"]
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
        
        plt.bar(range(len(self.xgb_classifier_model.feature_importances_)), self.xgb_classifier_model.feature_importances_)
        plt.xticks(ticks=range(len(train_x.columns)), labels=train_x.columns, rotation=90)
        plt.title("Feature Importances XGB Classifier Model")
        plt.tight_layout()  # Helps prevent label cutoff
        return plt #return the plot

# Input Presets
presets = {
    "Eligible for Loan 🎉":{
        "Annual_Income": 71948.0,
        "person_age": 22.0,
        "loan_amnt": 35000.0,
        "loan_int_rate": 16.02,
        "person_gender": "Female",
        "person_education": "Master",
        "person_home_ownership": "Rent",
        "loan_intent": "Personal",
        "previous_loan_default": "No",
        "loan_percent_income": 35000.0 / 71948.0 if 71948.0 else 0,
        "person_emp_exp": 0,
        "cb_person_cred_hist_length": 3,
        "credit_score": 561
    },
    "Not Eligible for Loan 🥹":{
        "Annual_Income": 12282.0,
        "person_age": 21.0,
        "loan_amnt": 1000.0,
        "loan_int_rate": 11.14,
        "person_gender": "Female",
        "person_education": "High School",
        "person_home_ownership": "Own",
        "loan_intent": "Personal",
        "previous_loan_default": "Yes",
        "loan_percent_income": 1000.0 / 12282.0 if 12282.0 else 0,
        "person_emp_exp": 0,
        "cb_person_cred_hist_length": 2,
        "credit_score": 504
    }
}
     
# App
st.title("Loan Eligibility Predictor 🏦💸")

#initialize session state
if "current_preset" not in st.session_state:
        st.session_state.gender = "Male"
        st.session_state.education = "High School"
        st.session_state.home_ownership = "Rent"
        st.session_state.loan_intent = "Personal"
        st.session_state.previous_loans = "No"
        st.session_state.income = 0
        st.session_state.loan_amount = 0
        st.session_state.loan_rate = 0
        st.session_state.credit_score = 0
        st.session_state.credit_hist_length = 0
        st.session_state.person_emp_exp = 0
        st.session_state.age = 0

with st.sidebar:
    selected_preset = "None"
    
    if st.button("Eligible for Loan 🎉"):
            selected_preset = "Eligible for Loan 🎉"
    if st.button("Not Eligible for Loan 🥹"):
            selected_preset = "Not Eligible for Loan 🥹"

if selected_preset != "None":    
        preset = presets[selected_preset]
        st.session_state.gender = preset["person_gender"]
        st.session_state.education = preset["person_education"]
        st.session_state.home_ownership = preset["person_home_ownership"]
        st.session_state.loan_intent = preset["loan_intent"]
        st.session_state.previous_loans = preset["previous_loan_default"]
        st.session_state.income = preset["Annual_Income"]
        st.session_state.loan_amount = preset["loan_amnt"]
        st.session_state.loan_rate = preset["loan_int_rate"]
        st.session_state.credit_score = preset["credit_score"]
        st.session_state.credit_hist_length = preset["cb_person_cred_hist_length"]
        st.session_state.person_emp_exp = preset["person_emp_exp"]
        st.session_state.age = preset["person_age"]
else:
    gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(st.session_state.gender))
    education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
                            index=["High School", "Associate", "Bachelor", "Master", "Doctorate"].index(st.session_state.education))
    home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage", "Other"],
                                index=["Rent", "Own", "Mortgage", "Other"].index(st.session_state.home_ownership))
    loan_intent = st.selectbox("Loan Intent", ["Education", "Medical", "Venture", "Personal", "Home Improvement", "Debt Consolidation"],
                            index=["Education", "Medical", "Venture", "Personal", "Home Improvement", "Debt Consolidation"].index(st.session_state.loan_intent))
    previous_loans = st.selectbox("Previous Loan Default", ["Yes", "No"],
                                index=["Yes", "No"].index(st.session_state.previous_loans))

    income = st.number_input("Annual Income ($)", min_value=0, value=st.session_state.income)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=st.session_state.loan_amount)
    loan_rate = st.number_input("Interest Rate (%)", min_value=0.0, format="%.2f", value=st.session_state.loan_rate)
    credit_score = st.number_input("Credit Score", min_value=0, value=st.session_state.credit_score)
    credit_hist_length = st.number_input("Credit Duration (in one year)", min_value=0, value=st.session_state.credit_hist_length)
    person_emp_exp = st.number_input("Work Experience (in years)", min_value=0, value=st.session_state.person_emp_exp)
    age = st.number_input("Age", min_value=18, value=st.session_state.age)


model = XGB_Classifier()

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "person_income": income,
        "person_age": age,
        "loan_amnt": loan_amount,
        "loan_int_rate": loan_rate,
        "person_gender": gender.strip().lower(),
        "person_education": education.strip(),
        "person_home_ownership": home_ownership.strip().upper(),
        "loan_intent": loan_intent.strip().upper(),
        "previous_loan_default": previous_loans.strip().capitalize(),
        "loan_percent_income": loan_amount / income if income else 0,
        "person_emp_exp": person_emp_exp,
        "cb_person_cred_hist_length": credit_hist_length,
        "credit_score": credit_score
    }])


    # Make the prediction
    with st.spinner("🧠 Predicting..."):
        time.sleep(random.uniform(0.5, 2.0))
    
    prediction = model.predict(input_data)

    if prediction == 1:
        st.success("Eligible for Loan! 🎉")
    else:
        st.error("Not Eligible for Loan. 🥹")