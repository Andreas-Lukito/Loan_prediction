# This is an streamlit app that will be used to predict if a person is eligible for a loan or not
import numpy as np
import streamlit as st
import pandas as pd
import os
import pickle
from pandas import DataFrame
import matplotlib.pyplot as plt

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
        
    def predict(self, data: DataFrame):
        # Ensure 'input_data' has the expected columns and values
        if isinstance(data, pd.DataFrame):
            # Add missing columns with default values, ensuring no mismatch in row length
            data["person_emp_exp"] = 0
            data["cb_person_cred_hist_length"] = 0
            data["credit_score"] = 0
            data["loan_percent_income"] = loan_amount / income if income else 0
        else:
            # Handle the case where input_data is not a DataFrame
            raise ValueError("Input data is not in the expected DataFrame format")
        
        home_ownership_encoded = pd.DataFrame(
            self.encoders.home_ownership.transform(data[["person_home_ownership"]]),
            columns=self.encoders.home_ownership.get_feature_names_out(["person_home_ownership"])
        )

        loan_intent_encoded = pd.DataFrame(
            self.encoders.loan_intent.transform(data[["loan_intent"]]),
            columns=self.encoders.loan_intent.get_feature_names_out(["loan_intent"])
        )

        data["person_gender"] = self.encoders.gender.transform(data["person_gender"])
        data["previous_loan_default"] = self.encoders.previous_loans.transform(data["previous_loan_default"])
        data["person_education"] = self.encoders.education.transform(data[["person_education"]])

        # merge all data
        data = data.drop(columns=["person_home_ownership", "loan_intent"])
        data = pd.concat([data, home_ownership_encoded, loan_intent_encoded], axis=1)
        
        return self.xgb_model.predict(data)

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
        
# App
st.title("Loan Eligibility Predictor 🏦💸")

gender = st.selectbox("Gender", ["male", "female"])
education = st.selectbox("Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage", "Other"])
loan_intent = st.selectbox("Loan Intent", ["Education", "Medical", "Venture", "Personal", "Home Improvement", "Debt Consolidation"])
previous_loans = st.selectbox("Previous Loan Default", ["Yes", "No"])

income = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_rate = st.number_input("Interest Rate (%)", min_value=0.0, format="%.2f")
loan_term = st.number_input("Loan Term (in months)", min_value=0)
age = st.number_input("Age", min_value=18)

model = XGB_Classifier()
st.write(model.xgb_model.feature_names_in_.tolist())

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "person_income": income,
        "person_age": age,
        "loan_amnt": loan_amount,
        "loan_int_rate": loan_rate,
        "loan_term": loan_term,
        "person_gender": gender.strip().lower(),
        "person_education": education.strip(),
        "person_home_ownership": home_ownership.strip().upper(),
        "loan_intent": loan_intent.strip().upper(),
        "previous_loan_default": previous_loans.strip().capitalize(),
        "loan_percent_income": loan_amount / income if income else 0,
        "person_emp_exp": 0,
        "cb_person_cred_hist_length": 0,
        "credit_score": 0
    }])


    # Make the prediction
    prediction = model.predict(input_data)

    if prediction == 1:
        st.success("Eligible for Loan!")
    else:
        st.error("Not Eligible for Loan.")