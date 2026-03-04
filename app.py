import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

model = pickle.load(open("insurance_model_new.pkl","rb"))

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------

st.set_page_config(
    page_title="Health Insurance Cost Prediction",
    layout="wide"
)

st.title("🏥 Health Insurance Cost Prediction System")

# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------

section = st.sidebar.radio(
    "Navigation",
    ["About Data","EDA Dashboard","Prediction"]
)

# ---------------------------------------------------
# ABOUT DATA SECTION
# ---------------------------------------------------

if section == "About Data":

    st.header("📊 About the Dataset")

    st.write("""
This project predicts **medical insurance charges** based on personal attributes.

### Dataset Features
""")

    data = {
        "Feature":[
            "age",
            "sex",
            "bmi",
            "children",
            "smoker",
            "region",
            "charges"
        ],

        "Description":[
            "Age of the person",
            "Gender (Male/Female)",
            "Body Mass Index",
            "Number of children/dependents",
            "Smoking status",
            "Residential region",
            "Medical insurance cost"
        ]
    }

    df_info = pd.DataFrame(data)

    st.table(df_info)

    st.subheader("📌 Problem Statement")

    st.write("""
Insurance companies need to estimate medical costs for individuals.  
This project uses **Machine Learning** to predict expected insurance charges based on personal attributes.
""")

# ---------------------------------------------------
# EDA SECTION
# ---------------------------------------------------

elif section == "EDA Dashboard":

    st.header("📈 Exploratory Data Analysis")

    df = pd.read_csv("insurance.csv")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Summary Statistics")

    st.write(df.describe())

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("Age Distribution")

        fig, ax = plt.subplots()
        sns.histplot(df["age"], kde=True)
        st.pyplot(fig)

    with col2:

        st.subheader("BMI Distribution")

        fig, ax = plt.subplots()
        sns.histplot(df["bmi"], kde=True)
        st.pyplot(fig)

    col3,col4 = st.columns(2)

    with col3:

        st.subheader("Charges vs Age")

        fig, ax = plt.subplots()
        sns.scatterplot(x=df["age"], y=df["charges"])
        st.pyplot(fig)

    with col4:

        st.subheader("Smoker vs Charges")

        fig, ax = plt.subplots()
        sns.boxplot(x=df["smoker"], y=df["charges"])
        st.pyplot(fig)

# ---------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------

else:

    st.header("💰 Insurance Cost Prediction")

    col1,col2 = st.columns(2)

    with col1:

        age = st.slider("Age",18,100,30)

        sex = st.selectbox(
            "Sex",
            ["male","female"]
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0
        )

    with col2:

        children = st.slider(
            "Number of Children",
            0,
            5,
            1
        )

        smoker = st.selectbox(
            "Smoker",
            ["yes","no"]
        )

        region = st.selectbox(
            "Region",
            ["southwest","southeast","northwest","northeast"]
        )

    if st.button("Predict Insurance Cost"):

        input_data = pd.DataFrame({

            "age":[age],
            "sex":[sex],
            "bmi":[bmi],
            "children":[children],
            "smoker":[smoker],
            "region":[region]

        })

        prediction = model.predict(input_data)

        st.success(f"Estimated Insurance Charges: ${round(prediction[0],2)}")