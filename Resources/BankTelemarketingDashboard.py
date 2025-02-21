import pickle

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from BinPDays import bin_pdays
from ImportModel import import_model
from PipelineWrapper import PipelineWrapper


def rename_columns_encoding(X):
    columns_name = []

    for i in range(len(encoding_transformer.transformers_)):
        if i == 0:
            columns_name += ["Last_Contact_Within_30_Days"]
            continue

        columns_name += list(
            encoding_transformer.transformers_[i][1].get_feature_names_out()
        )

    X.columns = columns_name
    return X


def get_feat_names():
    features_name = []

    for i in range(3):
        if i == 0:
            features_name += ["Last_Contact_Within_30_Days"]
            continue

        features_name += list(
            encoding_transformer.transformers_[i][1].get_feature_names_out()
        )

    for i in range(2):
        features_name += list(
            scaling_transformer.transformers_[i][1].get_feature_names_out()
        )

    return features_name


# def load_model():
#     with open("Projects/Models/final_model.pkl", "rb") as file:
#         model = pickle.load(file)
#     return model


model = import_model()

prep_pipeline = model.named_steps["Preparation"].pipeline
feature_pipeline = prep_pipeline.named_steps["feature_engineering"].pipeline
model_pipeline = model.named_steps["Modeling"]

scaling_transformer = feature_pipeline.named_steps["Feature Scaling"]
encoding_transformer = feature_pipeline.named_steps["Feature Encoding"]

st.markdown(
    "<h1 style='text-align: center;'>Bank Telemarketing Prediction Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; font-weight: bold'>DTIDS-0206 Final Project by Alpha Team</p>",
    unsafe_allow_html=True,
)
st.divider()


st.sidebar.header("User Input Features")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
job = st.sidebar.selectbox(
    "Job",
    [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    ],
)
marital = st.sidebar.selectbox(
    "Marital Status", ["divorced", "married", "single", "unknown"]
)
education = st.sidebar.selectbox(
    "Education",
    [
        "basic.4y",
        "basic.6y",
        "basic.9y",
        "high.school",
        "illiterate",
        "professional.course",
        "university.degree",
        "unknown",
    ],
)
default = st.sidebar.selectbox("Has Credit in Default?", ["no", "yes", "unknown"])
housing = st.sidebar.selectbox("Has Housing Loan?", ["no", "yes", "unknown"])
loan = st.sidebar.selectbox("Has Personal Loan?", ["no", "yes", "unknown"])
contact = st.sidebar.selectbox("Contact Type", ["cellular", "telephone"])
month = st.sidebar.selectbox(
    "Last Contact Month",
    ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
)
day_of_week = st.sidebar.selectbox(
    "Last Contact Day of the Week", ["mon", "tue", "wed", "thu", "fri"]
)
duration = st.sidebar.number_input(
    "Last Contact Duration (seconds)", min_value=0, value=180
)
campaign = st.sidebar.number_input("Number of Contacts Performed", min_value=1, value=1)
pdays = st.sidebar.number_input("Days Since Last Contact", min_value=-1, value=999)
previous = st.sidebar.number_input("Number of Previous Contacts", min_value=0, value=0)
poutcome = st.sidebar.selectbox(
    "Previous Campaign Outcome", ["failure", "nonexistent", "success"]
)
emp_var_rate = st.sidebar.number_input("Employment Variation Rate", value=-1.8)
cons_price_idx = st.sidebar.number_input("Consumer Price Index", value=93.994)
cons_conf_idx = st.sidebar.number_input("Consumer Confidence Index", value=-36.4)
euribor3m = st.sidebar.number_input("Euribor 3 Month Rate", value=4.857)
nr_employed = st.sidebar.number_input("Number of Employees", value=5191.0)

input_data = pd.DataFrame(
    {
        "age": [age],
        "job": [job],
        "marital": [marital],
        "education": [education],
        "default": [default],
        "housing": [housing],
        "loan": [loan],
        "contact": [contact],
        "month": [month],
        "day_of_week": [day_of_week],
        "duration": [duration],
        "campaign": [campaign],
        "pdays": [pdays],
        "previous": [previous],
        "poutcome": [poutcome],
        "emp.var.rate": [emp_var_rate],
        "cons.price.idx": [cons_price_idx],
        "cons.conf.idx": [cons_conf_idx],
        "euribor3m": [euribor3m],
        "nr.employed": [nr_employed],
    }
)

st.markdown(
    """
    <style>
    .prediction-container {
        background-color: #1B4965;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container():
    if st.sidebar.button("Predict"):
        prediction = model.predict(input_data)[0]
        y_pred_proba = model.predict_proba(input_data)[:, 1]

        prediction_text = (
            "likely to open a deposit account"
            if prediction == 1
            else "not likely to open a deposit account"
        )

        st.markdown(
            f"""
            <div class='prediction-container'>
                <h4>This customer is {prediction_text}</h4>
                <p>With the probability of opening a deposit account: <b>{y_pred_proba[0]:.2f}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container():
            st.markdown(
                """
                <style>
                .abc-container {
                    background-color: #1B4965;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 20px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div class='abc-container'><h4>Feature Importance using SHAP</h4>",
                unsafe_allow_html=True,
            )

            input_transformed = prep_pipeline.transform(input_data)
            input_transformed = pd.DataFrame(
                input_transformed, columns=get_feat_names()
            )

            explainer = shap.TreeExplainer(model_pipeline)
            shap_values = explainer.shap_values(input_transformed)

            fig, ax = plt.subplots(figsize=(8, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    feature_names=get_feat_names(),
                ),
                show=False,
            )
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("</div>", unsafe_allow_html=True)
