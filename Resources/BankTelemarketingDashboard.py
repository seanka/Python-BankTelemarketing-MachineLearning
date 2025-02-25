import pickle

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from FeatureEncodeMethods import bin_pdays, change_education, change_job
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

    for i in range(4):
        if i == 0:
            features_name += ["Last_Contact_Within_30_Days"]
            continue

        features_name += list(
            encoding_transformer.transformers_[i][1].get_feature_names_out()
        )

        for i in range(2):
            if i == 0:
                continue

            features_name += list(
                scaling_transformer.transformers_[i][1].get_feature_names_out()
            )

        col_to_remove = {
            "previous",
            "emp.var.rate",
            "loan_yes",
            "day_of_week_tue",
            "job_management",
            "nr.employed",
            "Last_Contact_Within_30_Days",
        }
        features_name = [col for col in features_name if col not in col_to_remove]

    return features_name


model = import_model()

prep_pipeline = model.named_steps["Preparation"].pipeline
feature_pipeline = prep_pipeline.named_steps["feature_engineering"].pipeline

pipeline_selection = model.named_steps["Feature Selection"]
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


st.sidebar.header("Input Features")
singleTab, batchTab = st.sidebar.tabs(["📗 Single Input", "📚 Batch Input"])

onTapPredictSingle = False
onTapPredictBatch = False

# TAB SINGLE INPUT
with singleTab:
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    job = st.selectbox(
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
        ],
    )
    marital = st.selectbox("Marital Status", ["divorced", "married", "single"])
    education = st.selectbox(
        "Education",
        [
            "basic.4y",
            "basic.6y",
            "basic.9y",
            "high.school",
            "illiterate",
            "professional.course",
            "university.degree",
        ],
    )
    default = st.selectbox("Has Credit in Default?", ["no", "yes"])
    housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
    loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone"])
    month = st.selectbox(
        "Last Contact Month",
        [
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ],
    )
    day_of_week = st.selectbox(
        "Last Contact Day of the Week",
        ["mon", "tue", "wed", "thu", "fri"],
    )
    duration = st.number_input(
        "Last Contact Duration (seconds)", min_value=0, value=180
    )
    campaign = st.number_input("Number of Contacts Performed", min_value=1, value=1)
    pdays = st.number_input("Days Since Last Contact", min_value=-1, value=999)
    previous = st.number_input("Number of Previous Contacts", min_value=0, value=0)
    poutcome = st.selectbox(
        "Previous Campaign Outcome", ["failure", "nonexistent", "success"]
    )
    emp_var_rate = st.number_input("Employment Variation Rate", value=-1.8)
    cons_price_idx = st.number_input("Consumer Price Index", value=93.994)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4)
    euribor3m = st.number_input("Euribor 3 Month Rate", value=4.857)
    nr_employed = st.number_input("Number of Employees", value=5191.0)

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

    if st.button("Predict Single Outcome", key="onTapPredictSingle"):
        onTapPredictSingle = True

with batchTab:
    st.write("Import a CSV containing records to be predicted")

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        batch_input_data = pd.read_csv(uploaded_file)

    if st.button("Predict Batch Outcome", key="onTapPredictBatch"):
        onTapPredictBatch = True


# PREDICTION RESULT CONTAINER
with st.container():
    st.markdown(
        """
        <style>
        .prediction-container {
            background-color: #1B4965;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .feat-imp-container {
            margin-top: 20px;
            background-color: #1B4965;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if onTapPredictSingle:
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
                "<div class='feat-imp-container'><h4>Feature Importance using SHAP</h4>",
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

    elif onTapPredictBatch:
        prediction_results = []
        y_pred_proba_results = []

        for i in range(batch_input_data.shape[0]):
            input_data_loop = batch_input_data.iloc[[i]]

            prediction_results.append(model.predict(input_data_loop)[0])
            y_pred_proba_results.append(model.predict_proba(input_data_loop)[:, 1])

        prediction_dict = {0: 0, 1: 0}
        for result in prediction_results:
            prediction_dict[result] += 1

        st.markdown(
            f"""
            <div class='prediction-container'>
                <h4>{prediction_dict[1]} customers are likely will open a deposit account,</h4>
                <h5>while {prediction_dict[0]} will not.</h5>
                <p></p>
                <p style='text-align:left'>Deposit Revenue&emsp;&emsp;:&emsp;<b>€{prediction_dict[1] * 7.5}</b></p>
                <p style='text-align:left'>Saved Cost&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;:&emsp;<b>€{prediction_dict[0] * 1.0944}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='feat-imp-container'><h4>Feature Importance using SHAP</h4>",
            unsafe_allow_html=True,
        )

        test_shap = prep_pipeline.transform(batch_input_data)
        test_shap = pipeline_selection.transform(test_shap)
        test_shap.columns = get_feat_names()

        explainer = shap.TreeExplainer(model_pipeline)
        shap_values = explainer.shap_values(test_shap)

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
