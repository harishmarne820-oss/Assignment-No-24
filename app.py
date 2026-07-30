import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Multi Model Prediction",
    page_icon="🤖"
)

st.title("🤖 Multi Model Prediction System")

problem = st.sidebar.selectbox(
    "Select Problem Type",
    ["Classification", "Regression"]
)

# Classification

if problem == "Classification":

    model_name = st.selectbox(
        "Select Algorithm",
        [
            "Logistic Regression",
            "Decision Tree",
            "SVM",
            "KNN",
            "Naive Bayes"
        ]
    )

    if model_name == "Logistic Regression":
        model = joblib.load("logistic_regression.pkl")

    elif model_name == "Decision Tree":
        model = joblib.load("decision_tree_classifier.pkl")

    elif model_name == "SVM":
        model = joblib.load("svm_classifier.pkl")

    elif model_name == "KNN":
        model = joblib.load("knn_classifier.pkl")

    else:
        model = joblib.load("naive_bayes.pkl")

    scaler = joblib.load("classification_scaler.pkl")
    columns = joblib.load("classification_columns.pkl")

    st.subheader("Enter Input Values")

    values = {}

    for col in columns:
        values[col] = st.number_input(col, value=0.0)

    if st.button("Predict"):

        df = pd.DataFrame([values])
        df = df[columns]

        df = scaler.transform(df)

        prediction = model.predict(df)

        if prediction[0] == 1:
            st.success("Heart Disease Detected")
        else:
            st.success("No Heart Disease")

# Regression

else:

    model_name = st.selectbox(
        "Select Algorithm",
        [
            "Linear Regression",
            "Decision Tree Regressor",
            "SVR",
            "KNN Regressor"
        ]
    )

    if model_name == "Linear Regression":
        model = joblib.load("linear_regression.pkl")

    elif model_name == "Decision Tree Regressor":
        model = joblib.load("decision_tree_regressor.pkl")

    elif model_name == "SVR":
        model = joblib.load("svr.pkl")

    else:
        model = joblib.load("knn_regressor.pkl")

    scaler = joblib.load("regression_scaler.pkl")
    columns = joblib.load("regression_columns.pkl")

    st.subheader("Enter Input Values")

    values = {}

    for col in columns:
        values[col] = st.number_input(col, value=0.0)

    if st.button("Predict"):

        df = pd.DataFrame([values])
        df = df[columns]

        df = scaler.transform(df)

        prediction = model.predict(df)

        st.success(f"Predicted House Price: {prediction[0]:.2f}")