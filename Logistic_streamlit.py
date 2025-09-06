import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

st.title("Breast Cancer Risk Predictor")
st.header("This app helps you understand whether a breast tumor is likely to be benign (non-cancerous) or malignant (cancerous) based on a few basic measurements from a medical test.")
st.text("🧪 Note: This is a basic demonstration and should not be used for medical decisions.")
st.header("Enter the following measurements from your test report")
mean_radius = st.number_input("Mean radius(size of cell nuclei): ")
mean_texture = st.number_input("Mean Texture(variation in texture): ")
mean_perimeter = st.number_input("Mean perimeter(boundary length ): ")
mean_area = st.number_input("Mean area(area of nuclei): ")

data = load_breast_cancer()
X = data.data
y = data.target
result_placeholder = st.empty()

def predict():
    input_values = [[mean_radius, mean_texture, mean_perimeter, mean_area] + [0] * (X.shape[1] - 4)]
    model = LogisticRegression(max_iter=10000)
    model.fit(X, y)
    ans = model.predict(input_values)
    if ans == 1:
        result_placeholder.success("You are benign")
    else:
        result_placeholder.error("You have breast cancer")
st.header("Prediction result")
st.button("click here to predict",on_click = predict)


