import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("IRIS.csv")

# Features and target
X = df.drop("species", axis=1)
y = df["species"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Title
st.title("🌸 Iris Flower Prediction App")

st.write("Adjust the sliders to predict the iris species:")

# User inputs
sepal_length = st.slider("Sepal Length", float(df["sepal_length"].min()), float(df["sepal_length"].max()))
sepal_width = st.slider("Sepal Width", float(df["sepal_width"].min()), float(df["sepal_width"].max()))
petal_length = st.slider("Petal Length", float(df["petal_length"].min()), float(df["petal_length"].max()))
petal_width = st.slider("Petal Width", float(df["petal_width"].min()), float(df["petal_width"].max()))

# Prediction
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)

# Output
st.subheader("Prediction:")
st.success(prediction[0])
