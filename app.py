import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# Function
def get_flower_image(species):
    if species == "Iris-setosa":
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Iris_setosa_flower.jpg/640px-Iris_setosa_flower.jpg"
    elif species == "Iris-versicolor":
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/640px-Iris_versicolor_3.jpg"
    else:
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/640px-Iris_virginica.jpg"

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
img_url = get_flower_image(prediction[0])
st.image(img_url, caption="Predicted Flower", use_container_width=True)

# Output
st.subheader("Prediction:")
st.success(prediction[0])
st.subheader("Input Values:")
input_df = pd.DataFrame(input_data, columns=X.columns)
st.write(input_df)
if st.button("Predict Species"):
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")
    
# Inputs layout
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length", value=5.0)
    sepal_width = st.number_input("Sepal Width", value=3.0)

with col2:
    petal_length = st.number_input("Petal Length", value=4.0)
    petal_width = st.number_input("Petal Width", value=1.0)

import matplotlib.pyplot as plt

probs = model.predict_proba(input_data)[0]

fig, ax = plt.subplots()
ax.bar(model.classes_, probs)

st.pyplot(fig)
st.subheader("Prediction Result")
st.markdown(f"### 🌿 {prediction[0]}")
st.info("This prediction is based on petal and sepal measurements using a Random Forest model.")
from sklearn.metrics import accuracy_score
st.write("Model Accuracy:", accuracy_score(y, model.predict(X)))
st.title("🌸 AI-Powered Iris Classifier")
