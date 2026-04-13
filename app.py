import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# -----------------------------
# FUNCTION: Get flower image
# -----------------------------
def get_flower_image(species):
    if species == "Iris-setosa":
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Iris_setosa_flower.jpg/640px-Iris_setosa_flower.jpg"
    elif species == "Iris-versicolor":
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/640px-Iris_versicolor_3.jpg"
    else:
        return "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/640px-Iris_virginica.jpg"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("IRIS.csv")

X = df.drop("species", axis=1)
y = df["species"]

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = RandomForestClassifier()
model.fit(X, y)

# -----------------------------
# TITLE
# -----------------------------
st.title("🌸 AI-Powered Iris Classifier")
st.write("Enter flower measurements to predict the species")

# -----------------------------
# INPUTS (COLUMNS)
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length", value=5.0)
    sepal_width = st.number_input("Sepal Width", value=3.0)

with col2:
    petal_length = st.number_input("Petal Length", value=4.0)
    petal_width = st.number_input("Petal Width", value=1.0)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("Predict Species"):

    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

    prediction = model.predict(input_data)
    probs = model.predict_proba(input_data)[0]

    # -----------------------------
    # SHOW IMAGE
    # -----------------------------
    img_url = get_flower_image(prediction[0])
    st.image(img_url, caption="Predicted Flower", use_container_width=True)

    # -----------------------------
    # RESULT
    # -----------------------------
    st.subheader("Prediction Result")
    st.success(prediction[0])

    # -----------------------------
    # INPUT VALUES
    # -----------------------------
    st.subheader("Input Values")
    input_df = pd.DataFrame(input_data, columns=X.columns)
    st.write(input_df)

    # -----------------------------
    # CONFIDENCE CHART
    # -----------------------------
    st.subheader("Prediction Confidence")

    fig, ax = plt.subplots()
    ax.bar(model.classes_, probs)
    ax.set_ylabel("Probability")
    ax.set_title("Confidence Levels")

    st.pyplot(fig)

    # -----------------------------
    # CONFIDENCE TEXT
    # -----------------------------
    st.subheader("Confidence (%)")
    for i, class_name in enumerate(model.classes_):
        st.write(f"{class_name}: {probs[i]*100:.2f}%")

    # -----------------------------
    # MODEL INFO
    # -----------------------------
    st.info("This prediction is based on petal and sepal measurements using a Random Forest model.")
