import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("titanic_model.pkl")

# Title and description
st.title("Titanic Survival Prediction 🚢")
st.write("""
This app predicts whether a passenger would have survived the Titanic disaster based on the given inputs.
""")

# Input fields
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid (Fare)", min_value=0.0, max_value=600.0, value=30.0, step=1.0)
embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"])

# Convert categorical inputs to numerical values
sex_map = {"Male": 0, "Female": 1}
embarked_map = {"C": 0, "Q": 1, "S": 2}

input_data = np.array([
    pclass,
    sex_map[sex],
    age,
    sibsp,
    parch,
    fare,
    embarked_map[embarked]
]).reshape(1, -1)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]
    
    if prediction == 1:
        st.success(f"Prediction: Survived! (Confidence: {prob[1]*100:.2f}%)")
    else:
        st.error(f"Prediction: Did not survive. (Confidence: {prob[0]*100:.2f}%)")
