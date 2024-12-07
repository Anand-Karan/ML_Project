import streamlit as st
import pandas as pd
import pickle
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

model = pickle.load(open("artifacts/model.pkl", 'rb'))
preprocessor = pickle.load(open("artifacts/preprocessor.pkl", 'rb'))

st.title("Student Score Prediction")

# User Inputs
gender = st.selectbox("Gender", ["male", "female"])
race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental_level_of_education = st.selectbox("Parental Level of Education", ["high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, step=1)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100, step=1)


if st.button("Predict"):
    data = CustomData(
        gender = gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education = parental_level_of_education,
        lunch = lunch,
        test_preparation_course = test_preparation_course,
        reading_score = reading_score,
        writing_score = writing_score
    )

    features = data.get_data_as_data_frame()

    predictor = PredictPipeline()
    prediction = predictor.predict(features)

    st.success(f"Predicted Math Score: {prediction[0]}")