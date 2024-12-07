import streamlit as st
import pandas as pd
import pickle
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.train_pipeline import Models_report
from matplotlib import pyplot as plt


## Loading Score Dictionary
with open("artifacts/score.pkl", "rb") as file:
    score_dict = pickle.load(file)


st.title("Student Score Prediction")

st.subheader("Step1: Select Model to Visualize Performance")
model_names = list(score_dict.keys())
selected_model = st.multiselect(
    "Select models to look at the R2 Score on Test Dataset",
    model_names
)

if selected_model:
    selected_report = {model:score_dict[model] for model in selected_model}
    selected_df = pd.DataFrame(list(selected_report.items()), columns=["Model", "R2 Score"])
    selected_df = selected_df.sort_values(by="R2 Score", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.barh(selected_df["Model"], selected_df["R2 Score"], color="skyblue")
    plt.xlabel("R2 Score")
    plt.ylabel("Model")
    plt.title("R2 Scores of Selected Models")
    st.pyplot(plt)


st.subheader("Step 3: Input Features for Prediction")
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