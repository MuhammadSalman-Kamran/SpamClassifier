import streamlit as st
from src.pipeline.prediction_pipeline import Prediction
pred_obj = Prediction()

st.set_page_config(page_title='Spam/Ham')
st.title('Spam Classifier')
user_input = st.text_area('Enter the text')
button = st.button('Predict')

result = pred_obj.prediction(user_input)

if button:
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')