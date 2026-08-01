import streamlit as st

from components.header import show_header
from components.footer import show_footer
from components.input_section import get_input_text
from components.prediction_card import show_prediction

from src.predict import predict_text


st.set_page_config(
    page_title="LLM Generated Text Detection",
    page_icon="📝",
    layout="centered",
)

show_header()

text, predict = get_input_text()

if predict:

    if text.strip() == "":

        st.error("Please enter some text.")

    else:

        with st.spinner("Analyzing text..."):

            result = predict_text(text)

        show_prediction(result)

show_footer()