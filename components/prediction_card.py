"""
Prediction Card Component
"""

import streamlit as st


def show_prediction(result):
    """
    Display prediction result.
    """

    prediction = result["prediction"]
    confidence = result["confidence"]

    if prediction == "Human Written":
        st.success(f"Prediction: {prediction}")
    else:
        st.warning(f"Prediction: {prediction}")

    st.metric(
        label="Confidence",
        value=f"{confidence*100:.2f}%"
    )

    st.progress(float(confidence))