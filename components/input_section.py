"""
Input Section Component
"""

import streamlit as st


def get_input_text():
    """
    User input textbox.
    """

    text = st.text_area(
        "Paste your text below",
        height=250,
        placeholder="Enter essay or paragraph..."
    )

    predict = st.button(
        "Predict",
        use_container_width=True
    )

    return text, predict