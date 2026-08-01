"""
Application Header
"""

import streamlit as st


def show_header():

    st.title("📝 LLM Generated Text Detection")

    st.markdown(
        """
Detect whether a piece of text is **Human Written**
or **AI Generated** using an LSTM-based Deep Learning model.
"""
    )

    st.divider()