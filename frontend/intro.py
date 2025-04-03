import streamlit as st

st.set_page_config(
    page_title="Intro",
    page_icon="👋",
)

st.sidebar.success("Select a demo above.")

with open("../README.md", "r", encoding="utf-8") as file:
    file_content = file.read()
    st.markdown(file_content)

st.balloons()