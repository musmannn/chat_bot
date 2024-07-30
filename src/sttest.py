import streamlit as st

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

files = st.file_uploader(
    "Upload some files",
    accept_multiple_files=True,
    key=st.session_state["file_uploader_key"],
)


if st.button("Clear uploaded files"):
    st.session_state["file_uploader_key"] += 1
    st.rerun()

