import streamlit as st
import requests, fitz

# Configuration
REMOTE_SERVER_URL = "http://localhost:5001/"  # local forward from remote

st.set_page_config(page_title="GenNarrate Chat", layout="centered")
st.title("GenNarrate")

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf", label_visibility="collapsed")

if uploaded_file is not None and "uploaded_pdf_text" not in st.session_state:
    import fitz  # PyMuPDF

    pdf_text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            pdf_text += page.get_text()

    st.session_state["uploaded_pdf_text"] = pdf_text
    st.toast("PDF uploaded and processed successfully!", icon="✅")


# Input box
if prompt := st.chat_input("Ask about what you just heard..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send to remote server
    try:
        response = requests.get(REMOTE_SERVER_URL, timeout=10)
        if response.ok:
            reply = response.json().get("message", "[No response]")

        else:
            reply = f"[Server error: {response.status_code}]"
    except Exception as e:
        reply = f"[Request failed: {e}]"

    # Display assistant message
    st.chat_message("assistant").markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

