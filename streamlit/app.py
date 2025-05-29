import streamlit as st
import requests, fitz, json
from pathlib import Path
from openai import OpenAI

# Configuration
REMOTE_SERVER_URL = "http://localhost:5001/"
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="GenNarrate Chat", layout="centered")
st.title("GenNarrate")

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Upload and process PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

if uploaded_file is not None and "uploaded_pdf_text" not in st.session_state:
    pdf_text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            pdf_text += page.get_text()

    st.session_state["uploaded_pdf_text"] = pdf_text
    st.toast("PDF uploaded and processed successfully!", icon="✅")

    # Send to GPT-4 for chunking
    prompt = f"""
You are a narrator AI. Your task is to segment the following document into narratable chunks.

Each chunk should:
- Be coherent and self-contained
- Be around 300–500 words (unless a natural section is shorter)
- Preserve logical or narrative boundaries (like paragraphs or scenes)

Respond only with a JSON array of strings, where each string is a chunk of the original text.

Text:
\"\"\"
{pdf_text}
\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        output_text = response.choices[0].message.content

        # Clean and save response to file
        output_text_clean = output_text.strip().removeprefix("```json").removesuffix("```").strip()
        Path("chunk_output.json").write_text(output_text_clean, encoding="utf-8")
        st.success("Chunked response saved to chunk_output.json")

        # Extract and display first chunk
        chunk_list = json.loads(output_text_clean)
        first_chunk = chunk_list[0] if chunk_list else "[No chunks extracted]"
        st.markdown(f"**First Chunk Preview:**\n\n{first_chunk}")

        # Send chunk to remote server for TTS
        with st.spinner("Sending to remote server and waiting for audio..."):
            tts_response = requests.post(
                f"{REMOTE_SERVER_URL}generateVoiceAudio",
                json={"text": first_chunk},
                timeout=60
            )

        if tts_response.ok:
            # Streamlit only plays from bytes or file; we fetch the file from remote
            audio_url = f"{REMOTE_SERVER_URL}output2.wav"
            audio_response = requests.get(audio_url, timeout=30)
            if audio_response.ok:
                st.audio(audio_response.content, format="audio/wav")
            else:
                st.error("Failed to retrieve audio file from remote server.")
        else:
            st.error("Remote TTS request failed.")

    except Exception as e:
        st.error(f"Processing failed: {e}")

# Input box
if prompt := st.chat_input("Ask about what you just heard..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        response = requests.get(REMOTE_SERVER_URL, timeout=10)
        if response.ok:
            reply = response.json().get("message", "[No response]")
        else:
            reply = f"[Server error: {response.status_code}]"
    except Exception as e:
        reply = f"[Request failed: {e}]"

    st.chat_message("assistant").markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
