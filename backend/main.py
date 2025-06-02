#!/usr/bin/env python3

# Flask and communication libraries
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import openai
# General imports
import requests
import json
from pymongo import MongoClient

# RAG imports
import fitz
import uuid
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

client = OpenAI(api_key="you need your own")

chunk_dict = {} # dictionary that keeps track of all chunks text
audio_file_names = {} # dictionary that keeps track of corresponding file names of chunks

def create_conversation(data, prompt, history):
    decoded_history = json.loads(history)

    conversation = ""
    for entry in decoded_history:
        role = entry.get("role")
        content = entry.get("content")
        conversation += f"{role.capitalize()}: {content}\n"
    if prompt:
        conversation += f"User: {prompt}\n"

    data["history"] = history
    data["conversation"] = conversation

def relay_audio_stream(url, data, files=None):
    with requests.post(url, data=data, files=files, stream=True) as response:
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                yield chunk
"""
    Cleans a chunk from any newline characters.
    Returns a 'clean' chunk
"""
def clean_chunk(text):
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


"""
    Receives a document worth of text, sends it to GPT4 and receives the document in JSON formatted chunks
    Returns a dictionary with all chunks received from the GPT model.
"""

def chunk_text_via_gpt(full_text):
    prompt = f"""
You are a narrator AI. Your task is to segment the following document into narratable chunks.

Each chunk should:
- Be coherent and self-contained
- Be around 300–500 words (unless a natural section is shorter)
- Preserve logical or narrative boundaries (like paragraphs or scenes)

Respond only with a JSON array of strings, where each string is a chunk of the original text.

Text:
\"\"\"
{full_text}
\"\"\"
"""
    try:
        response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
    except Exception as e:
        raise RuntimeError(f"GPT chunking request failed: {e}")
    

    output_text = response.choices[0].message.content

    # Strip code fences if present
    output_text_clean = (
        output_text
        .strip()
        .removeprefix("```json")
        .removesuffix("```")
        .strip()
    )
    try:
        chunk_list = json.loads(output_text_clean)
        if not isinstance(chunk_list, list):
            raise ValueError("Expected a JSON array of strings.")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse GPT output as JSON: {e}")
    
    out_dict: dict[int, str] = {}
    for i, chunk_text in enumerate(chunk_list):
        if not isinstance(chunk_text, str):
            raise RuntimeError(f"Chunk #{i} is not a string.")
        chunk_text = clean_chunk(chunk_text)
        out_dict[i] = chunk_text.strip()
    return out_dict

"""
    Supposed to send a chunk to the inference server and receive and store
    the audio file it receives, but does for some reason not work...
"""
def send_chunk_for_narration(text, number):
    data = {}
    data["prompt"] = text
    url = "http://localhost:5000/narrateText"

    def save_audio_stream():
            with requests.post(url, data=data, stream=True) as response:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        with open(f"./audio/output_{number}.wav", "ab") as f:
                            f.write(chunk)
                            audio_file_names[number] = f"output_{number}.wav"

    try:    
        save_audio_stream()
    except (requests.exceptions.RequestException, RuntimeError) as e:
        raise RuntimeError(f"TTS failed for chunk {number}: {e}")


    return 0

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return Response("API is running", mimetype='text/plain')

@app.route("/upload/speech/", methods=["POST"])
def upload_speech():
    speech = request.files.get("audio")
    history = request.form.get("history")
    tts = request.form.get("tts")

    data = {}

    if history:
        create_conversation(data, False, history)
    data["tts"] = tts

    files = {"audio": (speech.filename, speech.stream, speech.mimetype)}

    url = "http://localhost:5001/recognizeTextFromSpeech"

    if tts:
        return Response(stream_with_context(relay_audio_stream(url, data, files)), mimetype='audio/wav')
    else:
        response = requests.post(url, files=files, data=data)
        text_response = response.text

        return Response(text_response, mimetype='text/plain')

@app.route("/upload/text/", methods=["POST"])
def upload_text():
    prompt = request.form.get("prompt")
    history = request.form.get("history")
    tts = request.form.get("tts")

    data = {}
    if history:
        create_conversation(data, prompt, history)
    data["prompt"] = prompt
    data["tts"] = tts

    if tts:
        url = "http://localhost:5001/generateSpeechFromText"
        
        return Response(stream_with_context(relay_audio_stream(url, data)), mimetype='audio/wav')
    
    else:
        url = "http://localhost:5001/generateTextFromText"
        response = requests.post(url, data=data)
        text_response = response.text

        return Response(text_response, mimetype='text/plain')

@app.route("/upload/file/", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text() + "\n"

    """
        RAG chunking and embedding
    """
    def chunk_document():
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_text(text)
    chunks = chunk_document()

    url = "http://localhost:5001/generateEmbeddings"
    data = {
        "texts": chunks,
    }
    response = requests.post(url, json=data)
    embeddings = response.json().get("embeddings")

    embeddings_array = np.array(embeddings)
    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)

    url = "http://localhost:5001/indexEmbeddings"
    data = {
        "chunks": chunks,
        "embeddings": embeddings,
        "doc_id": str(uuid.uuid4()),
    }
    response = requests.post(url, json=data)
    if response.status_code != 200:
        return Response("Failed to store file", status=500, mimetype='text/plain')
    else:
        return Response("Successfully stored file", status=200, mimetype='text/plain')


@app.route("/upload/file/narrate", methods=["POST"])
def upload_file_for_narration():
    #file = request.files.get("file")
    file = fitz.open("report.pdf")
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        full_text = "".join(page.get_text() for page in doc)
    chunk_dict = chunk_text_via_gpt(full_text)
    i = 0
    for chunk in chunk_dict.keys():
        if send_chunk_for_narration(chunk_dict[i]):
            print("success")
        else:
            print("Failure")
    return 0





if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
