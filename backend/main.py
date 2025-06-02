#!/usr/bin/env python3
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import requests
import json
from pymongo import MongoClient

import uuid

# RAG imports
import fitz
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
