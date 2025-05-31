#!/usr/bin/env python3
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import fitz
import requests
import json
from pymongo import MongoClient

def create_conversation(data, prompt, history):
    decoded_history = json.loads(history)

    conversation = ""
    for entry in decoded_history:
        role = entry.get("role")
        text = entry.get("text")
        conversation += f"{role.capitalize()}: {text}\n"
    if prompt:
        conversation += f"User: {prompt}\n"

    data["history"] = history
    data["conversation"] = conversation

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return jsonify({"message": "Backend API is up and running"})

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

        def relay_audio_stream():
            with requests.post(url, data=data, stream=True) as response:
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        yield chunk
        
        return Response(stream_with_context(relay_audio_stream()), mimetype='audio/wav')
    
    else:
        url = "http://localhost:5001/generateTextFromText"
        response = requests.post(url, json=data)
        text_response = response.text

        return Response(text_response, mimetype='text/plain')
    
@app.route("/upload/chunk/", methods=["POST"])
def upload_chunk():
    prompt = request.form.get("prompt")
    history = request.form.get("history")
    tts = request.form.get("tts")

    data = {}
    if history:
        create_conversation(data, prompt, history)
    data["prompt"] = prompt

    url = "http://localhost:5001/narrate"

    def save_audio_stream():
        with requests.post(url, data=data, stream=True) as response:
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    with open("output.wav", "ab") as f:
                        f.write(chunk)
    save_audio_stream()
    
    return Response("Chunk stored", mimetype='text/plain')
    
@app.route("/upload/file/", methods=["POST"])
def upload_files():
    media = request.form.get("media")
    
    text = ""
    with fitz.open(stream=media, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text() + "\n"

    url = "http://localhost:5001/generateTextFromText"
    data = {
        "text": text,
    }

    response = requests.post(url, data=data)
    text_response = response.text

    return Response(text_response, mimetype='text/plain')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
