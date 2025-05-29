#!/usr/bin/env python3
from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz
import requests
import json
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return jsonify({"message": "Backend API is up and running"})

@app.route("/upload/file/", methods=["POST"])
def upload_files():
    media = request.form.get("media")
    
    text = ""
    with fitz.open(stream=media, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text() + "\n"

    url = "http://localhost:5001/generateTextString"
    data = {
        "text": text,
    }

    response = requests.post(url, json=data)
    text_response = response.json().get("response")

    return jsonify({"message": text_response})

@app.route("/upload/text/", methods=["POST"])
def upload_text():
    prompt = request.form.get("prompt")
    history = request.form.get("history")
    file = request.files.get("file")

    if history:
        decoded_history = json.loads(history)

        conversation = ""
        for msg in decoded_history:
            role = msg["role"]
            text = msg["text"]
            conversation += f"{role.capitalize()}: {text}\n"
        conversation += f"User: {prompt}\n"
    
    url = "http://localhost:5001/generateTextString"
    data = {
        "prompt": prompt,
        "history": history,
        "conversation": conversation,
    }

    response = requests.post(url, json=data)
    text_response = response.json().get("response")

    return jsonify({"message": text_response})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
