#!/usr/bin/env python3
from flask import Flask, request, jsonify
import fitz

import torch
from torch.serialization import add_safe_globals
from transformers import AutoTokenizer, AutoModelForCausalLM

from TTS.tts.configs.xtts_config import XttsConfig              # type: ignore
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs       # type: ignore
from TTS.config.shared_configs import BaseDatasetConfig         # type: ignore

app = Flask(__name__)

model_id = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

@app.route("/")
def index():
    return jsonify({"message": "API is up and running"})

@app.route("/generateTextString", methods=["POST"])
def text_to_text():
    data = request.get_json()
    text = data["text"]

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=700, temperature=0.8)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response})

@app.route("/generateVoiceAudio", methods=["POST"])
def text_to_speech():
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

    tts.tts_to_file(
        text=ai_text,
        file_path="output2.wav",
        speaker_wav="female.wav",
        language="en"
    )
    
    return jsonify({"message": "Audio generated successfully!"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
