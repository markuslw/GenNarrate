#!/usr/bin/env python3
from flask import Flask, request, jsonify                       # type: ignore

import torch                                                    # type: ignore
from torch.serialization import add_safe_globals                # type: ignore

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import GenerationConfig
from transformers import pipeline

from TTS.tts.configs.xtts_config import XttsConfig              # type: ignore
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs       # type: ignore
from TTS.config.shared_configs import BaseDatasetConfig         # type: ignore

app = Flask(__name__)

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA/GPU")

model_id = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0
)

@app.route("/")
def index():
    return jsonify({"message": "Inference API is up and running"})

def classify_prompt(prompt):
    candidate_labels = ["text_summary", "image_generation", "text_to_text"]
    result = classifier(prompt, candidate_labels)

    return result["labels"][0]

@app.route("/generateTextString", methods=["POST"])
def text_to_text():
    data = request.get_json()

    prompt = data.get("prompt")
    history = data.get("history")
    conversation = data.get("conversation")

    label = classify_prompt(prompt)

    inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=700, temperature=0.8)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Botty:" in generated:
        response = generated.rsplit("Botty:", 1)[-1].strip()
    else:
        response = generated.strip()

    return jsonify({"response": response})

@app.route("/generateVoiceAudio", methods=["POST"])
def text_to_speech():
    data = request.get_json()
    text = data["text"]

    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

    tts.tts_to_file(
        text=text,
        file_path="output2.wav",
        speaker_wav="female.wav",
        language="en"
    )
    
    return jsonify({"message": "Audio generated successfully!"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
