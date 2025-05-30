#!/usr/bin/env python3

# Flask and communication libraries
import io
from flask import Flask, request, jsonify, send_file, Response, stream_with_context

# General AI models imports
import torch
from transformers import pipeline

# LLM imports
from transformers import AutoTokenizer, AutoModelForCausalLM

# Text-to-Speech imports
from kokoro import KPipeline
from IPython.display import Audio
import soundfile as sf

# Automatic Speech Recognition imports
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset

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

speech_model_id = "openai/whisper-large-v3"
speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    speech_model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
speech_model.to("cuda:0")

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0
)

app = Flask(__name__)

def classify_prompt(prompt):
    candidate_labels = ["text_summary", "image_generation", "text_to_text"]
    result = classifier(prompt, candidate_labels)

    return result["labels"][0]

@app.route("/")
def index():
    return jsonify({"message": "Inference API is up and running"})

@app.route("/recognizeSpeechFromText", methods=["POST"])
def recognize_speech_from_text():
    data = request.get_json()

    speech = data.get("speech")


@app.route("/generateSpeechFromText", methods=["POST"])
def generate_speech_from_text():
    data = request.get_json()

    prompt = data.get("prompt")
    history = data.get("history")
    conversation = data.get("conversation")

    inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=600, temperature=0.7)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Botty:" in generated:
        response = generated.rsplit("Botty:", 1)[-1].strip()
    else:
        response = generated.strip()

    def generate_audio_stream():
        pipeline = KPipeline(lang_code='a')
        generator = pipeline(response, voice='af_heart')

        for _, _, audio in generator:
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio, 24000, format='WAV')
            audio_buffer.seek(0)
            yield audio_buffer.read()

    return Response(stream_with_context(generate_audio_stream()), mimetype='audio/wav')

@app.route("/generateTextString", methods=["POST"])
def text_to_text():
    data = request.get_json()

    prompt = data.get("prompt")
    history = data.get("history")
    conversation = data.get("conversation")

    inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=600, temperature=0.7)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Botty:" in generated:
        response = generated.rsplit("Botty:", 1)[-1].strip()
    else:
        response = generated.strip()

    return jsonify({"response": response, "audio": "output.wav"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)