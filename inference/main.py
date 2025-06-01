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
import torchaudio
import torchaudio.transforms as T

# RAG imports
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA/GPU")

# LLM
model_id = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# ASR
speech_model_id = "openai/whisper-large-v3"
speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    speech_model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
speech_model.to("cuda:0")
speech_processor = AutoProcessor.from_pretrained(speech_model_id)

# ZSC 
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0
)

# RAG
embedder = SentenceTransformer("all-MiniLM-L6-v2")

## ChromaDB client and collection
chroma_client = chromadb.Client(Settings())
chroma_collection = chroma_client.get_or_create_collection(name="documents")

"""
    This function classifies the prompt into one of the labels.
"""
def classify_prompt(prompt):
    candidate_labels = ["text_summary", "image_generation", "text_to_text"]
    result = classifier(prompt, candidate_labels)

    return result["labels"][0]

"""
    This function retreives relevant embedded data
    from ChromaDB based on embedded user prompt.
"""
def retrieve_relevant_context(prompt, k=3):
    query_embedding = embedder.encode([prompt])[0].tolist()

    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return "\n\n".join(results["documents"][0])

"""
    This function generates a response based on the conversation history
    including the latest user input.
"""
def generate_response(conversation, prompt):
    context = retrieve_relevant_context(prompt)

    full_prompt = f"""
        You are a helpful assistant. Use the following 
        context to help answer the user's question.

        Context:
        {context}

        Conversation so far:
        {conversation}

        User: {prompt}
        Botty:"""

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=600, temperature=0.7)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Botty:" in generated:
        response = generated.rsplit("Botty:", 1)[-1].strip()
    else:
        response = generated.strip()

    return response

"""
    This function generates an audio stream from the response using 
    the Kokoro TTS pipeline with a generator that yields audio chunks
    (for streaming, not sending).
"""
def generate_audio_stream(response):
    pipeline = KPipeline(lang_code='a')
    generator = pipeline(response, voice='af_heart')

    for _, _, audio in generator:
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio, 24000, format='WAV')
        audio_buffer.seek(0)
        yield audio_buffer.read()

app = Flask(__name__)

@app.route("/")
def index():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)

    response = f"""
    API running with alloc {allocated}MB and reserved {reserved}MB
    """

    return Response(response, mimetype='text/plain')

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    return jsonify({
        "cuda_memory_allocated_mb": allocated,
        "cuda_memory_reserved_mb": reserved
    })

"""
    This endpoint is used to reccognize text from input speech.
"""
@app.route("/recognizeTextFromSpeech", methods=["POST"])
def recognize_text_from_speech():
    speech = request.files.get("audio")
    history = request.form.get("history")
    conversation = request.form.get("conversation")
    tts = request.form.get("tts")

    audio_bytes = speech.read()
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))

    """
        Whisper requires a sample rate of 16000 Hz,
        most microphones use 44100 Hz or 48000 Hz.
    """
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform_np = waveform.squeeze().numpy()

    pipeline_asr = pipeline(
        "automatic-speech-recognition",
        model=speech_model,
        tokenizer=speech_processor.tokenizer,
        feature_extractor=speech_processor.feature_extractor,
        torch_dtype=torch.float16,
        device=0
    )

    """
        Recongizes the prompt from the speech and appends
        it to the conversation history.
    """
    result = pipeline_asr(waveform_np)
    prompt = result["text"]
    conversation += f"User: {prompt}\n"

    """
        Generates a response from the conversation history
        and decides whether to return text or text-to-speech.
    """
    response = generate_response(conversation, prompt)

    if tts:
        return Response(stream_with_context(generate_audio_stream(response)), mimetype='audio/wav')
    else:
        return Response(response, mimetype='text/plain')

@app.route("/generateSpeechFromText", methods=["POST"])
def generate_speech_from_text():
    prompt = request.form.get("prompt")
    history = request.form.get("history")
    conversation = request.form.get("conversation")

    response = generate_response(conversation, prompt)

    return Response(stream_with_context(generate_audio_stream(response)), mimetype='audio/wav')

@app.route("/generateTextFromText", methods=["POST"])
def text_to_text():
    prompt = request.form.get("prompt")
    history = request.form.get("history")
    conversation = request.form.get("conversation")

    response = generate_response(conversation, prompt)

    return Response(response, mimetype='text/plain')

@app.route("/generateEmbeddings", methods=["POST"])
def embed_text():
    data = request.get_json()
    chunks = data["texts"]

    embeddings = embedder.encode(chunks).tolist()  # Convert to JSON-safe format

    return jsonify({"embeddings": embeddings})

@app.route("/indexEmbeddings", methods=["POST"])
def store_embeddings():
    data = request.get_json()
    chunks = data["chunks"]
    embeddings = data["embeddings"]
    doc_id = data["doc_id"]

    ids = []
    metadatas = []

    for i in range(len(chunks)):
        ids.append(f"{doc_id}-chunk-{i}")
        metadatas.append({"doc_id": doc_id})

    chroma_collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    
    return Response("Embeddings stored successfully", status=200, mimetype='text/plain')


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)