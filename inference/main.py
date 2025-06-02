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
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA/GPU")

# LLM
llm_model_id = "deepseek-ai/deepseek-llm-7b-chat"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
llm_model.to("cuda:0")      # Move model to GPU
llm_model.eval()            # Set model to evaluation mode

allocated = torch.cuda.memory_allocated() / (1024 ** 2)
reserved = torch.cuda.memory_reserved() / (1024 ** 2)

# Coder
coder_model_id = "TroyDoesAI/MermaidStable3B"
coder_tokenizer = AutoTokenizer.from_pretrained(coder_model_id)
coder_model = AutoModelForCausalLM.from_pretrained(
    coder_model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
coder_model.to("cuda:0")    # Move model to GPU
coder_model.eval()          # Set model to evaluation mode

# ASR
speech_model_id = "openai/whisper-large-v3"
speech_processor = AutoProcessor.from_pretrained(speech_model_id)
speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    speech_model_id, 
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
speech_model.to("cuda:0")   # Move model to GPU
speech_model.eval()         # Set model to evaluation mode

# ZSC 
classifier_model_id = "facebook/bart-large-mnli"
classifier = pipeline(
    "zero-shot-classification",
    model=classifier_model_id,
    device=torch.device("cuda:0").index
)

# RAG
rag_model_id = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(rag_model_id)
embedder.to("cuda:0")

## ChromaDB client and collection
chroma_client = chromadb.Client(Settings())
chroma_collection = chroma_client.get_or_create_collection(name="documents")

"""
    This function classifies the prompt into one of the labels.
"""
def classify_prompt(prompt):
    candidate_labels = [
        # Routes to MermaidStable3B
        "flowchart_diagram",
        "uml_diagram",
        "sequence_diagram",
        "state_machine",
        "graph_visualization",

        # Routes to LLM
        "text_explanation",
        "math_reasoning",
        "data_insight",
        "general_question",
        "code_explanation",
    ]

    result = classifier(prompt, candidate_labels)
    top_label = result["labels"][0]

    return top_label

"""
    This function retreives relevant embedded data
    from ChromaDB based on embedded user prompt.
"""
def retrieve_relevant_context(prompt, k=3):
    # Embeds the prompt using the embedder to match prompt with embeddings
    query_embedding = embedder.encode([prompt])[0].tolist()

    # Query the ChromaDB database for top k documents
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    # Separate top hits with "\n\n"
    response = "\n\n".join(results["documents"][0])

    return response

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
    
    label = classify_prompt(prompt)

    print(f"\n\nClassified prompt as: {label}\n\n", flush=True)

    if (
        label == "text_explaination" or \
        label == "general_question" or \
        label == "code_explanation" or \
        label == "math_reasoning" or \
        label == "data_insight"
        ):
        simple_label = "text"
        response_model = llm_model
        response_tokenizer = llm_tokenizer
    else:
        simple_label = "mermaid"
        response_model = coder_model
        response_tokenizer = coder_tokenizer

    inputs = response_tokenizer(full_prompt, return_tensors="pt").to(response_model.device)
    with torch.no_grad():
        outputs = response_model.generate(**inputs, max_new_tokens=600, temperature=0.7)
    generated = response_tokenizer.decode(outputs[0], skip_special_tokens=True)

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

    response = f"alloc {allocated}MB and reserved {reserved}MB\n"

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

    embeddings = embedder.encode(chunks).tolist()

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
