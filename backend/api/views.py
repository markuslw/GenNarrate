from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import fitz
from TTS.api import TTS

# Create your views here.

@csrf_exempt
def upload_file(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({"message": "No file provided!"}, status=400)

        print("Received file:", file.name, file.size)

        pdf_bytes = file.read()
        text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text() + "\n"
        
        return JsonResponse({"message": "File uploaded successfully!"}, status=200)
    return JsonResponse({"message": "Wrong request method!"}, status=400)
