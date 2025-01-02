from django.shortcuts import render
from django.http import JsonResponse
from .models import VoiceMessage
from .emotion_detection import detect_emotion_from_audio_file
from .chatbot_setup import setup_chatbot

chatbot = setup_chatbot()

def home(request):
    return render(request, 'home.html')

def get_response(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        response = chatbot.get_response(message)
        return JsonResponse({'response': str(response)})

def upload_voice_message(request):
    audio_file = request.FILES['audio_file']
    voice_message = VoiceMessage.objects.create(audio_file=audio_file)
    emotion = detect_emotion_from_audio_file(voice_message.audio_file.path)
    voice_message.emotion = emotion
    voice_message.save()
    response = chatbot.get_response(emotion)  # You need to implement this function
    return JsonResponse({'response': str(response)})
