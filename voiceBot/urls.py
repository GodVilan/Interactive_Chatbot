from django.contrib import admin
from django.urls import path
from MyVoiceBot.views import home, get_response, upload_voice_message

urlpatterns = [
    path('', home, name='home'),
    path('get_response/', get_response, name='get_response'),
    path('upload_voice_message/', upload_voice_message, name='upload_voice_message'),  # Add this line
]
