from django.db import models

class VoiceMessage(models.Model):
    audio_file = models.FileField(upload_to='voice_messages/')
    emotion = models.CharField(max_length=200, blank=True)
