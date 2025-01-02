import librosa
import numpy as np
from tensorflow.keras.models import load_model

def detect_emotion_from_audio_file(audio_file_path):
    audio_data, _ = librosa.load(audio_file_path, sr=None)

    mfccs = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=13)
    features = np.mean(mfccs, axis=1)

    model = load_model('trained_model.h5')

    emotion = model.predict(features.reshape(1, -1))

    return emotion
