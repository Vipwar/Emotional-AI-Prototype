import librosa
import numpy as np
import joblib

def extract_features(audio_file):
    audio, sample_rate = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Load model
model = joblib.load('models/voice_model.pkl')

# Dự đoán cảm xúc từ file test
test_file = 'test_voice.wav'
features = extract_features(test_file)
emotion = model.predict([features])

print("🎙️ Voice Emotion:", emotion[0])
