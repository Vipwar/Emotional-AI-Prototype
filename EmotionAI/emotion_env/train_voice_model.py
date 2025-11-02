import librosa
import numpy as np
from sklearn.svm import SVC
import joblib

# Hàm trích đặc trưng MFCC
def extract_features(audio_file):
    audio, sample_rate = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Dữ liệu training (bạn thay file của bạn ở đây)
X_train = [
    extract_features("audio/happy.wav"),
    extract_features("audio/sad.wav")
]
y_train = ['happy', 'sad']

# Train SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Lưu model
joblib.dump(model, 'models/voice_model.pkl')
print("✅ Voice model trained and saved.")
