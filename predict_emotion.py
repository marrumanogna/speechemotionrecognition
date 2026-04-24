import numpy as np
import librosa
import tensorflow as tf
from keras.models import load_model

class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.model = load_model("emotion_model.h5")  # Load your trained model
        self.class_labels = ["HAPPY", "SAD", "ANGRY", "SURPRISED", "NEUTRAL", "FEAR", "DISGUST"]

    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return np.expand_dims(mfccs_scaled, axis=0)  

    def predict_emotion(self, file_path):
        features = self.extract_features(file_path)
        prediction = self.model.predict(features)
        predicted_index = np.argmax(prediction)
        return self.class_labels[predicted_index] 