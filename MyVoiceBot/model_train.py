import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score

SAMPLE_RATE = 22050
DURATION = 5
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(directory, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = np.ceil(num_samples_per_segment / hop_length)

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(root.split("/")[-1])

    return np.array(data["mfcc"]), np.array(data["labels"])

def prepare_dataset(data_path):
    X, y = save_mfcc(data_path)
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def build_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(17, activation='softmax')
    ])
    return model

def train_model(model, X_train, X_test, y_train, y_test):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

    hist_df = pd.DataFrame(history.history)
    best_accuracy = np.max(hist_df['accuracy'])

    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)
    TP = sum((y_test == 1) & (predictions == 1))
    FP = sum((y_test == 0) & (predictions == 1))
    FN = sum((y_test == 1) & (predictions == 0))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    best_accuracy_percent = best_accuracy * 100
    precision_percent = precision * 100
    recall_percent = recall * 100
    f1_percent = f1 * 100
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [best_accuracy_percent, precision_percent, recall_percent, f1_percent]
    })

    fig, ax = plt.subplots(1, 1)
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc = 'center', loc='center')

    plt.show()

def main():
    X_train, X_test, y_train, y_test = prepare_dataset("AudioFiles")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)
    train_model(model, X_train, X_test, y_train, y_test)
    model.save('trained_model.h5')

if __name__ == "__main__":
    main()