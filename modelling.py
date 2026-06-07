import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.tensorflow

# Inisialisasi MLflow Tracking di lokal laptop Anda
mlflow.set_experiment("Eksperimen_Hate_Speech_Ammar_Fernanda_Khoiri")
mlflow.tensorflow.autolog() # Otomatis mencatat parameter dan metric training

with mlflow.start_run(run_name="Bi-LSTM_Tuned_Ammar"):
    
    print("=== 1. Membaca Data Preprocessing ===")
    # Pastikan file CSV ini sudah dipindahkan ke folder 'Membangun_model'
    df = pd.read_csv("namadataset_preprocessing.csv")
    df['Tweet_Clean'] = df['Tweet_Clean'].fillna('').astype(str)
    
    X = df['Tweet_Clean'].values
    y = df['HS'].values 

    print("=== 2. Splitting Data ===")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("=== 3. Tokenisasi & Padding ===")
    MAX_WORDS = 10000
    MAX_LEN = 50

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)

    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

    print("=== 4. Membangun Arsitektur Model Bi-LSTM (Anti-Overfitting) ===")
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
        # Ditambahkan dropout internal pada layer LSTM untuk membatasi hafalan model
        Bidirectional(LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)),
        Dense(32, activation='relu'),
        Dropout(0.5), # Mencegah ketergantungan antar neuron
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("=== 5. Menyiapkan Callback Early Stopping ===")
    # Menghentikan training secara otomatis jika val_loss memburuk selama 2 epoch berturut-turut
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True # Mengembalikan bobot terbaik ke model saat training berhenti
    )

    print("=== 6. Memulai Proses Training Model ===")
    model.fit(
        X_train_pad, y_train,
        epochs=20, # Batas maksimal tetap 20, tapi akan berhenti otomatis di epoch awal
        batch_size=32,
        validation_data=(X_test_pad, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    print("\n=== 7. Evaluasi Akhir ===")
    loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=1)
    print(f"Loss Model    : {loss:.4f}")
    print(f"Akurasi Model : {accuracy:.4f}")
    
    # Mencatat parameter tokenisasi secara manual ke MLflow dashboard
    mlflow.log_param("max_words", MAX_WORDS)
    mlflow.log_param("max_len", MAX_LEN)