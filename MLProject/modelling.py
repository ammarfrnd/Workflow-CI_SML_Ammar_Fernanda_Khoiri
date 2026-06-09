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
import mlflow.keras
import mlflow.tensorflow

# 1. KONFIGURASI MLFLOW (Paling atas)
mlflow.set_tracking_uri("sqlite:///mlflow_data.db")
("Eksperimen_Hate_Speech_Ammar_Fernanda_Khoiri")

# Autologging mencatat metrik training otomatis
mlflow.tensorflow.autolog()

# Memulai satu blok run utama
(run_name="Bi-LSTM_Tuned_Ammar"):
    print("=== 1. Membaca Data ===")
    df = pd.read_csv("namadataset_preprocessing.csv")
    df['Tweet_Clean'] = df['Tweet_Clean'].fillna('').astype(str)
    X, y = df['Tweet_Clean'].values, df['HS'].values 

    print("=== 2. Splitting Data ===")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("=== 3. Tokenisasi & Padding ===")
    MAX_WORDS, MAX_LEN = 10000, 50
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)
    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN, padding='post', truncating='post')

    print("=== 4. Membangun Arsitektur Model ===")
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=64, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("=== 5. Training Model ===")
    model.fit(
        X_train_pad, y_train, epochs=20, batch_size=32,
        validation_data=(X_test_pad, y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],
        verbose=1
    )
    
    print("=== 6. Menyimpan Artefak Model ===")
    # Log model ke MLflow (Ini akan menghasilkan MLmodel, conda.yml, dll secara otomatis)
    mlflow.keras.log_model(model, "model_bi_lstm")
    
    print("\n=== 7. Evaluasi & Log Parameter ===")
    mlflow.log_param("max_words", MAX_WORDS)
    mlflow.log_param("max_len", MAX_LEN)
    model.evaluate(X_test_pad, y_test, verbose=1)
    
    print("Selesai! Buka dashboard dengan: mlflow ui --backend-store-uri sqlite:///mlflow_data.db")