import os
import logging
from contextlib import redirect_stdout

# 1. MENYEMBUNYIKAN WARNING & LOG INTERNAL TENSORFLOW
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('absl').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from streamlit_autorefresh import st_autorefresh

# ==================================
# CONFIG STREAMLIT
# ==================================
st.set_page_config(
    page_title="Dashboard Tanaman Bayam CNN-1D",
    layout="wide"
)

# ==================================
# LOAD DATA & PREPROCESSING (CACHED)
# ==================================
@st.cache_data
def load_and_preprocess_all_data():
    df_raw = pd.read_excel("Data_Bayam_1440 2026.xlsx", header=1)
    df_raw.columns = ["NO", "Hari", "Tanggal", "Waktu", "Kelembapan", "Suhu", "Status_Tanah"]
    
    df_raw["Kelembapan"] = pd.to_numeric(df_raw["Kelembapan"], errors="coerce")
    df_raw["Suhu"] = pd.to_numeric(df_raw["Suhu"], errors="coerce")
    df_raw = df_raw.dropna().reset_index(drop=True)
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_raw[["Kelembapan", "Suhu"]].values)
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(df_raw["Status_Tanah"].values)
    
    window = 6
    X, y = [], []
    for i in range(len(X_scaled) - window):
        X.append(X_scaled[i:i+window])
        y.append(y_encoded[i+window])
        
    return df_raw, np.array(X), np.array(y), scaler, encoder

df, X_all, y_all, scaler, encoder = load_and_preprocess_all_data()

# ==================================
# TRAINING MODEL FUNCTION (CACHED)
# ==================================
@st.cache_resource
def train_cnn_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    model = Sequential([
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(6,2)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=2, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=0)
    
    model.save("cnn_bayam_brazil.h5")
    return model

model = train_cnn_model(X_all, y_all)

# ==================================
# NAVIGASI DASHBOARD (SIDEBAR)
# ==================================
st.sidebar.title("🌱 Navigasi Sistem")
st.sidebar.info("Dashboard ini menampilkan visualisasi kondisi tanah secara berkala (realtime).")

# ==================================
# HALAMAN: MONITORING REALTIME
# ==================================
st.title("🌿 Sistem Monitoring Tanaman Bayam Realtime kelompok 1 Hydrotech")

# Auto-refresh setiap 1 detik
counter = st_autorefresh(interval=1000, key="realtime_counter")

window = 6
index_data = (counter % len(df)) + window

# Mengambil akumulasi data yang berjalan
data_tampil = df.iloc[:index_data]
data_terakhir = data_tampil.iloc[-1]

# Batasi data yang masuk ke grafik (hanya 50 data terakhir agar grafik bergerak maju/scroll)
data_grafik = data_tampil.tail(50)

# Jalankan Prediksi Realtime
sample_raw = data_tampil[["Kelembapan", "Suhu"]].iloc[-window:]
sample_scaled = scaler.transform(sample_raw.values)
sample_input = sample_scaled.reshape(1, window, 2)

# Mengalihkan output ke devnull agar log prediksi tidak membanjiri terminal
with open(os.devnull, 'w') as f, redirect_stdout(f):
    hasil_pred = model.predict(sample_input, verbose=0)

kelas_pred = np.argmax(hasil_pred)
prediksi_status = encoder.classes_[kelas_pred]

# Tampilan Indikator Utama
c1, c2, c3, c4 = st.columns(4)
c1.metric("Kelembapan Sensor", f"{data_terakhir['Kelembapan']}%")
c2.metric("Suhu Lingkungan", f"{data_terakhir['Suhu']}°C")
c3.metric("Status Aktual Alat", data_terakhir["Status_Tanah"])
c4.metric("Prediksi Pintar CNN", prediksi_status)

st.write("")
if prediksi_status == "Basah":
    st.success("🌊 **Kondisi Tanah Tanaman saat ini: BASAH**")
elif prediksi_status == "Normal":
    st.info("🌱 **Kondisi Tanah Tanaman saat ini: NORMAL**")
elif prediksi_status == "Kering":
    st.error("☀️ **Kondisi Tanah Tanaman saat ini: KERING (Butuh Penyiraman!)**")
st.write("")

# Visualisasi Grafik Realtime Berjalan (Menggunakan data_grafik)
g1, g2 = st.columns(2)
with g1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data_grafik["Waktu"].astype(str) + " (" + data_grafik["NO"].astype(str) + ")", 
                              y=data_grafik["Kelembapan"], mode="lines+markers", name="Kelembapan", line=dict(color='#1f77b4', width=2)))
    fig1.update_layout(title="Tren Realtime Kelembapan Tanah (%) - 50 Data Terakhir", xaxis_title="Waktu (No Data)", yaxis_title="Kelembapan (%)")
    st.plotly_chart(fig1, use_container_width=True)
    
with g2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data_grafik["Waktu"].astype(str) + " (" + data_grafik["NO"].astype(str) + ")", 
                              y=data_grafik["Suhu"], mode="lines+markers", name="Suhu", line=dict(color='#ff7f0e', width=2)))
    fig2.update_layout(title="Tren Realtime Suhu Udara (°C) - 50 Data Terakhir", xaxis_title="Waktu (No Data)", yaxis_title="Suhu (°C)")
    st.plotly_chart(fig2, use_container_width=True)
    
st.subheader("📋 Log Riwayat 20 Data Sensor Terakhir")
st.dataframe(data_tampil.tail(20).sort_values(by="NO", ascending=False), use_container_width=True)
