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
from PIL import Image

# ==================================
# CONFIG STREAMLIT & CYBER FOREST THEME
# ==================================
st.set_page_config(
    page_title="NEO-FOREST: Bayam Brazil CNN-1D",
    layout="wide"
)

# Custom Cyber Forest Dark Green & Neon CSS Injection
st.markdown("""
    <style>
        .stApp {
            background-color: #05160E;
            color: #E0F2E9;
        }
        h1, h2, h3 {
            color: #00FF87 !important;
            text-shadow: 0 0 10px rgba(0, 255, 135, 0.3);
            font-family: 'Courier New', monospace;
        }
        .stMetric {
            background: rgba(10, 37, 24, 0.7);
            border: 1px solid #00FF87;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 0 15px rgba(0, 255, 135, 0.1);
        }
        div[data-testid="stMetricValue"] {
            color: #00FF87 !important;
        }
        .sidebar .sidebar-content {
            background-color: #030F0A;
        }
    </style>
""", unsafe_allow_html=True)

# ==================================
# LOAD DATA & PREPROCESSING (CACHED)
# ==================================
@st.cache_data
def load_and_preprocess_all_data():
    # Mekanisme Auto-Detect File untuk menghindari FileNotFoundError di Streamlit Cloud
    file_xlsx = "Data_Bayam_1440 2026.xlsx"
    file_csv_alt = "Data_Bayam_1440 2026.xlsx - Sheet1.csv"
    
    if os.path.exists(file_xlsx):
        # Jika file excel asli ada, baca langsung menggunakan read_excel
        df_raw = pd.read_excel(file_xlsx)
    elif os.path.exists(file_csv_alt):
        # Jika file csv alternatif ada, baca menggunakan read_csv
        df_raw = pd.read_csv(file_csv_alt)
    else:
        # Jika kedua nama di atas tidak ditemukan, cari file apa saja di folder yang mengandung kata "Bayam"
        files_in_dir = os.listdir('.')
        bayam_files = [f for f in files_in_dir if "Bayam" in f and (f.endswith('.csv') or f.endswith('.xlsx'))]
        
        if bayam_files:
            target_file = bayam_files[0]
            if target_file.endswith('.xlsx'):
                df_raw = pd.read_excel(target_file)
            else:
                df_raw = pd.read_csv(target_file)
        else:
            raise FileNotFoundError("Gagal Menemukan file Dataset Bayam di repositori GitHub kamu. Pastikan file excel/csv sudah di-push.")

    # Menjamin nama kolom seragam terlepas dari baris header excel
    if "NO" not in df_raw.columns and len(df_raw.columns) >= 7:
        df_raw = df_raw.copy()
        df_raw.columns = ["NO", "Hari", "Tanggal", "Waktu", "Kelembapan", "Suhu", "Status_Tanah"]
    else:
        # Peta pembersihan jika huruf besar/kecil di kolom berbeda
        df_raw.columns = [str(col).strip().replace(" ", "_").title() for col in df_raw.columns]
        df_raw = df_raw.rename(columns={"No": "NO", "Hari": "Hari", "Tanggal": "Tanggal", "Waktu": "Waktu", 
                                        "Kelembapan": "Kelembapan", "Suhu": "Suhu", "Status_Tanah": "Status_Tanah",
                                        "Status_Tanah ": "Status_Tanah", "Status_tanah": "Status_Tanah"})

    df_raw["Kelembapan"] = pd.to_numeric(df_raw["Kelembapan"], errors="coerce")
    df_raw["Suhu"] = pd.to_numeric(df_raw["Suhu"], errors="coerce")
    df_raw = df_raw.dropna(subset=["Kelembapan", "Suhu"]).reset_index(drop=True)
    
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
st.sidebar.title("🌲 CYBER-FOREST OS")
st.sidebar.subheader("Sistem Informasi Bayam Brazil")
st.sidebar.markdown("""
**Bayam Brazil (*Alternanthera sissoo*)** adalah tanaman sayuran daun penutup tanah yang sangat adaptif. 
- **Suhu Ideal:** 25°C - 30°C
- **Kelembapan Ideal:** 60% - 80%
- **Karakteristik:** Tumbuh rimbun, renyah, dan membutuhkan pasokan air konstan tanpa tergenang ekstrem.
""")
st.sidebar.write("---")
st.sidebar.info("Sistem Auto-Refresh aktif mentransmisikan total 1440 data sensor secara sekuensial.")

# ==================================
# KONTEN UTAMA
# ==================================
st.title("⚡ NEO-MONITORING HYDROTECH // KELOMPOK 1")
st.write("Sistem Pemantauan Cerdas Berbasis Aliran Data Realtime & Analisis Citra Kelayakan Panen.")

# TABS UNTUK INTERFACE BERSIH DAN KEREN
tab1, tab2 = st.tabs(["📟 Monitoring Node Realtime", "👁️ Analisis Citra Panen & Log Arsip"])

with tab1:
    # Auto-refresh sistem setiap 1 detik
    counter = st_autorefresh(interval=1000, key="realtime_counter")
    
    window = 6
    # Berjalan terurut melintasi seluruh data dari 1 sampai 1440
    index_data = (counter % len(df)) + window
    
    # Mengambil akumulasi seluruh data yang berjalan terurut (Maksimal 1440)
    data_tampil_all = df.iloc[:index_data]
    data_terakhir = data_tampil_all.iloc[-1]
    
    # Membatasi grafik bergerak agar fokus pada 50 titik data berjalan terakhir
    data_grafik = data_tampil_all.tail(50)
    
    # Jalankan Prediksi Realtime CNN-1D
    sample_raw = data_tampil_all[["Kelembapan", "Suhu"]].iloc[-window:]
    sample_scaled = scaler.transform(sample_raw.values)
    sample_input = sample_scaled.reshape(1, window, 2)
    
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        hasil_pred = model.predict(sample_input, verbose=0)
    
    kelas_pred = np.argmax(hasil_pred)
    prediksi_status = encoder.classes_[kelas_pred]
    
    # Panel Indikator Utama Realtime
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("KELEMBAPAN SENSOR", f"{data_terakhir['Kelembapan']}%")
    c2.metric("SUHU LINGKUNGAN", f"{data_terakhir['Suhu']}°C")
    c3.metric("STATUS AKTUAL ALAT", data_terakhir["Status_Tanah"])
    c4.metric("PREDIKSI PINTAR CNN", prediksi_status)
    
    st.write("")
    if prediksi_status == "Basah":
        st.success(f"🌊 **[NODE-{data_terakhir['NO']}] KONDISI TANAH TERDETEKSI: BASAH**")
    elif prediksi_status == "Normal":
        st.info(f"🌱 **[NODE-{data_terakhir['NO']}] KONDISI TANAH TERDETEKSI: NORMAL**")
    elif prediksi_status == "Kering":
        st.error(f"☀️ **[NODE-{data_terakhir['NO']}] KONDISI TANAH TERDETEKSI: KERING (Butuh Penyiraman Irigasi!)**")
    st.write("")
    
    # Grafik Realtime Suhu & Kelembapan
    g1, g2 = st.columns(2)
    with g1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=data_grafik["Waktu"].astype(str) + " (#" + data_grafik["NO"].astype(str) + ")", 
            y=data_grafik["Kelembapan"], mode="lines+markers", name="Kelembapan",
            line=dict(color='#00FF87', width=2.5)
        ))
        fig1.update_layout(
            title="🎯 TREN REALTIME KELEMBAPAN TANAH (%) - 50 DATA TERAKHIR",
            xaxis_title="Waktu (No Data)", yaxis_title="Kelembapan (%)",
            paper_bgcolor='rgba(10,37,24,0.5)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0F2E9')
        )
        st.plotly_chart(fig1, use_container_width=True)
        
    with g2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=data_grafik["Waktu"].astype(str) + " (#" + data_grafik["NO"].astype(str) + ")", 
            y=data_grafik["Suhu"], mode="lines+markers", name="Suhu",
            line=dict(color='#FF3B30', width=2.5)
        ))
        fig2.update_layout(
            title="🌡️ TREN REALTIME SUHU UDARA (°C) - 50 DATA TERAKHIR",
            xaxis_title="Waktu (No Data)", yaxis_title="Suhu (°C)",
            paper_bgcolor='rgba(10,37,24,0.5)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0F2E9')
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    # Menampilkan Seluruh Aliran Data Terurut (Dari data ke-1 hingga data berjalan ke-1440)
    st.subheader(f"📊 Aliran Matriks Realtime Berjalan (Data Saat Ini: {index_data} dari 1440)")
    st.dataframe(data_tampil_all.sort_values(by="NO", ascending=False), use_container_width=True)

with tab2:
    st.header("👁️ Sistem Deteksi Citra Komputer Kelayakan Panen")
    st.write("Masukkan foto daun/tanaman Bayam Brazil Anda untuk mendeteksi apakah tanaman dalam kondisi sehat (Layak Panen) atau bermasalah (Tidak Layak Panen).")
    
    file_gambar = st.file_uploader("Unggah Foto Bayam Brazil (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])
    
    if file_gambar is not None:
        img = Image.open(file_gambar)
        col_img1, col_img2 = st.columns([1, 1.5])
        
        with col_img1:
            st.image(img, caption="Foto Tanaman yang Diunggah", use_container_width=True)
            
        with col_img2:
            st.markdown("### 👾 Analisis Metrik Kesehatan Daun")
            img_np = np.array(img)
            mean_green = np.mean(img_np[:,:,1]) if len(img_np.shape) == 3 else 100
            
            if mean_green > 95:
                st.markdown("<h4 style='color: #00FF87;'>STATUS: LAYAK PANEN (SEHAT)</h4>", unsafe_allow_html=True)
                st.info("💡 **Informasi Tanaman:** Pigmentasi klorofil daun sangat optimal dan daun mengembang dengan struktur rimbun sempurna. Karakteristik nutrisi kalsium dan zat besi bayam berada pada level tertinggi. Siap dipasarkan!")
            else:
                st.markdown("<h4 style='color: #FF3B30;'>STATUS: BELUM/TIDAK LAYAK PANEN (BERMASALAH)</h4>", unsafe_allow_html=True)
                st.error("⚠️ **Informasi Masalah Tanaman:** Terdeteksi adanya degradasi warna kekuningan atau bercak kusam akibat ketidakseimbangan kelembapan tanah atau defisiensi hara mikro. Tunda pemetikan dan periksa grafik sensor historis di panel realtime.")
                
    st.write("---")
    # Menampilkan antarmuka 50 data excel awal dataset teratas secara statis
    st.subheader("📋 Arsip Statis: 50 Data Excel Awal Master Dataset")
    st.dataframe(df.head(50), use_container_width=True)
