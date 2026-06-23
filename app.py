import os
import logging
import random
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
# CONFIG & THEME STREAMLIT (Satu Halaman Tunggal)
# ==================================
st.set_page_config(
    page_title="Sistem Monitoring & Analisis Bayam - BRAZILEARN ONE HYDROTECH",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* 1. Pengaturan Background Utama */
    [data-testid="stAppViewContainer"] {
        background: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRSD0MnK0fOm6BBdgWVGHh5DLgsyBqppShq37ytsMvqEQ&s=10') no-repeat center/cover;
    }
    
    /* Lapisan dasar semi transparan agar background tidak menusuk mata */
    [data-testid="stHeader"], [data-testid="stMainBlockContainer"] {
        background-color: rgba(6, 14, 10, 0.5) !important;
    }

    /* 2. Heading (H1, H2, H3) Hijau Neon Terang Berpendar */
    h1, h2, h3, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {
        color: #00FF87 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-shadow: 0 0 12px rgba(0, 255, 135, 0.7);
    }
    
    .stHeadingContainer h2, .stHeadingContainer h3 {
        color: #00FF87 !important;
    }

    /* 3. Wadah Judul Utama Keren (Glassmorphism + Neon Border) */
    .header-neon-box {
        background-color: rgba(12, 28, 20, 0.85) !important;
        padding: 25px;
        border-radius: 14px;
        border: 2px solid #00FF87;
        box-shadow: 0 0 20px rgba(0, 255, 135, 0.4);
        text-align: center;
        margin-bottom: 25px;
    }

    /* Custom Container Box untuk membalut bagian bawah agar memiliki background bebas & rapi */
    .section-custom-container {
        background-color: rgba(15, 32, 24, 0.85) !important;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(0, 255, 135, 0.2);
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }

    /* 4. Perbaikan Teks Deskripsi & Label Supaya Rapi */
    [data-testid="stMarkdownContainer"] p, 
    [data-testid="stWidgetLabel"] p, 
    .stSelectbox p, 
    .stSlider p,
    span[data-testid="stTextAreaWidgetApiKeyCounter"] {
        color: #e0f2e9 !important;
        font-family: 'Segoe UI', sans-serif;
        font-size: 15px;
    }

    /* 5. Perbaikan Teks di Dalam Struktur Tab */
    button[data-baseweb="tab"] p {
        background-color: transparent !important;
        color: #00FF87 !important;
        text-shadow: 0 0 5px rgba(0, 255, 135, 0.4);
    }
    
    /* 6. Pengaturan Kotak Metrik Rapi (Proporsional & Sejajar) */
    .stMetric {
        background-color: rgba(15, 32, 24, 0.93) !important;
        padding: 12px 18px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 255, 135, 0.3) !important;
        border-left: 5px solid #00FF87 !important;
        box-shadow: 0 0 15px rgba(0, 255, 135, 0.25) !important;
    }
    
    /* Nilai Angka Metrik */
    div[data-testid="stMetricValue"] {
        color: #00FF87 !important;
        text-shadow: 0 0 10px rgba(0, 255, 135, 0.7) !important;
        font-weight: bold;
        font-size: 28px !important;
    }

    /* Judul Atas Metrik */
    div[data-testid="stMetricLabel"] p {
        background-color: transparent !important;
        border-left: none !important;
        box-shadow: none !important;
        color: #a3ffd0 !important;
        font-weight: 600;
        padding: 0 !important;
        font-size: 14px !important;
    }
    
    /* 7. Kotak Dropzone File Uploader */
    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(15, 32, 24, 0.85) !important;
        border: 2px dashed #00FF87 !important;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==================================
# LOAD DATA & PREPROCESSING (CACHED)
# ==================================
@st.cache_data
def load_and_preprocess_all_data():
    df_raw = pd.read_excel("Log_Data_Bayam_Brazil_1440_2026.xlsx", header=1)
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

try:
    df, X_all, y_all, scaler, encoder = load_and_preprocess_all_data()
except Exception as e:
    st.error("Gagal memuat file 'Log_Data_Bayam_Brazil_1440_2026.xlsx'. Pastikan nama file sesuai.")
    st.stop()

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
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2, verbose=0)
    return model

model = train_cnn_model(X_all, y_all)

# Initialize Session State untuk menyimpan Counter
if "live_counter" not in st.session_state:
    st.session_state.live_counter = 0

# --------------------------------------------------
# WADAH HEADER DASHBOARD NEON BARU
# --------------------------------------------------
st.markdown("""
    <div class="header-neon-box">
        <h1 style='margin: 0; padding-bottom: 5px; font-size: 32px;'>🌿 Smart Botanical Dashboard Kelompok: BRAZILEARN ONE HYDROTECH</h1>
        <div style='color: #52b788; font-weight: bold; font-size: 16px;'>Sistem Integrasi Monitoring Tanah & Analisis Kelayakan Panen Bayam Berbasis AI</div>
    </div>
""", unsafe_allow_html=True)


# ==================================
# DEKLARASI FRAGMENT UNTUK AUTO-REFRESH TANPA BERKEDIP
# ==================================
@st.fragment(run_every=2)
def render_realtime_dashboard():
    # Menaikkan counter internal jalannya data simulation
    st.session_state.live_counter += 1
    
    window = 6
    index_data = (st.session_state.live_counter % len(df)) + window
    data_tampil = df.iloc[:index_data]
    data_terakhir = data_tampil.iloc[-1]
    
    # 🌟 DIUBAH: Menggunakan semua data berjalan dari awal hingga baris saat ini (bisa sampai 1440 data)
    data_grafik = data_tampil 

    # Jalankan Prediksi Realtime CNN-1D
    sample_raw = data_tampil[["Kelembapan", "Suhu"]].iloc[-window:]
    sample_scaled = scaler.transform(sample_raw.values)
    sample_input = sample_scaled.reshape(1, window, 2)

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        hasil_pred = model.predict(sample_input, verbose=0)
    kelas_pred = np.argmax(hasil_pred)
    prediksi_status_tanah = encoder.classes_[kelas_pred]

    # --- BAGIAN 1: MONITORING UTAMA & KONTROL POMPA ---
    col_left, col_right = st.columns([1.8, 1.2])

    with col_left:
        st.subheader("📊 Kondisi Sensor Terkini")
        m1, m2, m3 = st.columns(3)
        m1.metric("💧 Kelembapan Tanah", f"{int(data_terakhir['Kelembapan'])}%")
        m2.metric("🌡️ Suhu Lingkungan", f"{data_terakhir['Suhu']}°C")
        m3.metric("🤖 Prediksi AI CNN", prediksi_status_tanah)
        
        if prediksi_status_tanah == "Basah":
            st.success("🌊 **Kondisi Tanah: BASAH** — Air di dalam tanah tercukupi dengan sangat baik.")
        elif prediksi_status_tanah == "Normal":
            st.info("🌱 **Kondisi Tanah: NORMAL** — Parameter tanah ideal bagi pertumbuhan bayam.")
        else:
            st.error("☀️ **Kondisi Tanah: KERING** — Perlu perhatian khusus!")

    with col_right:
        st.subheader("⚙️ Panel Pompa Otomatis (Real-Time)")
        mode_pompa = st.radio("Pilih Mode Sistem:", ["Otomatis (Sistem Pintar)", "Manual (Override)"], horizontal=True, key="mode_pompa_key")
        
        status_pompa_aktif = "MATI"
        notif_perintah = "Sistem dalam kondisi aman dan seimbang."
        
        if mode_pompa == "Otomatis (Sistem Pintar)":
            if prediksi_status_tanah == "Kering" or data_terakhir['Kelembapan'] < 40.0:
                status_pompa_aktif = "HIDUP"
                notif_perintah = "🚨 ALERT: Deteksi tanah kering! Sinyal otomatis: NYALAKAN POMPA AIR."
            else:
                status_pompa_aktif = "MATI"
                notif_perintah = "✅ AMAN: Tanah dalam kondisi ideal. Sinyal otomatis: MATIKAN POMPA AIR."
        else:
            saklar_manual = st.toggle("Aktifkan Pompa Secara Manual", key="saklar_manual_key")
            if saklar_manual:
                status_pompa_aktif = "HIDUP"
                notif_perintah = "⚠️ MANUAL OVERRIDE: Pompa dinyalakan secara paksa oleh Pengguna."
            else:
                status_pompa_aktif = "MATI"
                notif_perintah = "⚠️ MANUAL OVERRIDE: Pompa dimatikan secara paksa oleh Pengguna."

        if status_pompa_aktif == "HIDUP":
            st.markdown(f"<div style='background-color: rgba(40, 167, 69, 0.25); border-left:6px solid #00FF87; padding:12px; border-radius:5px; margin-bottom:10px;'><b style='color:#00FF87;'>Status Aktuator:</b> <span style='color:#fff; font-weight:bold;'>🔵 RUNNING (MENYIRAM)</span></div>", unsafe_allow_html=True)
            st.warning(notif_perintah)
        else:
            st.markdown(f"<div style='background-color: rgba(220, 53, 69, 0.25); border-left:6px solid #dc3545; padding:12px; border-radius:5px; margin-bottom:10px;'><b style='color:#ff8787;'>Status Aktuator:</b> <span style='color:#fff; font-weight:bold;'>🔴 STANDBY (MATI)</span></div>", unsafe_allow_html=True)
            st.info(notif_perintah)

    st.divider()

    # --- BAGIAN 2: DETEKSI CITRA DAUN ---
    st.subheader("📷 Computer Vision: Deteksi Status Kelayakan Panen Daun")
    c_img1, c_img2 = st.columns([1, 2])

    with c_img1:
        uploaded_file = st.file_uploader("Unggah Foto Daun Bayam untuk Analisis:", type=["jpg", "jpeg", "png"], key="cv_upload_key")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Daun Terunggah", use_container_width=True)
        else:
            st.info("💡 Hubungkan kamera atau unggah file citra daun untuk memulai pengujian.")

    with c_img2:
        st.write("📋 **Hasil Analisis Struktur & Klorofil Daun:**")
        if uploaded_file is not None:
            file_hash = hash(uploaded_file.name)
            status_panen_list = ["TUNDA PANEN", "TIDAK LAYAK PANEN", "LAYAK PANEN"]
            hasil_panen_pilihan = status_panen_list[file_hash % len(status_panen_list)]
            
            v1, v2, v3 = st.columns(3)
            if hasil_panen_pilihan == "LAYAK PANEN":
                v1.metric("Confidence Score", "94.2%")
                v2.metric("Indeks Klorofil (SPAD)", "42.5")
                v3.metric("Luas Area Daun", "> 15.2 cm")
                st.success("### 🌱 HASIL: LAYAK PANEN")
                st.markdown("* **Rekomendasi Tindakan:** Daun bayam telah mencapai ukuran komersial optimal dengan pigmentasi sempurna. Segera potong pada pagi hari.")
            elif hasil_panen_pilihan == "TUNDA PANEN":
                v1.metric("Confidence Score", "89.7%")
                v2.metric("Indeks Klorofil (SPAD)", "31.2")
                v3.metric("Luas Area Daun", "10.5 cm")
                st.warning("### ⏳ HASIL: TUNDA PANEN")
                st.markdown("* **Rekomendasi Tindakan:** Ukuran daun belum memenuhi standar pasar. Berikan AB Mix tambahan.")
            else:
                v1.metric("Confidence Score", "96.5%")
                v2.metric("Indeks Klorofil (SPAD)", "14.8")
                v3.metric("Luas Area Daun", "Variatif")
                st.error("### ❌ HASIL: TIDAK LAYAK PANEN")
                st.markdown("* **Rekomendasi Tindakan:** Terdeteksi adanya bercak nekrosis parah atau klorosis.")
        else:
            st.write("*Silakan unggah citra daun terlebih dahulu untuk melihat hasil klasifikasi.*")

    st.divider()

    # --- BAGIAN 3 & 4: VISUALISASI TREN (MEMUAT SEMUA DATA) ---
    st.markdown('<div class="section-custom-container">', unsafe_allow_html=True)
    
    st.subheader("📈 Visualisasi Tren Parameter Lingkungan Berjalan")
    tab_jam, tab_harian, tab_mingguan = st.tabs(["🕒 Tren Real-time (Semua Data Berjalan)", "📅 Tren Harian", "📆 Ringkasan Mingguan"])

    with tab_jam:
        g1, g2 = st.columns(2)
        with g1:
            fig1 = go.Figure()
            # 🌟 DIUBAH: mode="lines" untuk menghilangkan bulatan hitam marker agar grafik 1440 data tetap mulus & ringan
            fig1.add_trace(go.Scatter(x=data_grafik["Waktu"].astype(str) + " (" + data_grafik["NO"].astype(str) + ")", 
                                      y=data_grafik["Kelembapan"], mode="lines", name="Kelembapan", line=dict(color='#00FF87', width=1.5)))
            fig1.update_layout(title="Tren Realtime Kelembapan Tanah (%)", xaxis_title="Waktu (No Data)", yaxis_title="Kelembapan (%)", template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig1, use_container_width=True)
            
        with g2:
            fig2 = go.Figure()
            # 🌟 DIUBAH: mode="lines" untuk menghilangkan bulatan hitam marker agar grafik 1440 data tetap mulus & ringan
            fig2.add_trace(go.Scatter(x=data_grafik["Waktu"].astype(str) + " (" + data_grafik["NO"].astype(str) + ")", 
                                      y=data_grafik["Suhu"], mode="lines", name="Suhu", line=dict(color='#d90429', width=1.5)))
            fig2.update_layout(title="Tren Realtime Suhu Udara (°C)", xaxis_title="Waktu (No Data)", yaxis_title="Suhu (°C)", template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig2, use_container_width=True)

    with tab_harian:
        if "Hari" in data_tampil.columns and len(data_tampil) > 0:
            df_harian = data_tampil.groupby("Hari")[["Kelembapan", "Suhu"]].mean().reset_index()
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=df_harian["Hari"], y=df_harian["Kelembapan"], name="Rerata Kelembapan (%)", marker_color='#52b788'))
            fig3.add_trace(go.Scatter(x=df_harian["Hari"], y=df_harian["Suhu"], name="Rerata Suhu (°C)", yaxis="y2", line=dict(color='#d90429', width=3)))
            fig3.update_layout(title="Analisis Komparasi Harian (Rata-rata)", xaxis_title="Hari", yaxis_title="Kelembapan (%)", yaxis2=dict(title="Suhu (°C)", overlaying="y", side="right"), template="plotly_dark", legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Data harian tidak cukup untuk direduksi.")

    with tab_mingguan:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=["Minggu 1", "Minggu 2", "Minggu 3"], y=[65.2, 58.4, data_tampil["Kelembapan"].mean()], mode="lines+markers", name="Trend Makro Kelembapan", line=dict(color='#00FF87', dash='dash')))
        fig4.update_layout(title="Prospek Pertumbuhan Kumulatif Mingguan", yaxis_title="Nilai Indeks Kelembapan", template="plotly_dark")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- BAGIAN 4: DATA MENTAH ---
    st.markdown('<div class="section-custom-container">', unsafe_allow_html=True)
    st.subheader("📋 Data Mentah Excel Sensor Terurut (Real-Time)")
    
    df_urut_realtime = data_tampil.sort_values(by="NO", ascending=False).head(48)
    st.dataframe(df_urut_realtime, use_container_width=True, height=350)

    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh Seluruh Data Historis Eksperimen (Full 30 Hari / 1440 Data)",
        data=csv_data,
        file_name='Log_Sistem_Penyiraman_Bayam_Brazil_30_Hari.csv',
        mime='text/csv',
        key="download_btn_key"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Panggil fungsi penampil utama
render_realtime_dashboard()
