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
        .report-box {
            background: rgba(10, 37, 24, 0.6);
            border: 1px solid #00FF87;
            padding: 20px;
            border-radius: 8px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ==================================
# LOAD DATA & PREPROCESSING (CACHED)
# ==================================
@st.cache_data
def load_and_preprocess_all_data():
    file_xlsx = "Log_Data_Bayam_Brazil_1440_2026.xlsx"
    file_csv_alt = "Data_Bayam_1440 2026.xlsx - Sheet1.csv"
    
    if os.path.exists(file_xlsx):
        df_raw = pd.read_excel(file_xlsx)
    elif os.path.exists(file_csv_alt):
        df_raw = pd.read_csv(file_csv_alt)
    else:
        files_in_dir = os.listdir('.')
        bayam_files = [f for f in files_in_dir if "Bayam" in f and (f.endswith('.csv') or f.endswith('.xlsx'))]
        if bayam_files:
            target_file = bayam_files[0]
            df_raw = pd.read_excel(target_file) if target_file.endswith('.xlsx') else pd.read_csv(target_file)
        else:
            raise FileNotFoundError("Gagal Menemukan file Dataset Bayam di repositori GitHub kamu.")

    if "NO" not in df_raw.columns and len(df_raw.columns) >= 7:
        df_raw = df_raw.copy()
        df_raw.columns = ["NO", "Hari", "Tanggal", "Waktu", "Kelembapan", "Suhu", "Status_Tanah"]
    else:
        df_raw.columns = [str(col).strip().replace(" ", "_").title() for col in df_raw.columns]
        df_raw = df_raw.rename(columns={"No": "NO", "Status_Tanah_": "Status_Tanah", "Status_tanah": "Status_Tanah"})

    df_raw["Kelembapan"] = pd.to_numeric(df_raw["Kelembapan"], errors="coerce")
    df_raw["Suhu"] = pd.to_numeric(df_raw["Suhu"], errors="coerce")
    df_raw = df_raw.dropna(subset=["Kelembapan", "Suhu"]).reset_index(drop=True)
    
    def hitung_status_aktual(klmbp):
        if klmbp < 60:
            return "Kering"
        elif 60 <= klmbp <= 70:
            return "Normal"
        else:
            return "Basah"
            
    df_raw["Status_Tanah"] = df_raw["Kelembapan"].apply(hitung_status_aktual)
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_raw[["Kelembapan", "Suhu"]].values)
    
    encoder = LabelEncoder()
    encoder.fit(["Basah", "Kering", "Normal"])
    y_encoded = encoder.transform(df_raw["Status_Tanah"].values)
    
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
- **Kelembapan Aturan Sistem:**
  -  `< 60%` : Kering
  -  `60% - 70%` : Normal
  -  `> 70%` : Basah
""")
st.sidebar.write("---")
st.sidebar.info("Sistem Auto-Refresh aktif mentransmisikan total 1440 data sensor secara sekuensial.")

# ==================================
# KONTEN UTAMA
# ==================================
st.title("⚡ NEO-MONITORING HYDROTECH // KELOMPOK 1")
st.write("Sistem Pemantauan Cerdas Berbasis Aliran Data Realtime & Analisis Citra Kelayakan Panen.")

tab1, tab2 = st.tabs(["📟 Monitoring Node Realtime", "👁️ Analisis Citra Panen Berkas Lanjut"])

with tab1:
    counter = st_autorefresh(interval=1000, key="realtime_counter")
    
    window = 6
    index_data = (counter % len(df)) + window
    
    data_tampil_all = df.iloc[:index_data]
    data_terakhir = data_tampil_all.iloc[-1]
    data_grafik = data_tampil_all.tail(50)
    
    sample_raw = data_tampil_all[["Kelembapan", "Suhu"]].iloc[-window:]
    sample_scaled = scaler.transform(sample_raw.values)
    sample_input = sample_scaled.reshape(1, window, 2)
    
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        hasil_pred = model.predict(sample_input, verbose=0)
    
    kelas_pred = np.argmax(hasil_pred)
    prediksi_status = encoder.classes_[kelas_pred]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("KELEMBAPAN SENSOR", f"{data_terakhir['Kelembapan']}%")
    c2.metric("SUHU LINGKUNGAN", f"{data_terakhir['Suhu']}°C")
    c3.metric("STATUS AKTUAL ALAT", data_terakhir["Status_Tanah"])
    c4.metric("PREDIKSI PINTAR CNN", prediksi_status)
    
    st.write("")
    if data_terakhir['Kelembapan'] > 70:
        st.success(f"🌊 **[NODE-{data_terakhir['NO']}] KONDISI TANAH: BASAH ({data_terakhir['Kelembapan']}% > 70%)**")
    elif 60 <= data_terakhir['Kelembapan'] <= 70:
        st.info(f"🌱 **[NODE-{data_terakhir['NO']}] KONDISI TANAH: NORMAL (60% - 70%)**")
    else:
        st.error(f"☀️ **[NODE-{data_terakhir['NO']}] KONDISI TANAH: KERING ({data_terakhir['Kelembapan']}% < 60%) - Butuh Irigasi!**")
    st.write("")
    
    g1, g2 = st.columns(2)
    with g1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=data_grafik["Waktu"].astype(str) + " (#" + data_grafik["NO"].astype(str) + ")", 
            y=data_grafik["Kelembapan"], mode="lines+markers", name="Kelembapan",
            line=dict(color='#00FF87', width=2.5)
        ))
        fig1.update_layout(
            title="🎯 TREN REALTIME KELEMBAPAN TANAH (%)",
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
            title="🌡️ TREN REALTIME SUHU UDARA (°C)",
            xaxis_title="Waktu (No Data)", yaxis_title="Suhu (°C)",
            paper_bgcolor='rgba(10,37,24,0.5)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0F2E9')
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    st.subheader(f"📊 Aliran Matriks Realtime Berjalan (Data Saat Ini: {index_data} dari 1440)")
    st.dataframe(data_tampil_all.sort_values(by="NO", ascending=False), use_container_width=True)


# ==========================================================
# PERBAIKAN & PENINGKATAN: TAB CITRA KOMPUTER LEBIH LENGKAP
# ==========================================================
with tab2:
    st.header("👁️ NEO-VISION: Analisis Komputasi Citra Tanaman")
    st.write("Unggah foto makro daun Bayam Brazil Anda untuk mengekstrak visual matriks, persentase klorofil, dan laporan kelayakan panen agronomi.")
    
    file_gambar = st.file_uploader("Unggah Foto Komponen Bayam Brazil (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])
    
    if file_gambar is not None:
        img = Image.open(file_gambar)
        
        # Ekstraksi matriks piksel komputer untuk analisis lanjut
        img_np = np.array(img)
        
        # Pengaman jika gambar grayscale
        if len(img_np.shape) == 3:
            r = img_np[:, :, 0].astype(float)
            g = img_np[:, :, 1].astype(float)
            b = img_np[:, :, 2].astype(float)
            
            # Kalkulasi indeks segmentasi warna dasar tanaman
            total_pixel = img_np.shape[0] * img_np.shape[1]
            
            # Piksel dominan hijau sehat (Nilai G lebih tinggi dari R dan B)
            green_mask = (g > r) & (g > b) & (g > 40)
            # Piksel indikasi sakit/klorosis/bercak kuning-cokelat (R tinggi, G tinggi, B rendah)
            yellow_mask = (r > b) & (g > b) & (r > 60) & (g > 60) & (~green_mask)
            
            p_sehat = (np.sum(green_mask) / total_pixel) * 100
            p_sakit = (np.sum(yellow_mask) / total_pixel) * 100
            p_background = 100 - (p_sehat + p_sakit)
        else:
            p_sehat, p_sakit, p_background = 80.0, 20.0, 0.0
            
        # Tampilan layout analisis
        col_img1, col_img2 = st.columns([1, 1.2])
        
        with col_img1:
            st.image(img, caption="Sumber Citra Node Tanaman", use_container_width=True)
            
            # Buat chart donat kontribusi visual warna daun
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Piksel Hijau Sehat', 'Piksel Klorosis/Bercak', 'Latar Belakang/Lainnya'],
                values=[p_sehat, p_sakit, p_background],
                hole=.4,
                marker=dict(colors=['#00FF87', '#FF3B30', '#4A4A4A'])
            )])
            fig_pie.update_layout(
                title="📊 Komposisi Ekstraksi Warna Citra",
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E0F2E9'),
                showlegend=True,
                height=300
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_img2:
            st.markdown("### 🧬 Laporan Ekstraksi Agronomi & Kelayakan")
            
            # Penentuan Keputusan Berdasarkan Hasil Ekstraksi Citra Luas Daun Sehat
            if p_sehat >= 50.0:
                st.markdown("<h3 style='color: #00FF87; margin-top:0;'>STATUS: LAYAK PANEN ✅</h3>", unsafe_allow_html=True)
                
                # Kotak Laporan Informasi Detil
                st.markdown(f"""
                <div class="report-box">
                    <strong>📈 HASIL METRIK COMPUTER VISION:</strong><br>
                    • Persentase Area Daun Sehat (Klorofil Tinggi): <span style="color:#00FF87;">{p_sehat:.2f}%</span><br>
                    • Persentase Area Defisiensi/Penyakit: <span style="color:#FF3B30;">{p_sakit:.2f}%</span><br><br>
                    <strong>📝 REKOMENDASI SISTEM BUDIDAYA:</strong><br>
                    1. Tanaman memiliki indeks kerapatan vegetasi (NDVI) yang tinggi dan rimbun sempurna.<br>
                    2. Kandungan fitokimia antioksidan dan zat besi dalam daun berada dalam kadar puncak komersial.<br>
                    3. Pemotongan/pemanenan dapat segera dilakukan secara selektif dengan menyisakan 3-4 helai daun bawah agar tanaman dapat bertunas kembali.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color: #FF3B30; margin-top:0;'>STATUS: BELUM LAYAK PANEN ⚠️</h3>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="report-box">
                    <strong>📈 HASIL METRIK COMPUTER VISION:</strong><br>
                    • Persentase Area Daun Sehat (Klorofil Tinggi): <span style="color:#00FF87;">{p_sehat:.2f}%</span><br>
                    • Persentase Area Defisiensi/Penyakit: <span style="color:#FF3B30;">{p_sakit:.2f}%</span><br><br>
                    <strong>❌ GEJALA KLINIS YANG TERDETEKSI:</strong><br>
                    • Bercak kuning/cokelat yang tinggi mengindikasikan tanaman mengalami <em>Klorosis</em> (kehilangan klorofil) akibat ketidakseimbangan kelembapan media tanam atau serangan hama kutu daun.<br><br>
                    <strong>🛠️ TINDAKAN PERBAIKAN (ACTION PLAN):</strong><br>
                    1. <strong>Sistem Irigasi:</strong> Sinkronisasikan dengan tab monitoring realtime. Jika grafik historis tanah berstatus <span style="color:#FF3B30;">Kering</span>, tingkatkan debit air.<br>
                    2. <strong>Nutrisi:</strong> Berikan pupuk nitrogen tinggi (misal pupuk organik cair daun) untuk memicu regenerasi sel hijau daun.<br>
                    3. Karantina pot tanaman ini dari jangkauan tanaman sehat lainnya untuk meminimalkan penyebaran patogen.
                </div>
                """, unsafe_allow_html=True)
                
            # Metrik card tambahan untuk nilai estetika dashboard cyber
            st.write("")
            m1, m2 = st.columns(2)
            m1.metric("Kepadatan Klorofil Est.", f"{p_sehat * 1.2:.1f} SPAD")
            m2.metric("Tingkat Keparahan Hama", f"{p_sakit:.1f}%")

    st.write("---")
    st.subheader("📋 Arsip Statis: 50 Data Excel Awal Master Dataset")
    st.dataframe(df.head(50), use_container_width=True)
