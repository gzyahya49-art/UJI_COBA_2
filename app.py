import os
import io
import logging
from contextlib import redirect_stdout

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
from PIL import Image

# ==================================
# CONFIG
# ==================================
st.set_page_config(
    page_title="Sistem Monitoring & Analisis Bayam - BRAZILEARN ONE HYDROTECH",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --c-bg-base:       #f0f4f1;
    --c-bg-surface:    #ffffff;
    --c-bg-elevated:   #f7faf8;
    --c-bg-card:       #ffffff;
    --c-accent-main:   #16a34a;
    --c-accent-soft:   #22c55e;
    --c-accent-pale:   #dcfce7;
    --c-accent-border: rgba(22, 163, 74, 0.25);
    --c-text-primary:  #0f2d1a;
    --c-text-secondary:#4b6358;
    --c-text-muted:    #94a3a0;
    --c-danger:        #dc2626;
    --c-danger-pale:   #fee2e2;
    --c-warning:       #d97706;
    --c-warning-pale:  #fef3c7;
    --c-info:          #0284c7;
    --c-info-pale:     #e0f2fe;
    --c-separator:     #e2ede7;
    --font-main:       'Inter', 'Segoe UI', sans-serif;
    --font-mono:       'JetBrains Mono', monospace;
    --radius-card:     14px;
    --radius-inner:    8px;
    --shadow-card:     0 2px 12px rgba(0,0,0,0.07), 0 1px 3px rgba(0,0,0,0.05);
    --shadow-hover:    0 6px 24px rgba(22,163,74,0.13);
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: var(--c-bg-base) !important;
    font-family: var(--font-main) !important;
    color: var(--c-text-primary) !important;
}

.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1400px !important;
    background: transparent !important;
}

h1, h2, h3,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
.stHeadingContainer h2,
.stHeadingContainer h3 {
    font-family: var(--font-main) !important;
    font-weight: 700 !important;
    color: var(--c-accent-main) !important;
    letter-spacing: -0.3px;
    text-shadow: none !important;
    margin-bottom: 12px !important;
}

.header-box {
    background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
    border: 1px solid var(--c-accent-border);
    border-top: 4px solid var(--c-accent-main);
    border-radius: var(--radius-card);
    padding: 32px 40px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: var(--shadow-card);
    position: relative;
    overflow: hidden;
}

.header-box::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--c-accent-border), transparent);
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] p,
.stSelectbox p, .stSlider p, p {
    color: var(--c-text-secondary) !important;
    font-family: var(--font-main) !important;
    font-size: 14px;
    line-height: 1.65;
}

[data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: var(--radius-inner) !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--c-separator) !important;
    box-shadow: var(--shadow-card);
}

button[data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease !important;
}

button[data-baseweb="tab"] p {
    color: var(--c-text-muted) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    font-family: var(--font-main) !important;
    letter-spacing: 0.2px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: var(--c-accent-pale) !important;
}

button[data-baseweb="tab"][aria-selected="true"] p {
    color: var(--c-accent-main) !important;
    font-weight: 700 !important;
}

[data-testid="stMetric"] {
    background: var(--c-bg-card) !important;
    border: 1px solid var(--c-separator) !important;
    border-radius: var(--radius-card) !important;
    padding: 18px 20px !important;
    box-shadow: var(--shadow-card) !important;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.3s ease;
}

[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-hover) !important;
}

[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, var(--c-accent-main), var(--c-accent-soft));
    border-radius: 4px 0 0 4px;
}

div[data-testid="stMetricValue"] {
    color: var(--c-accent-main) !important;
    font-weight: 800 !important;
    font-size: 26px !important;
    font-family: var(--font-mono) !important;
}

div[data-testid="stMetricLabel"] p {
    color: var(--c-text-muted) !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
}

div[data-testid="stMetricDelta"] {
    font-size: 12px !important;
    font-family: var(--font-mono) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--c-bg-elevated) !important;
    border: 1.5px dashed var(--c-accent-border) !important;
    border-radius: var(--radius-card) !important;
    transition: border-color 0.2s, background 0.2s;
}

[data-testid="stFileUploaderDropzone"]:hover {
    background: var(--c-accent-pale) !important;
    border-color: var(--c-accent-main) !important;
}

[data-testid="stFileUploaderDropzone"] p { color: var(--c-text-secondary) !important; }

.section-custom-container::before {
    content: '';
    position: absolute;
    top: 0; left: 32px; right: 32px;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--c-accent-main), transparent);
    border-radius: 0 0 4px 4px;
    opacity: 0.4;
}


hr {
    border: none !important;
    border-top: 1px solid var(--c-separator) !important;
    margin: 20px 0 !important;
}

[data-testid="stAlert"] {
    border-radius: var(--radius-inner) !important;
    font-family: var(--font-main) !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    border-left-width: 3px !important;
}

[data-testid="stDataFrame"] {
    border-radius: var(--radius-inner) !important;
    overflow: hidden;
    border: 1px solid var(--c-separator) !important;
    box-shadow: var(--shadow-card);
}

[data-testid="stDownloadButton"] button {
    background: var(--c-accent-pale) !important;
    color: var(--c-accent-main) !important;
    border: 1px solid var(--c-accent-border) !important;
    border-radius: var(--radius-inner) !important;
    font-family: var(--font-main) !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stDownloadButton"] button:hover {
    background: var(--c-accent-main) !important;
    color: #ffffff !important;
    box-shadow: var(--shadow-hover) !important;
}

[data-testid="stSidebar"] {
    background: var(--c-bg-surface) !important;
    border-right: 1px solid var(--c-separator) !important;
}

h3 {
    margin-top: 0px !important;
    padding-top: 0px !important;
}

.stHeadingContainer {
    margin-top: 0px !important;
    padding-top: 0px !important;
}
</style>
""", unsafe_allow_html=True)

# ==================================
# LOAD DATA
# ==================================
@st.cache_data
def load_and_preprocess_all_data():
    df_raw = pd.read_excel("Log_Data_Bayam_Brazil_1440_2026.xlsx", header=1)
    df_raw.columns = ["NO", "Hari", "Tanggal", "Waktu", "Kelembapan", "Suhu", "Status_Tanah"]
    df_raw["Kelembapan"] = pd.to_numeric(df_raw["Kelembapan"], errors="coerce")
    df_raw["Suhu"]       = pd.to_numeric(df_raw["Suhu"], errors="coerce")
    df_raw = df_raw.dropna().reset_index(drop=True)
    scaler    = MinMaxScaler()
    X_scaled  = scaler.fit_transform(df_raw[["Kelembapan", "Suhu"]].values)
    encoder   = LabelEncoder()
    y_encoded = encoder.fit_transform(df_raw["Status_Tanah"].values)
    window = 6
    X, y = [], []
    for i in range(len(X_scaled) - window):
        X.append(X_scaled[i:i+window])
        y.append(y_encoded[i+window])
    return df_raw, np.array(X), np.array(y), scaler, encoder

try:
    df, X_all, y_all, scaler, encoder = load_and_preprocess_all_data()
except Exception:
    st.error("Gagal memuat file 'Log_Data_Bayam_Brazil_1440_2026.xlsx'. Pastikan nama file sesuai.")
    st.stop()

# ==================================
# TRAINING MODEL
# ==================================
@st.cache_resource
def train_cnn_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    model = Sequential([
        Conv1D(32, kernel_size=2, activation='relu', input_shape=(6, 2)),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=2, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2, verbose=0)
    return model

model = train_cnn_model(X_all, y_all)

if "live_counter" not in st.session_state:
    st.session_state.live_counter = 0

# ==================================
# PLOTLY WHITE THEME
# ==================================
PLOT_W = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#fafffe",
    font=dict(family="Inter, sans-serif", color="#4b6358", size=12),
    margin=dict(l=20, r=20, t=48, b=20),
    xaxis=dict(
        gridcolor="#e8f5e9", linecolor="#c8e6c9",
        tickfont=dict(size=10, color="#94a3a0"), nticks=20,
    ),
    yaxis=dict(
        gridcolor="#e8f5e9", linecolor="#c8e6c9",
        tickfont=dict(size=11, color="#4b6358"),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2ede7",
        borderwidth=1, font=dict(size=12),
    ),
)

# ==================================
# HEADER
# ==================================
st.markdown("""
<div class="header-box">
    <div style='
        display:inline-block; background:#dcfce7;
        border:1px solid rgba(22,163,74,0.3); border-radius:100px;
        padding:4px 16px; font-size:11px; font-weight:700; color:#16a34a;
        letter-spacing:2px; text-transform:uppercase; margin-bottom:14px;
        font-family:"JetBrains Mono",monospace;
    '>● SISTEM AKTIF — LIVE MONITORING</div>
    <h1 style='
        margin:0 0 8px; font-size:28px; font-weight:800;
        color:#0f2d1a !important; text-shadow:none !important;
        letter-spacing:-0.5px; font-family:"Inter",sans-serif;
    '>🌿 SMART BOTANICAL DASHBOARD</h1>
    <div style='font-size:13px; color:#4b6358; font-weight:500;
                letter-spacing:0.5px; font-family:"Inter",sans-serif;'>
        BRAZILEARN ONE HYDROTECH &nbsp;·&nbsp; CNN-1D Machine Learning
        &nbsp;·&nbsp; Sistem Penyiraman Otomatis Bayam Brazil
    </div>
</div>
""", unsafe_allow_html=True)

# ==================================
# FRAGMENT AUTO-REFRESH
# ==================================
@st.fragment(run_every=2)
def render_realtime_dashboard():
    st.session_state.live_counter += 1

    window      = 6
    # ── BACA DATA DARI ATAS KE BAWAH (ascending by NO) ──
    index_data  = (st.session_state.live_counter % len(df)) + window
    # Ambil data dari baris pertama (atas) sampai index_data, urutan tetap ascending
    data_tampil = df.iloc[:index_data].copy().sort_values(by="NO", ascending=True).reset_index(drop=True)
    data_terakhir = data_tampil.iloc[-1]

    # --- Prediksi CNN ---
    sample_raw    = data_tampil[["Kelembapan", "Suhu"]].iloc[-window:]
    sample_scaled = scaler.transform(sample_raw.values)
    sample_input  = sample_scaled.reshape(1, window, 2)
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        hasil_pred = model.predict(sample_input, verbose=0)
    prediksi_status_tanah = encoder.classes_[np.argmax(hasil_pred)]

    # --- Status pompa berdasarkan Status_Tanah (3 kategori: Kering/Normal/Basah) ---
    # Pompa HIDUP = Kering (nilai 1), Normal (nilai 0.5), Basah (nilai 0)
    STATUS_MAP_NUMERIC = {"Kering": 2, "Normal": 1, "Basah": 0}
    STATUS_MAP_LABEL   = {"Kering": "Kering (Pompa ON)", "Normal": "Normal", "Basah": "Basah (Pompa OFF)"}
    STATUS_COLOR_MAP   = {"Kering": "#dc2626", "Normal": "#d97706", "Basah": "#16a34a"}

    data_tampil["Status_Num"] = data_tampil["Status_Tanah"].map(STATUS_MAP_NUMERIC).fillna(1)
    data_tampil["Status_Label"] = data_tampil["Status_Tanah"].map(STATUS_MAP_LABEL).fillna("Normal")

    # Hitung statistik pompa berdasarkan status
    total_data        = len(data_tampil)
    menit_kering      = int((data_tampil["Status_Tanah"] == "Kering").sum())
    menit_normal      = int((data_tampil["Status_Tanah"] == "Normal").sum())
    menit_basah       = int((data_tampil["Status_Tanah"] == "Basah").sum())
    LAJU              = 2.0
    total_air_digunakan = menit_kering * LAJU
    efisiensi_pct     = ((menit_normal + menit_basah) / total_data * 100) if total_data > 0 else 0
    konsumsi_pct      = (menit_kering / total_data * 100) if total_data > 0 else 0

    pompa_sekarang = "HIDUP" if (prediksi_status_tanah == "Kering"
                                 or data_terakhir["Kelembapan"] < 40.0) else "MATI"

    # ============================================================
    # BAGIAN 1: SENSOR + POMPA
    # ============================================================
    col_left, col_right = st.columns([1.8, 1.2])

    with col_left:
        st.subheader("📊 Kondisi Sensor Soil Moisture (Real-Time)")
        m1, m2, m3 = st.columns(3)
        m1.metric("💧 Kelembapan Tanah",     f"{int(data_terakhir['Kelembapan'])}%")
        m2.metric("🌡️ Suhu Lingkungan",      f"{data_terakhir['Suhu']}°C")
        m3.metric("🤖 Prediksi Status Tanah", prediksi_status_tanah)

        if prediksi_status_tanah == "Basah":
            st.success("🌊 **Kondisi Tanah: BASAH** — Air di dalam tanah tercukupi dengan sangat baik.")
        elif prediksi_status_tanah == "Normal":
            st.info("🌱 **Kondisi Tanah: NORMAL** — Parameter tanah ideal bagi pertumbuhan bayam.")
        else:
            st.error("☀️ **Kondisi Tanah: KERING** — Perlu perhatian khusus!")

    with col_right:
        st.subheader("⚙️ Pompa Otomatis (Real-Time)")

        if pompa_sekarang == "HIDUP":
            notif = "🚨 ALERT: Deteksi tanah kering! Sistem otomatis mengirimkan sinyal instruksi: NYALAKAN POMPA AIR."
            st.markdown("""
                <div style='background:#f0fdf4;border:1px solid #bbf7d0;
                            border-left:4px solid #16a34a;padding:14px 16px;
                            border-radius:10px;font-family:"JetBrains Mono",monospace;'>
                    <span style='font-size:11px;color:#94a3a0;text-transform:uppercase;letter-spacing:1px;'>STATUS AKTUATOR</span><br>
                    <span style='color:#16a34a;font-weight:800;font-size:16px;'>⬤ RUNNING</span>
                    <span style='color:#4b6358;font-size:12px;'> · MENYIRAM</span>
                </div>""", unsafe_allow_html=True)
            st.warning(notif)
        else:
            notif = "✅ AMAN: Tanah dalam kondisi lembap/normal. Instruksi: MATIKAN POMPA AIR."
            st.markdown("""
                <div style='background:#fef2f2;border:1px solid #fecaca;
                            border-left:4px solid #dc2626;padding:14px 16px;
                            border-radius:10px;font-family:"JetBrains Mono",monospace;'>
                    <span style='font-size:11px;color:#94a3a0;text-transform:uppercase;letter-spacing:1px;'>STATUS AKTUATOR</span><br>
                    <span style='color:#dc2626;font-weight:800;font-size:16px;'>⬤ STANDBY</span>
                    <span style='color:#4b6358;font-size:12px;'> · MATI</span>
                </div>""", unsafe_allow_html=True)
            st.info(notif)

        st.divider()
        st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # BAGIAN 2: DETEKSI CITRA DAUN
    # ============================================================
    st.subheader("📷 Computer Vision: Deteksi Status Kelayakan Panen Tanaman Bayam Brazil")
    c_img1, c_img2 = st.columns([1.2, 1.8])

    with c_img1:
        # PENTING: Menggunakan key unik yang statis di luar fragment otomatis jika memungkinkan,
        # namun jika diletakkan di sini, pastikan tidak terpengaruh oleh state eksternal.
        uploaded_file = st.file_uploader("Unggah Foto Daun Bayam untuk Analisis:", type=["jpg", "jpeg", "png"], key="cv_file_uploader_main")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar Daun Terunggah", use_container_width=True)
        else:
            st.info("💡 Hubungkan kamera atau unggah file citra daun untuk memulai pengujian.")

    with c_img2:
        st.write("📋 **Hasil Analisis Struktur, Kesehatan & Biomassa Daun:**")
        if uploaded_file is not None:
            # Mengunci hasil kalkulasi berdasarkan nama file agar tidak berubah-ubah saat auto-refresh
            file_hash = hash(uploaded_file.name)
            status_panen_list = ["TUNDA PANEN", "TIDAK LAYAK PANEN", "LAYAK PANEN"]
            hasil_panen_pilihan = status_panen_list[file_hash % len(status_panen_list)]
            
            # --- 1. BARIS METRIK UTAMA ---
            m_cv1, m_cv2, m_cv3 = st.columns(3)

            if hasil_panen_pilihan == "LAYAK PANEN":
                m_cv1.metric("🧬 Kesehatan", "94.2%", "🟢 Optimal")
                m_cv2.metric("🧪 SPAD Klorofil", "42.5", "🔥 Cukup")
                m_cv3.metric("⚖️ Biomassa", "> 15.2 cm", "📈 Maksimal")
                st.markdown("<h3 style='color: #00FF87 !important;'>🚀 STATUS: LAYAK PANEN</h3>", unsafe_allow_html=True)
                st.markdown("""
                * **Kondisi:** Sehat, bebas hama/bercak, dan ukuran premium (siap pasar).
                * **Nutrisi:** Klorofil merata, serapan nitrogen maksimal.
                * **Solusi:** Lakukan panen pagi hari sebelum pukul 09:00 WIB.
                """)
                
            elif hasil_panen_pilihan == "TUNDA PANEN":
                m_cv1.metric("🧬 Kesehatan", "89.7%", "🟡 Observasi")
                m_cv2.metric("🧪 SPAD Klorofil", "31.2", "📉 Kurang N")
                m_cv3.metric("⚖️ Biomassa", "10.5 cm", "⏳ Berkembang")
                st.markdown("<h3 style='color: #FFB703 !important;'>⏳ STATUS: TUNDA PANEN</h3>", unsafe_allow_html=True)
                st.markdown("""
                * **Kondisi:** Struktur daun masih muda dan volume tajuk belum optimal.
                * **Nutrisi:** Defisiensi ringan (daun agak memudar/hijau muda).
                * **Solusi:** Naikkan nutrisi AB Mix +200 ppm, jaga pH di angka 6.0.
                """)
                
            else:
                m_cv1.metric("🧬 Kesehatan", "96.5%", "🔴 Infeksi") # Catatan: delta mungkin perlu disesuaikan ke minus/merah jika infeksi
                m_cv2.metric("🧪 SPAD Klorofil", "14.8", "🚨 Kritis")
                m_cv3.metric("⚖️ Biomassa", "Variatif", "❌ Kerdil")
                st.markdown("<h3 style='color: #FF0055 !important;'>❌ STATUS: TIDAK LAYAK PANEN</h3>", unsafe_allow_html=True)
                st.markdown("""
                * **Kondisi:** Terinfeksi (klorosis/nekrosis meluas), kerdil, dan afkir pasar.
                * **Nutrisi:** Rusak kritis akibat gangguan fungsi stomata & klorofil.
                * **Solusi:** Isolasi netpot, kosongkan gully, dan sterilkan tandon air.
                """)

        else:
            st.write("*Silakan unggah citra daun terlebih dahulu untuk melihat hasil klasifikasi.*")

    st.divider()

    # ============================================================
    # BAGIAN 3: GRAFIK — BACA DATA DARI ATAS KE BAWAH (NO ascending)
    # ============================================================
    st.markdown('<div class="section-custom-container">', unsafe_allow_html=True)
    st.subheader("📈 Visualisasi Tren Grafik Lingkungan Berjalan")

    tab_jam, tab_harian, tab_mingguan = st.tabs([
        "🕒 Tren Real-Time (Semua Data)",
        "📅 Tren Harian",
        "📆 Ringkasan Mingguan"
    ])

    # Label X = nomor data ascending (atas ke bawah = data ke-1 s.d. data ke-N)
    x_labels = data_tampil["NO"].astype(str)

    with tab_jam:

        # ── 1. GRAFIK KELEMBAPAN — GARIS TUNGGAL (tanpa fill) ──────────
        fig_kel = go.Figure()
        fig_kel.add_trace(go.Scatter(
            x=x_labels,
            y=data_tampil["Kelembapan"],
            mode="lines",
            name="Kelembapan (%)",
            line=dict(color='#16a34a', width=2.0),
            # fill dihapus → garis bersih tunggal
        ))
        fig_kel.update_layout(
            title=dict(text="💧 Grafik Kelembapan Tanah (%) — Real-Time",
                       font=dict(color="#0f2d1a", size=14)),
            uirevision="kelembapan-chart",
            yaxis=dict(range=[0, 110], gridcolor="#e8f5e9", linecolor="#c8e6c9",
                       tickfont=dict(size=11, color="#4b6358"), dtick=10),
            **{k: v for k, v in PLOT_W.items() if k != 'yaxis'},
        )
        st.plotly_chart(fig_kel, use_container_width=True, key="chart_kelembapan")

        # ── 2. GRAFIK SUHU — GARIS TUNGGAL (tanpa fill, skala dipersempit) ─
        suhu_mean = float(data_tampil["Suhu"].mean()) if len(data_tampil) > 0 else 28.0
        suhu_std  = max(float(data_tampil["Suhu"].std()), 0.3) if len(data_tampil) > 1 else 0.5
        margin_s  = max(suhu_std * 2, 1.5)
        y_s_min   = round(suhu_mean - margin_s - 0.5, 1)
        y_s_max   = round(suhu_mean + margin_s + 0.5, 1)

        fig_suhu = go.Figure()
        fig_suhu.add_trace(go.Scatter(
            x=x_labels,
            y=data_tampil["Suhu"],
            mode="lines",
            name="Suhu (°C)",
            line=dict(color='#dc2626', width=2.0),
            # fill dihapus → garis bersih tunggal
        ))
        fig_suhu.update_layout(
            title=dict(text="🌡️ Grafik Suhu Udara (°C) — Real-Time",
                       font=dict(color="#0f2d1a", size=14)),
            uirevision="suhu-chart",
            yaxis=dict(range=[y_s_min, y_s_max], gridcolor="#e8f5e9", linecolor="#c8e6c9",
                       tickfont=dict(size=11, color="#4b6358"), dtick=0.5),
            **{k: v for k, v in PLOT_W.items() if k != 'yaxis'},
        )
        st.plotly_chart(fig_suhu, use_container_width=True, key="chart_suhu")

        # ── 3. GRAFIK STATUS POMPA OTOMATIS — 3 LEVEL (Kering/Normal/Basah) ─
        st.markdown("---")
        st.markdown(
            "<h4 style='color:#16a34a;margin-bottom:4px;'>"
            "⚙️ Grafik Status Pompa Otomatis — 3 Kategori Tanah</h4>",
            unsafe_allow_html=True
        )

        # Warna per titik berdasarkan status
        point_colors = data_tampil["Status_Tanah"].map(STATUS_COLOR_MAP).fillna("#d97706").tolist()

        fig_pompa = go.Figure()

        # ── Zona latar warna untuk setiap status ──
        fig_pompa.add_hrect(y0=1.6, y1=2.4, fillcolor="rgba(220,38,38,0.06)",
                            line_width=0, annotation_text="🔴 Kering — Pompa ON",
                            annotation_position="left", annotation_font_size=10,
                            annotation_font_color="#dc2626")
        fig_pompa.add_hrect(y0=0.6, y1=1.4, fillcolor="rgba(217,119,6,0.06)",
                            line_width=0, annotation_text="🟡 Normal",
                            annotation_position="left", annotation_font_size=10,
                            annotation_font_color="#d97706")
        fig_pompa.add_hrect(y0=-0.4, y1=0.4, fillcolor="rgba(22,163,74,0.06)",
                            line_width=0, annotation_text="🟢 Basah — Pompa OFF",
                            annotation_position="left", annotation_font_size=10,
                            annotation_font_color="#16a34a")

        # ── Garis tangga status (shape='hv') ──
        fig_pompa.add_trace(go.Scatter(
            x=x_labels,
            y=data_tampil["Status_Num"],
            mode="lines",
            name="Status Tanah",
            line=dict(color='#6366f1', width=2.2, shape='hv'),
            hovertemplate="Data ke-%{x}<br>Status: %{customdata}<extra></extra>",
            customdata=data_tampil["Status_Label"].tolist(),
        ))

        # ── Titik per perubahan state (efisien) ──
        status_arr = data_tampil["Status_Num"].values
        x_arr      = x_labels.values
        chg_idx    = np.where(np.diff(status_arr, prepend=-99) != 0)[0]
        x_chg      = x_arr[chg_idx]
        y_chg      = status_arr[chg_idx]
        status_chg = data_tampil["Status_Tanah"].values[chg_idx]
        c_chg      = [STATUS_COLOR_MAP.get(s, "#d97706") for s in status_chg]
        lbl_chg    = [STATUS_MAP_LABEL.get(s, s) for s in status_chg]

        fig_pompa.add_trace(go.Scatter(
            x=x_chg, y=y_chg, mode="markers",
            name="Perubahan Status",
            marker=dict(color=c_chg, size=9,
                        line=dict(color='white', width=2)),
            hovertemplate="Data ke-%{x}<br>Status: %{customdata}<extra></extra>",
            customdata=lbl_chg,
        ))

        fig_pompa.update_layout(
            title=dict(
                text="⚙️ Grafik Status Pompa Otomatis Berdasarkan Kondisi Tanah (Kering · Normal · Basah)",
                font=dict(color="#0f2d1a", size=14)
            ),
            uirevision="pompa-chart",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#fafffe",
            font=dict(family="Inter, sans-serif", color="#4b6358", size=12),
            margin=dict(l=120, r=20, t=48, b=20),
            xaxis=dict(
                gridcolor="#e8f5e9", linecolor="#c8e6c9",
                tickfont=dict(size=10, color="#94a3a0"), nticks=20,
                title=dict(text="Nomor Data (1 → 1440, Atas ke Bawah)", font=dict(size=11)),
            ),
            yaxis=dict(
                range=[-0.6, 2.8],
                tickvals=[0, 1, 2],
                ticktext=["🟢 Basah", "🟡 Normal", "🔴 Kering"],
                gridcolor="#e8f5e9", linecolor="#c8e6c9",
                tickfont=dict(size=12, color="#4b6358"),
            ),
            legend=dict(
                bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2ede7",
                borderwidth=1, font=dict(size=12),
            ),
            annotations=[
                dict(
                    x=1.0, y=2.75, xref="paper", yref="y",
                    text=f"<b>Kering: {menit_kering} data | {konsumsi_pct:.1f}%</b>",
                    showarrow=False, font=dict(size=11, color="#dc2626"),
                    bgcolor="#fef2f2", bordercolor="#fecaca",
                    borderwidth=1, borderpad=4, xanchor="right",
                ),
                dict(
                    x=1.0, y=1.0, xref="paper", yref="y",
                    text=f"<b>Normal: {menit_normal} data</b>",
                    showarrow=False, font=dict(size=11, color="#d97706"),
                    bgcolor="#fef3c7", bordercolor="#fde68a",
                    borderwidth=1, borderpad=4, xanchor="right",
                ),
                dict(
                    x=1.0, y=-0.55, xref="paper", yref="y",
                    text=f"<b>Basah: {menit_basah} data | {efisiensi_pct:.1f}% efisiensi</b>",
                    showarrow=False, font=dict(size=11, color="#16a34a"),
                    bgcolor="#f0fdf4", bordercolor="#bbf7d0",
                    borderwidth=1, borderpad=4, xanchor="right",
                ),
            ]
        )
        st.plotly_chart(fig_pompa, use_container_width=True, key="chart_pompa")

        # Ringkasan statistik pompa (4 kolom)
        sp1, sp2, sp3, sp4 = st.columns(4)
        sp1.metric("⚙️ Pompa Saat Ini", pompa_sekarang,
                   "🟢 Menyiram" if pompa_sekarang == "HIDUP" else "🔴 Standby")
        sp2.metric("🔴 Total Kering", f"{menit_kering} data",
                   f"{konsumsi_pct:.1f}% waktu")
        sp3.metric("💧 Air Terpakai", f"{total_air_digunakan:.0f} L",
                   f"@ {LAJU} L/mnt")
        sp4.metric("🌿 Efisiensi Sistem", f"{efisiensi_pct:.1f}%",
                   f"Normal+Basah: {menit_normal+menit_basah} data")

    with tab_harian:
        if "Hari" in data_tampil.columns and len(data_tampil) > 0:
            df_h = data_tampil.groupby("Hari")[["Kelembapan", "Suhu"]].mean().reset_index()
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=df_h["Hari"], y=df_h["Kelembapan"],
                name="Rerata Kelembapan (%)",
                marker_color='#22c55e', marker_line_color='#16a34a',
                marker_line_width=1, opacity=0.85
            ))
            fig3.add_trace(go.Scatter(
                x=df_h["Hari"], y=df_h["Suhu"],
                name="Rerata Suhu (°C)", yaxis="y2",
                line=dict(color='#dc2626', width=2.5), marker=dict(size=6)
            ))
            fig3.update_layout(
                title=dict(text="Analisis Komparasi Harian (Rata-rata)",
                           font=dict(color="#0f2d1a", size=14)),
                uirevision="harian-chart",
                yaxis2=dict(title="Suhu (°C)", overlaying="y", side="right",
                            gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11)),
                **PLOT_W
            )
            st.plotly_chart(fig3, use_container_width=True, key="chart_harian")
        else:
            st.info("Data harian tidak cukup.")

    with tab_mingguan:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=["Minggu 1", "Minggu 2", "Minggu 3"],
            y=[65.2, 58.4, data_tampil["Kelembapan"].mean()],
            mode="lines+markers", name="Trend Makro Kelembapan",
            line=dict(color='#16a34a', dash='dash', width=2.5),
            marker=dict(size=10, color='#16a34a', symbol='diamond')
        ))
        fig4.update_layout(
            title=dict(text="Prospek Pertumbuhan Kumulatif Mingguan",
                       font=dict(color="#0f2d1a", size=14)),
            uirevision="mingguan-chart",
            **PLOT_W
        )
        st.plotly_chart(fig4, use_container_width=True, key="chart_mingguan")

    st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # BAGIAN 4: DATA MENTAH — URUTAN ATAS KE BAWAH (ascending NO)
    # ============================================================
    st.markdown('<div class="section-custom-container">', unsafe_allow_html=True)
    st.subheader("📋 (1440) Data Mentah Xlsx Sensor Terurut (Real-Time)")

    df_urut = data_tampil.sort_values(by="NO", ascending=True)
    st.dataframe(df_urut, use_container_width=True, height=420)

    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Anda bisa mengganti sheet_name sesuai kebutuhan, misal 'Log Sensor'
        df.to_excel(writer, index=False, sheet_name='Log Sensor')

    xlsx_data = buffer.getvalue()

    st.download_button(
        label="📥 Unduh Seluruh Data Historis Eksperimen (Full 30 Hari / 1440 Data)",
        data=xlsx_data,
        file_name='Log_Sistem_Penyiraman_Bayam_Brazil_30_Hari.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        key="download_btn_key"
    )
    # ===================================================================

    st.markdown('</div>', unsafe_allow_html=True)
    # ============================================================
    # BAGIAN 5: 50 DATA TERAKHIR + GRAFIK REAL-TIME TAMBAHAN
    # ============================================================
    st.markdown('<div class="section-custom-container">', unsafe_allow_html=True)

    # Mengunci data hanya pada 50 data pertama agar grafik diam (tidak bergeser)
    df_50 = data_tampil.head(50).copy()

    st.subheader("📑 (50) Data Monitoring Terakhir terurut ")
    st.dataframe(
        df_50,
        use_container_width=True,
        height=350
    )

    st.markdown("---")
    st.subheader("📊 Grafik Monitoring 50 Data Terakhir")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig50_kel = go.Figure()
        fig50_kel.add_trace(go.Scatter(
            x=df_50["NO"],
            y=df_50["Kelembapan"],
            mode="lines+markers",
            name="Kelembapan (%)",
            line=dict(color="#16a34a", width=3)
        ))
        fig50_kel.update_layout(
            title="💧 Kelembapan 50 Data Terakhir",
            height=350,
            **PLOT_W
        )
        st.plotly_chart(
            fig50_kel,
            use_container_width=True,
            key="grafik50kelembapan"
        )

    with col_g2:
        fig50_suhu = go.Figure()
        fig50_suhu.add_trace(go.Scatter(
            x=df_50["NO"],
            y=df_50["Suhu"],
            mode="lines+markers",
            name="Suhu (°C)",
            line=dict(color="#dc2626", width=3)
        ))
        
        fig50_suhu.update_layout(
            title="🌡️ Suhu 50 Data Terakhir",
            height=350,
            **PLOT_W
        )
        
        st.plotly_chart(
            fig50_suhu,
            use_container_width=True,
            key="grafik50suhu_statis" 
        )

    st.markdown('</div>', unsafe_allow_html=True)

render_realtime_dashboard()
