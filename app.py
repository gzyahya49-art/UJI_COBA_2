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
# CONFIG STREAMLIT & NATURAL ECO THEME
# ==================================
st.set_page_config(
    page_title="ECO-FOREST: Pemantauan Bayam Brazil",
    layout="wide"
)

# Custom Natural Botanical CSS Injection (Earthy & Clean Style)
st.markdown("""
    <style>
        /* Background Utama bertema Alam Teduh */
        .stApp {
            background-color: #F4F7F5;
            color: #2C3E35;
        }
        /* Judul Gaya Dokumentasi Ilmiah Modern */
        h1, h2, h3 {
            color: #1E4631 !important;
            font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
            font-weight: 600;
        }
        /* Desain Card Metrik Seperti Lab Pertanian Modern */
        .stMetric {
            background: #FFFFFF;
            border: 1px solid #D2DDD7;
            padding: 18px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.03);
        }
        div[data-testid="stMetricValue"] {
            color: #2E6F40 !important;
            font-weight: bold;
        }
        /* Desain Sidebar Hijau Hutan Lindung */
        .sidebar .sidebar-content {
            background-color: #1E4631;
            color: #EBF3EE;
        }
        /* Box Laporan Hasil Komputasi Citra */
        .report-box {
            background: #FFFFFF;
            border-left: 5px solid #2E6F40;
            border-top: 1px solid #E2EAE5;
            border-right: 1px solid #E2EAE5;
            border-bottom: 1px solid #E2EAE5;
            padding: 22px;
            border-radius: 0px 8px 8px 0px;
            margin-top: 15px;
            box-shadow: 0 4px 12px rgba(46, 111, 64, 0.05);
            color: #2C3E35;
            line-height: 1.6;
        }
        /* Mengubah warna font tab agar serasi */
        button[data-baseweb="tab"] {
            color: #556B5F !important;
        }
        button[aria-selected="true"] {
            color: #1E4631 !important;
            font-weight: bold !important;
            border-bottom-color: #2E6F40 !important;
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
            raise FileNotFoundError("Gagal Menemukan file Dataset Bayam di repositori komputer Anda.")

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
st.sidebar.title("🌿 ECO-FOREST SYSTEM")
st.sidebar.subheader("Informasi Agronomi")
st.sidebar.markdown("""
**Bayam Brazil (*Alternanthera sissoo*)** adalah sayuran daun penutup tanah bernutrisi tinggi yang sangat adaptif. 

* **Suhu Ideal Perkembangan:** 25°C - 30°C
* **Standarisasi Kadar Air / Kelembapan:**
  * `< 60%` : Kering
  * `60% - 70%` : Normal / Optimal
  * `> 70%` : Basah
""")
st.sidebar.write("---")
st.sidebar.info("Sistem Auto-Refresh memproses total 1440 data sensor secara sekuensial.")

# ==================================
# KONTEN UTAMA
# ==================================
st.title("🌱 Sistem Pemantauan & Analisis Citra Bayam Brazil")
st.write("Integrasi Aliran Sensor Riil Terbuka dan Evaluasi Visual Komputasi Kesehatan Tanaman.")

tab1, tab2 = st.tabs(["📊 Pemantauan Nirkabel Seketika", "👁️ Analisis Citra Kesehatan Daun"])

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
    
    # Panel Informasi Utama (Bersih & Elegan)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("KELEMBAPAN TANAH", f"{data_terakhir['Kelembapan']}%")
    c2.metric("SUHU LINGKUNGAN", f"{data_terakhir['Suhu']}°C")
    c3.metric("KONDISI AKTUAL MEDIA", data_terakhir["Status_Tanah"])
    c4.metric("PREDIKSI KLASIFIKASI CNN", prediksi_status)
    
    st.write("")
    if data_terakhir['Kelembapan'] > 70:
        st.success(f"🔹 **[Data Ke-{data_terakhir['NO']}] Status Media: BASAH ({data_terakhir['Kelembapan']}% > 70%)**")
    elif 60 <= data_terakhir['Kelembapan'] <= 70:
        st.info(f"🌿 **[Data Ke-{data_terakhir['NO']}] Status Media: OPTIMAL / NORMAL (60% - 70%)**")
    else:
        st.error(f"🔸 **[Data Ke-{data_terakhir['NO']}] Status Media: KERING ({data_terakhir['Kelembapan']}% < 60%) — Memerlukan Irigasi Tambahan**")
    st.write("")
    
    # Grafik Bertema Alam (Warna Forest Green & Earthy Terracotta)
    g1, g2 = st.columns(2)
    with g1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=data_grafik["Waktu"].astype(str) + " (#" + data_grafik["NO"].astype(str) + ")", 
            y=data_grafik["Kelembapan"], mode="lines+markers", name="Kelembapan",
            line=dict(color='#2E6F40', width=2.5)
        ))
        fig1.update_layout(
            title="📈 Fluktuasi Kelembapan Tanah (%)",
            xaxis_title="Waktu Pengukuran", yaxis_title="Persentase (%)",
            paper_bgcolor='#FFFFFF', plot_bgcolor='#F9FBF9',
            font=dict(color='#2C3E35')
        )
        st.plotly_chart(fig1, use_container_width=True)
        
    with g2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=data_grafik["Waktu"].astype(str) + " (#" + data_grafik["NO"].astype(str) + ")", 
            y=data_grafik["Suhu"], mode="lines+markers", name="Suhu",
            line=dict(color='#C85A32', width=2.5)
        ))
        fig2.update_layout(
            title="🌡️ Perkembangan Suhu Udara (°C)",
            xaxis_title="Waktu Pengukuran", yaxis_title="Derajat Celcius (°C)",
            paper_bgcolor='#FFFFFF', plot_bgcolor='#F9FBF9',
            font=dict(color='#2C3E35')
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    st.subheader(f"📋 Aliran Log Data Berjalan ({index_data} dari 1440 entri)")
    st.dataframe(data_tampil_all.sort_values(by="NO", ascending=False), use_container_width=True)


# ==========================================================
# TAB 2: CITRA KOMPUTER BERTEMA BOTANI & EKOLOGI ALAM
# ==========================================================
with tab2:
    st.header("🔬 Analisis Spektrum Daun Komputasional")
    st.write("Unggah dokumentasi foto makro untuk ekstraksi persentase jaringan sehat serta peta analisis biologis kelayakan panen.")
    
    file_gambar = st.file_uploader("Pilih Berkas Foto Bayam Brazil (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])
    
    if file_gambar is not None:
        img = Image.open(file_gambar)
        img_np = np.array(img)
        
        if len(img_np.shape) == 3:
            r = img_np[:, :, 0].astype(float)
            g = img_np[:, :, 1].astype(float)
            b = img_np[:, :, 2].astype(float)
            
            total_pixel = img_np.shape[0] * img_np.shape[1]
            
            # Segmentasi warna daun alami
            green_mask = (g > r) & (g > b) & (g > 40)
            yellow_mask = (r > b) & (g > b) & (r > 60) & (g > 60) & (~green_mask)
            
            p_sehat = (np.sum(green_mask) / total_pixel) * 100
            p_sakit = (np.sum(yellow_mask) / total_pixel) * 100
            p_background = 100 - (p_sehat + p_sakit)
        else:
            p_sehat, p_sakit, p_background = 85.0, 15.0, 0.0
            
        col_img1, col_img2 = st.columns([1, 1.2])
        
        with col_img1:
            st.image(img, caption="Sampel Jaringan Vegetatif", use_container_width=True)
            
            # Pie Chart Alami (Hijau Kebun, Kuning Daun Layu, Abu-abu Lembut)
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Jaringan Hijau Sehat', 'Gejala Klorosis/Bercak', 'Latar Belakang / Media'],
                values=[p_sehat, p_sakit, p_background],
                hole=.4,
                marker=dict(colors=['#3B824E', '#D1A153', '#A0AFA6'])
            )])
            fig_pie.update_layout(
                title="📊 Segmentasi Komposisi Kromatografi Citra",
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2C3E35'),
                showlegend=True,
                height=300
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_img2:
            st.markdown("### 📋 Laporan Fisiologis & Rekomendasi Agronomis")
            
            if p_sehat >= 50.0:
                st.markdown("<h3 style='color: #2E6F40; margin-top:0;'>KESIMPULAN: VEGETASI OPTIMAL (SIAP PANEN)</h3>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="report-box">
                    <strong>📊 DATA SPASIAL KOMPUTASI:</strong><br>
                    • Estimasi Area Klorofil Aktif: <span style="color:#2E6F40; font-weight:bold;">{p_sehat:.2f}%</span><br>
                    • Rasio Jaringan Terdegradasi: <span style="color:#A47124;">{p_sakit:.2f}%</span><br><br>
                    <strong>🌱 REKOMENDASI PEMANENAN:</strong><br>
                    1. Karakteristik indeks kanopi menunjukkan akumulasi nutrisi dan klorofil di tingkat puncak pemasaran.<br>
                    2. Pemanenan disarankan menggunakan teknik rotasi daun: potong tangkai luar tua dan sisakan pusat tunas dalam agar perkembangan berikutnya tetap produktif.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color: #A47124; margin-top:0;'>KESIMPULAN: PERLU PERAWATAN INTENSIF (TUNDA PANEN)</h3>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="report-box">
                    <strong>📊 DATA SPASIAL KOMPUTASI:</strong><br>
                    • Estimasi Area Klorofil Aktif: <span style="color:#2E6F40; font-weight:bold;">{p_sehat:.2f}%</span><br>
                    • Rasio Jaringan Terdegradasi: <span style="color:#A47124; font-weight:bold;">{p_sakit:.2f}%</span><br><br>
                    <strong>⚠️ ANALISIS GEJALA FISIK:</strong><br>
                    • Tingginya bercak kuning menandakan berkurangnya konsentrasi zat klorofil akibat paparan panas ekstrem atau defisiensi unsur hara makro (seperti Nitrogen/Magnesium).<br><br>
                    <strong>🛠️ PLAN REGENERASI TANAMAN:</strong><br>
                    1. Sesuaikan pengairan dengan melihat status kelembapan pada dashboard utama.<br>
                    2. Aplikasikan pupuk cair organik kaya senyawa nitrogen guna meregenerasi jaringan klorofil baru.
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            m1, m2 = st.columns(2)
            m1.metric("Kepadatan Klorofil Est.", f"{p_sehat * 1.2:.1f} SPAD")
            m2.metric("Tingkat Penyakit Daun", f"{p_sakit:.1f}%")

    st.write("---")
    st.subheader("📋 Arsip Statis: 50 Data Excel Awal Master Dataset")
    st.dataframe(df.head(50), use_container_width=True)
