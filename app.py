import os
import logging
from contextlib import redirect_stdout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('absl').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from streamlit_autorefresh import st_autorefresh
from PIL import Image
import time

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NEO-FOREST | Bayam Brazil CNN-1D",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL CYBER FOREST CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

:root {
    --bg-deep:    #020D07;
    --bg-card:    #071A0E;
    --bg-panel:   #0A2416;
    --green-neon: #00FF87;
    --green-mid:  #00C96B;
    --green-dim:  #1A5C3A;
    --red-alert:  #FF3B30;
    --yellow-warn:#FFD60A;
    --blue-info:  #0AF0FF;
    --text-main:  #D6F5E3;
    --text-dim:   #6EA882;
    --border:     rgba(0,255,135,0.25);
}

/* === BASE === */
.stApp { background-color: var(--bg-deep); color: var(--text-main); font-family: 'Exo 2', sans-serif; }
section[data-testid="stSidebar"] { background: #030F08 !important; border-right: 1px solid var(--border); }
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card); border-radius: 8px; padding: 4px; border: 1px solid var(--border); gap: 4px; }
.stTabs [data-baseweb="tab"] { color: var(--text-dim) !important; font-family: 'Share Tech Mono', monospace; font-size: 13px; border-radius: 6px; padding: 8px 18px; }
.stTabs [aria-selected="true"] { background: var(--bg-panel) !important; color: var(--green-neon) !important; border: 1px solid var(--green-neon) !important; }

/* === HEADINGS === */
h1 { font-family: 'Share Tech Mono', monospace !important; color: var(--green-neon) !important; font-size: 1.8rem !important; letter-spacing: 2px; text-shadow: 0 0 18px rgba(0,255,135,0.4); margin-bottom: 4px !important; }
h2, h3 { font-family: 'Share Tech Mono', monospace !important; color: var(--green-neon) !important; letter-spacing: 1px; text-shadow: 0 0 10px rgba(0,255,135,0.25); }
p, li, span, label { color: var(--text-main) !important; font-family: 'Exo 2', sans-serif; }

/* === METRIC CARDS === */
div[data-testid="stMetric"] { 
    background: var(--bg-card); 
    border: 1px solid var(--border); 
    border-radius: 10px; 
    padding: 16px 20px !important;
    box-shadow: 0 0 20px rgba(0,255,135,0.06), inset 0 1px 0 rgba(0,255,135,0.08);
}
div[data-testid="stMetricLabel"] > div { color: var(--text-dim) !important; font-size: 11px !important; letter-spacing: 1.5px; text-transform: uppercase; font-family: 'Share Tech Mono', monospace !important; }
div[data-testid="stMetricValue"] > div { color: var(--green-neon) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 1.6rem !important; }
div[data-testid="stMetricDelta"] > div { font-family: 'Share Tech Mono', monospace !important; font-size: 12px !important; }

/* === DATAFRAME === */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px; }
.stDataFrame thead tr th { background: var(--bg-panel) !important; color: var(--green-neon) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 12px; }
.stDataFrame tbody tr td { color: var(--text-main) !important; font-size: 12px; }
.stDataFrame tbody tr:nth-child(even) { background: rgba(0,255,135,0.03) !important; }

/* === ALERTS === */
.stAlert { border-radius: 8px !important; border-left-width: 4px !important; font-family: 'Share Tech Mono', monospace !important; }

/* === BUTTONS === */
.stButton > button { background: transparent; border: 1px solid var(--green-neon); color: var(--green-neon); border-radius: 6px; font-family: 'Share Tech Mono', monospace; letter-spacing: 1px; transition: all 0.2s; }
.stButton > button:hover { background: rgba(0,255,135,0.1); box-shadow: 0 0 15px rgba(0,255,135,0.3); }

/* === CUSTOM COMPONENTS === */
.neo-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; margin: 8px 0; box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(0,255,135,0.06); }
.status-badge-panen { background: rgba(0,255,135,0.12); border: 2px solid var(--green-neon); border-radius: 8px; padding: 14px 20px; text-align: center; }
.status-badge-tunda { background: rgba(255,214,10,0.10); border: 2px solid var(--yellow-warn); border-radius: 8px; padding: 14px 20px; text-align: center; }
.status-badge-belum { background: rgba(255,59,48,0.10); border: 2px solid var(--red-alert); border-radius: 8px; padding: 14px 20px; text-align: center; }
.pipeline-step { display: flex; align-items: center; gap: 12px; padding: 10px 16px; background: var(--bg-panel); border-left: 3px solid var(--green-neon); border-radius: 0 8px 8px 0; margin: 6px 0; font-family: 'Share Tech Mono', monospace; font-size: 12px; color: var(--text-main); }
.pipeline-step .step-num { color: var(--green-neon); font-weight: bold; min-width: 20px; }
.signal-bar { height: 4px; background: linear-gradient(90deg, var(--green-neon), var(--blue-info)); border-radius: 2px; animation: pulse-bar 2s ease-in-out infinite; }
@keyframes pulse-bar { 0%,100%{opacity:0.4} 50%{opacity:1} }
.mono { font-family: 'Share Tech Mono', monospace; }
.tag-green { color: var(--green-neon); font-family: 'Share Tech Mono', monospace; font-size: 11px; background: rgba(0,255,135,0.08); padding: 2px 8px; border-radius: 4px; }
.tag-red { color: var(--red-alert); font-family: 'Share Tech Mono', monospace; font-size: 11px; background: rgba(255,59,48,0.08); padding: 2px 8px; border-radius: 4px; }
.tag-yellow { color: var(--yellow-warn); font-family: 'Share Tech Mono', monospace; font-size: 11px; background: rgba(255,214,10,0.08); padding: 2px 8px; border-radius: 4px; }

/* SIDEBAR */
.sidebar-label { font-family: 'Share Tech Mono', monospace; color: var(--green-neon); font-size: 11px; letter-spacing: 1px; text-transform: uppercase; }
div[data-testid="stSidebarContent"] h1, div[data-testid="stSidebarContent"] h2, div[data-testid="stSidebarContent"] h3 { font-size: 1rem !important; }

/* PROGRESS */
.stProgress > div > div { background: var(--green-neon) !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD & PREPROCESS DATA
# ============================================================
WINDOW = 6

@st.cache_data
def load_data():
    paths = [
        "Log_Data_Bayam_Brazil_1440_2026.xlsx",
        "/mnt/user-data/uploads/Log_Data_Bayam_Brazil_1440_2026.xlsx"
    ]
    df = None
    for p in paths:
        if os.path.exists(p):
            df = pd.read_excel(p)
            break
    if df is None:
        raise FileNotFoundError("Dataset tidak ditemukan.")
    
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
    rename_map = {"STATUS_TANAH_": "STATUS_TANAH", "STATUS TANAH": "STATUS_TANAH"}
    df = df.rename(columns=rename_map)
    
    df["KELEMBAPAN"] = pd.to_numeric(df["KELEMBAPAN"], errors="coerce")
    df["SUHU"] = pd.to_numeric(df["SUHU"], errors="coerce")
    df = df.dropna(subset=["KELEMBAPAN", "SUHU"]).reset_index(drop=True)
    
    def get_status(k):
        if k < 60: return "Kering"
        elif k <= 70: return "Normal"
        else: return "Basah"
    
    df["STATUS_TANAH"] = df["KELEMBAPAN"].apply(get_status)
    df["WAKTU"] = df["WAKTU"].astype(str)
    return df

@st.cache_data
def prepare_features(df):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[["KELEMBAPAN", "SUHU"]].values)
    
    encoder = LabelEncoder()
    encoder.fit(["Basah", "Kering", "Normal"])
    y_enc = encoder.transform(df["STATUS_TANAH"].values)
    
    X, y = [], []
    for i in range(len(X_scaled) - WINDOW):
        X.append(X_scaled[i:i+WINDOW])
        y.append(y_enc[i+WINDOW])
    
    return np.array(X), np.array(y), scaler, encoder

@st.cache_resource
def train_model(X, y, _encoder):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Sequential([
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(WINDOW, 2), name="Conv1D_L1"),
        MaxPooling1D(pool_size=2, name="MaxPool_L1"),
        Conv1D(filters=64, kernel_size=2, activation='relu', name="Conv1D_L2"),
        Flatten(name="Flatten"),
        Dense(64, activation='relu', name="Dense_Hidden"),
        Dropout(0.2, name="Dropout"),
        Dense(3, activation='softmax', name="Dense_Output")
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,
        epochs=50, batch_size=16,
        validation_split=0.2, verbose=0
    )
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=_encoder.classes_, output_dict=True)
    
    return model, history, acc, loss, cm, X_test, y_test, y_pred, report

# ============================================================
# HARVEST ASSESSMENT (ML-DRIVEN)
# ============================================================
def compute_harvest_score(df_window):
    """Score kelayakan panen dari pola sensor 6 titik terakhir."""
    klmbp_vals = df_window["KELEMBAPAN"].values
    suhu_vals = df_window["SUHU"].values
    
    # Faktor 1: Konsistensi kelembapan optimal (60-70%)
    optimal_count = np.sum((klmbp_vals >= 60) & (klmbp_vals <= 70))
    f1 = optimal_count / len(klmbp_vals)
    
    # Faktor 2: Suhu ideal (25-30°C)
    temp_ok = np.sum((suhu_vals >= 25) & (suhu_vals <= 30))
    f2 = temp_ok / len(suhu_vals)
    
    # Faktor 3: Stabilitas kelembapan (std rendah = stabil)
    std_klmbp = np.std(klmbp_vals)
    f3 = max(0, 1 - (std_klmbp / 20))
    
    # Faktor 4: Rata-rata kelembapan mendekati tengah optimal (65%)
    mean_klmbp = np.mean(klmbp_vals)
    dist_from_optimal = abs(mean_klmbp - 65) / 35
    f4 = max(0, 1 - dist_from_optimal)
    
    score = (f1 * 0.35 + f2 * 0.25 + f3 * 0.20 + f4 * 0.20) * 100
    return round(score, 1), f1*100, f2*100, f3*100, f4*100

def get_harvest_status(score):
    if score >= 65:
        return "SIAP PANEN", "#00FF87", "✅"
    elif score >= 40:
        return "TUNDA PANEN", "#FFD60A", "⚠️"
    else:
        return "BELUM LAYAK", "#FF3B30", "❌"

# ============================================================
# PLOTLY THEME HELPER
# ============================================================
PLOT_THEME = dict(
    paper_bgcolor='rgba(7,26,14,0.6)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#D6F5E3', family='Share Tech Mono, monospace', size=11),
    xaxis=dict(gridcolor='rgba(0,255,135,0.06)', showline=True, linecolor='rgba(0,255,135,0.3)'),
    yaxis=dict(gridcolor='rgba(0,255,135,0.06)', showline=True, linecolor='rgba(0,255,135,0.3)'),
    margin=dict(l=40, r=20, t=40, b=40)
)

# ============================================================
# LOAD EVERYTHING
# ============================================================
with st.spinner("🌿 Memuat dataset & melatih model CNN-1D..."):
    df = load_data()
    X_all, y_all, scaler, encoder = prepare_features(df)
    model, history, test_acc, test_loss, cm, X_test, y_test, y_pred, report = train_model(X_all, y_all, encoder)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<p class="sidebar-label">🌿 NEO-FOREST OS v2.0</p>', unsafe_allow_html=True)
    st.markdown("### Sistem Monitoring Bayam Brazil")
    st.markdown("---")
    
    st.markdown("""
<div class="neo-card" style="padding:14px;">
<p class="sidebar-label">📌 Info Tanaman</p>
<p style="margin:8px 0 4px;font-size:13px;"><b>Bayam Brazil</b><br><i>Alternanthera sissoo</i></p>
<p style="font-size:12px;color:#6EA882;margin:4px 0;">Tanaman sayuran daun penutup tanah yang adaptif dan kaya antioksidan.</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("""
<div class="neo-card" style="padding:14px;margin-top:10px;">
<p class="sidebar-label">🎛️ Threshold Sensor</p>
<p style="font-size:12px;margin:6px 0;"><span style="color:#FF3B30;">⬛ Kering</span> &nbsp; Kelembapan &lt; 60%</p>
<p style="font-size:12px;margin:6px 0;"><span style="color:#00FF87;">⬛ Normal</span> &nbsp; Kelembapan 60–70%</p>
<p style="font-size:12px;margin:6px 0;"><span style="color:#0AF0FF;">⬛ Basah</span> &nbsp; Kelembapan &gt; 70%</p>
<hr style="border-color:rgba(0,255,135,0.15);margin:8px 0;">
<p style="font-size:12px;margin:4px 0;"><span style="color:#FFD60A;">🌡️ Suhu Ideal:</span> 25°C – 30°C</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("""
<div class="neo-card" style="padding:14px;margin-top:10px;">
<p class="sidebar-label">🤖 CNN-1D Architecture</p>
<p style="font-size:11px;color:#6EA882;margin:4px 0;font-family:'Share Tech Mono',monospace;">
Input (6 × 2)<br>
→ Conv1D [32, k=2]<br>
→ MaxPool [2]<br>
→ Conv1D [64, k=2]<br>
→ Flatten<br>
→ Dense [64, ReLU]<br>
→ Dropout [0.2]<br>
→ Dense [3, Softmax]
</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown(f"""
<div class="neo-card" style="padding:14px;margin-top:10px;border-color:rgba(0,255,135,0.4);">
<p class="sidebar-label">📊 Model Performance</p>
<p style="font-size:13px;margin:6px 0;">Akurasi Test: <span style="color:#00FF87;font-family:'Share Tech Mono',monospace;">{test_acc*100:.2f}%</span></p>
<p style="font-size:13px;margin:6px 0;">Loss: <span style="color:#FFD60A;font-family:'Share Tech Mono',monospace;">{test_loss:.4f}</span></p>
<p style="font-size:13px;margin:6px 0;">Total Data: <span style="color:#0AF0FF;font-family:'Share Tech Mono',monospace;">1440</span></p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# ⚡ NEO-FOREST MONITORING SYSTEM")
    st.markdown('<p style="color:#6EA882;font-family:\'Share Tech Mono\',monospace;font-size:12px;letter-spacing:2px;">HYDROTECH INTELLIGENCE // CNN-1D SOIL CLASSIFICATION // KELOMPOK 1</p>', unsafe_allow_html=True)
with col_h2:
    st.markdown(f'<div style="text-align:right;padding-top:10px;"><span class="tag-green">● ONLINE</span> &nbsp; <span style="color:#6EA882;font-size:11px;font-family:\'Share Tech Mono\',monospace;">{pd.Timestamp.now().strftime("%d %b %Y")}</span></div>', unsafe_allow_html=True)

st.markdown('<div class="signal-bar" style="margin-bottom:20px;"></div>', unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📡  Monitoring Realtime",
    "🧠  Arsitektur & Evaluasi CNN-1D",
    "🌾  Analisis Kelayakan Panen",
    "📷  Computer Vision Daun"
])

# ============================================================
# TAB 1 — MONITORING REALTIME
# ============================================================
with tab1:
    counter = st_autorefresh(interval=1000, key="rt_counter")
    
    idx = (counter % (len(df) - WINDOW)) + WINDOW
    df_live = df.iloc[:idx]
    row_now = df_live.iloc[-1]
    df_chart = df_live.tail(60)
    
    # CNN-1D prediction
    sample_raw = df_live[["KELEMBAPAN", "SUHU"]].iloc[-WINDOW:]
    sample_sc = scaler.transform(sample_raw.values).reshape(1, WINDOW, 2)
    pred_prob = model.predict(sample_sc, verbose=0)[0]
    pred_idx = np.argmax(pred_prob)
    pred_label = encoder.classes_[pred_idx]
    pred_conf = pred_prob[pred_idx] * 100
    
    # Harvest score
    score, f1s, f2s, f3s, f4s = compute_harvest_score(df_live.tail(WINDOW))
    harv_status, harv_color, harv_icon = get_harvest_status(score)
    
    # === ALUR PIPELINE ===
    st.markdown("#### 🔄 Alur Proses CNN-1D Realtime")
    st.markdown(f"""
<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:16px;">
  <div class="pipeline-step"><span class="step-num">01</span>Sensor membaca KELEMBAPAN &amp; SUHU</div>
  <div style="color:#00FF87;font-size:18px;">→</div>
  <div class="pipeline-step"><span class="step-num">02</span>Buffer 6 titik waktu (window=6)</div>
  <div style="color:#00FF87;font-size:18px;">→</div>
  <div class="pipeline-step"><span class="step-num">03</span>MinMaxScaler normalisasi [0,1]</div>
  <div style="color:#00FF87;font-size:18px;">→</div>
  <div class="pipeline-step"><span class="step-num">04</span>CNN-1D ekstrak fitur temporal</div>
  <div style="color:#00FF87;font-size:18px;">→</div>
  <div class="pipeline-step"><span class="step-num">05</span>Softmax → Prediksi Status Tanah</div>
  <div style="color:#00FF87;font-size:18px;">→</div>
  <div class="pipeline-step"><span class="step-num">06</span>Keputusan Irigasi Otomatis</div>
</div>
""", unsafe_allow_html=True)
    
    # === METRIC CARDS ===
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💧 Kelembapan", f"{row_now['KELEMBAPAN']}%", f"Data #{int(row_now['NO'])}")
    c2.metric("🌡️ Suhu", f"{row_now['SUHU']}°C")
    c3.metric("🏷️ Status Aktual", row_now["STATUS_TANAH"])
    c4.metric("🤖 CNN-1D Prediksi", pred_label, f"Conf: {pred_conf:.1f}%")
    c5.metric("🌾 Skor Panen", f"{score}%", harv_status)
    
    # === STATUS BANNER ===
    st.markdown("")
    k_val = int(row_now['KELEMBAPAN'])
    if k_val > 70:
        st.markdown(f'<div style="background:rgba(10,240,255,0.08);border:1px solid #0AF0FF;border-radius:8px;padding:12px 20px;font-family:\'Share Tech Mono\',monospace;font-size:13px;color:#0AF0FF;">🌊 <b>[NODE-{int(row_now["NO"])}] STATUS TANAH: BASAH</b> — Kelembapan {k_val}% melebihi batas atas. Sistem menonaktifkan pompa irigasi.</div>', unsafe_allow_html=True)
    elif 60 <= k_val <= 70:
        st.markdown(f'<div style="background:rgba(0,255,135,0.08);border:1px solid #00FF87;border-radius:8px;padding:12px 20px;font-family:\'Share Tech Mono\',monospace;font-size:13px;color:#00FF87;">🌱 <b>[NODE-{int(row_now["NO"])}] STATUS TANAH: NORMAL</b> — Kelembapan {k_val}% dalam rentang optimal. Irigasi terjadwal.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="background:rgba(255,59,48,0.08);border:1px solid #FF3B30;border-radius:8px;padding:12px 20px;font-family:\'Share Tech Mono\',monospace;font-size:13px;color:#FF3B30;">☀️ <b>[NODE-{int(row_now["NO"])}] STATUS TANAH: KERING</b> — Kelembapan {k_val}% di bawah batas minimum. Pompa irigasi AKTIF sekarang!</div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    # === CHARTS ===
    g1, g2 = st.columns(2)
    
    with g1:
        fig1 = go.Figure()
        # Zona optimal shading
        fig1.add_hrect(y0=60, y1=70, fillcolor="rgba(0,255,135,0.06)", line_width=0, annotation_text="Zona Optimal", annotation_font=dict(color="#00FF87", size=10))
        fig1.add_trace(go.Scatter(
            x=df_chart["WAKTU"].astype(str) + " #" + df_chart["NO"].astype(str),
            y=df_chart["KELEMBAPAN"],
            mode="lines", fill="tozeroy",
            fillcolor="rgba(0,255,135,0.07)",
            line=dict(color='#00FF87', width=2),
            name="Kelembapan"
        ))
        fig1.add_hline(y=60, line_dash="dot", line_color="rgba(255,59,48,0.5)", annotation_text="60% min")
        fig1.add_hline(y=70, line_dash="dot", line_color="rgba(10,240,255,0.5)", annotation_text="70% max")
        fig1.update_layout(title="📡 TREN KELEMBAPAN TANAH (%)", **PLOT_THEME, height=280, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with g2:
        fig2 = go.Figure()
        fig2.add_hrect(y0=25, y1=30, fillcolor="rgba(255,214,10,0.05)", line_width=0, annotation_text="Zona Ideal", annotation_font=dict(color="#FFD60A", size=10))
        fig2.add_trace(go.Scatter(
            x=df_chart["WAKTU"].astype(str) + " #" + df_chart["NO"].astype(str),
            y=df_chart["SUHU"],
            mode="lines", fill="tozeroy",
            fillcolor="rgba(255,59,48,0.07)",
            line=dict(color='#FF3B30', width=2),
            name="Suhu"
        ))
        fig2.update_layout(title="🌡️ TREN SUHU UDARA (°C)", **PLOT_THEME, height=280, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    # CNN Confidence bar
    st.markdown("#### 🤖 Distribusi Probabilitas CNN-1D (Output Softmax)")
    prob_cols = st.columns(3)
    classes = encoder.classes_
    colors = {"Basah": "#0AF0FF", "Kering": "#FF3B30", "Normal": "#00FF87"}
    for i, cls in enumerate(classes):
        with prob_cols[i]:
            pct = pred_prob[i] * 100
            col_c = colors.get(cls, "#00FF87")
            st.markdown(f"""
<div style="background:var(--bg-card);border:1px solid {col_c}30;border-radius:8px;padding:14px;text-align:center;">
  <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#6EA882;letter-spacing:1px;">{cls.upper()}</div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:1.6rem;color:{col_c};margin:4px 0;">{pct:.1f}%</div>
  <div style="background:rgba(0,0,0,0.3);border-radius:4px;height:6px;overflow:hidden;">
    <div style="width:{pct}%;height:100%;background:{col_c};border-radius:4px;transition:width 0.5s;"></div>
  </div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown(f"#### 📋 Stream Data Realtime &nbsp;<span class='tag-green'>{idx} / 1440</span>", unsafe_allow_html=True)
    st.dataframe(
        df_live.sort_values("NO", ascending=False)[["NO","HARI","TANGGAL","WAKTU","KELEMBAPAN","SUHU","STATUS_TANAH"]].head(30),
        use_container_width=True, height=250
    )

# ============================================================
# TAB 2 — CNN-1D ARCHITECTURE & EVALUATION
# ============================================================
with tab2:
    st.markdown("#### 🧠 Arsitektur CNN-1D untuk Klasifikasi Status Tanah")
    
    # Architecture diagram
    st.markdown("""
<div class="neo-card" style="padding:20px;">
<p style="font-family:'Share Tech Mono',monospace;color:#6EA882;font-size:11px;letter-spacing:1px;">ARSITEKTUR MODEL</p>
<div style="display:flex;gap:8px;align-items:stretch;flex-wrap:wrap;margin-top:12px;">
  <div style="background:rgba(0,240,255,0.08);border:1px solid rgba(0,240,255,0.3);border-radius:8px;padding:12px;text-align:center;min-width:120px;">
    <div style="color:#0AF0FF;font-family:'Share Tech Mono',monospace;font-size:11px;">INPUT</div>
    <div style="color:#D6F5E3;font-size:13px;margin:6px 0;font-weight:600;">6 × 2</div>
    <div style="color:#6EA882;font-size:10px;">6 timestep<br>2 fitur</div>
  </div>
  <div style="color:#00FF87;font-size:20px;display:flex;align-items:center;">→</div>
  <div style="background:rgba(0,255,135,0.08);border:1px solid rgba(0,255,135,0.3);border-radius:8px;padding:12px;text-align:center;min-width:120px;">
    <div style="color:#00FF87;font-family:'Share Tech Mono',monospace;font-size:11px;">CONV1D #1</div>
    <div style="color:#D6F5E3;font-size:13px;margin:6px 0;font-weight:600;">32 filter</div>
    <div style="color:#6EA882;font-size:10px;">kernel=2<br>ReLU</div>
  </div>
  <div style="color:#00FF87;font-size:20px;display:flex;align-items:center;">→</div>
  <div style="background:rgba(0,255,135,0.06);border:1px solid rgba(0,255,135,0.2);border-radius:8px;padding:12px;text-align:center;min-width:110px;">
    <div style="color:#00FF87;font-family:'Share Tech Mono',monospace;font-size:11px;">MAXPOOL</div>
    <div style="color:#D6F5E3;font-size:13px;margin:6px 0;font-weight:600;">pool=2</div>
    <div style="color:#6EA882;font-size:10px;">Reduksi<br>dimensi</div>
  </div>
  <div style="color:#00FF87;font-size:20px;display:flex;align-items:center;">→</div>
  <div style="background:rgba(0,255,135,0.08);border:1px solid rgba(0,255,135,0.3);border-radius:8px;padding:12px;text-align:center;min-width:120px;">
    <div style="color:#00FF87;font-family:'Share Tech Mono',monospace;font-size:11px;">CONV1D #2</div>
    <div style="color:#D6F5E3;font-size:13px;margin:6px 0;font-weight:600;">64 filter</div>
    <div style="color:#6EA882;font-size:10px;">kernel=2<br>ReLU</div>
  </div>
  <div style="color:#00FF87;font-size:20px;display:flex;align-items:center;">→</div>
  <div style="background:rgba(255,214,10,0.06);border:1px solid rgba(255,214,10,0.2);border-radius:8px;padding:12px;text-align:center;min-width:100px;">
    <div style="color:#FFD60A;font-family:'Share Tech Mono',monospace;font-size:11px;">FLATTEN</div>
    <div style="color:#D6F5E3;font-size:13px;margin:6px 0;font-weight:600;">→ 1D</div>
    <div style="color:#6EA882;font-size:10px;">Ratakan<br>tensor</div>
  </div>
  <div style="color:#00FF87;font-size:20px;display:flex;align-items:center;">→</div>
  <div style="background:rgba(255,59,48,0.06);border:1px solid rgba(255,59,48,0.2);border-radius:8px;padding:12px;text-align:center;min-width:100px;">
    <div style="color:#FF3B30;font-family:'Share Tech Mono',monospace;font-size:11px;">DENSE+DROP</div>
    <div style="color:#D6F5E3;font-size:13px;margin:6px 0;font-weight:600;">64 + 0.2</div>
    <div style="color:#6EA882;font-size:10px;">ReLU<br>Dropout</div>
  </div>
  <div style="color:#00FF87;font-size:20px;display:flex;align-items:center;">→</div>
  <div style="background:rgba(0,255,135,0.12);border:2px solid #00FF87;border-radius:8px;padding:12px;text-align:center;min-width:110px;">
    <div style="color:#00FF87;font-family:'Share Tech Mono',monospace;font-size:11px;">OUTPUT</div>
    <div style="color:#D6F5E3;font-size:13px;margin:6px 0;font-weight:600;">3 kelas</div>
    <div style="color:#6EA882;font-size:10px;">Softmax<br>Basah/Kering/Normal</div>
  </div>
</div>
<p style="font-family:'Share Tech Mono',monospace;color:#6EA882;font-size:10px;margin-top:12px;">Optimizer: Adam | Loss: Sparse Categorical Crossentropy | Epochs: 50 | Batch: 16</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("")
    
    # Training curves + Confusion matrix
    ev1, ev2 = st.columns(2)
    
    with ev1:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss', line=dict(color='#FF3B30', width=2)))
        fig_loss.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss', line=dict(color='#FFD60A', width=2, dash='dash')))
        fig_loss.update_layout(title="📉 Kurva Loss Training", **PLOT_THEME, height=280, legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#D6F5E3')))
        st.plotly_chart(fig_loss, use_container_width=True)
        
    with ev2:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(y=history.history['accuracy'], name='Train Acc', line=dict(color='#00FF87', width=2)))
        fig_acc.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Val Acc', line=dict(color='#0AF0FF', width=2, dash='dash')))
        fig_acc.update_layout(title="📈 Kurva Akurasi Training", **PLOT_THEME, height=280, legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#D6F5E3')))
        st.plotly_chart(fig_acc, use_container_width=True)
    
    cm1, cm2 = st.columns(2)
    with cm1:
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=encoder.classes_, y=encoder.classes_,
            color_continuous_scale=[[0,'#020D07'],[0.5,'#0A5C34'],[1,'#00FF87']],
            labels=dict(x="Prediksi", y="Aktual", color="Jumlah")
        )
        fig_cm.update_layout(title="📊 Confusion Matrix", **PLOT_THEME, height=300, coloraxis_showscale=False)
        fig_cm.update_traces(textfont=dict(color='white', size=14))
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with cm2:
        st.markdown("#### 📋 Classification Report")
        for cls in encoder.classes_:
            r = report[cls]
            st.markdown(f"""
<div style="background:var(--bg-panel);border-radius:8px;padding:10px 16px;margin:6px 0;display:flex;justify-content:space-between;align-items:center;">
  <span style="font-family:'Share Tech Mono',monospace;font-size:13px;color:#D6F5E3;">{cls}</span>
  <span class="tag-green">P: {r['precision']:.2f}</span>
  <span class="tag-green">R: {r['recall']:.2f}</span>
  <span class="tag-green">F1: {r['f1-score']:.2f}</span>
  <span style="color:#6EA882;font-size:11px;">n={int(r['support'])}</span>
</div>
""", unsafe_allow_html=True)
        st.markdown(f"""
<div style="background:rgba(0,255,135,0.08);border:1px solid #00FF87;border-radius:8px;padding:10px 16px;margin-top:10px;">
  <span style="font-family:'Share Tech Mono',monospace;color:#00FF87;">AKURASI KESELURUHAN: {test_acc*100:.2f}%</span>
</div>
""", unsafe_allow_html=True)

    # Distribution pie
    st.markdown("#### 📊 Distribusi Kelas Dataset")
    dist_cols = st.columns(2)
    with dist_cols[0]:
        counts = df["STATUS_TANAH"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.5,
            marker=dict(colors=['#0AF0FF','#FF3B30','#00FF87'])
        ))
        fig_pie.update_layout(title="Distribusi Status Tanah", **PLOT_THEME, height=280, legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#D6F5E3')))
        st.plotly_chart(fig_pie, use_container_width=True)
    with dist_cols[1]:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=df["KELEMBAPAN"], nbinsx=30, marker_color='#00FF87', opacity=0.7, name="Kelembapan"))
        fig_hist.update_layout(title="Distribusi Kelembapan Sensor", **PLOT_THEME, height=280, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================
# TAB 3 — HARVEST ASSESSMENT
# ============================================================
with tab3:
    counter3 = st_autorefresh(interval=2000, key="harvest_counter")
    
    idx3 = (counter3 % (len(df) - WINDOW)) + WINDOW
    df_live3 = df.iloc[:idx3]
    window_df = df_live3.tail(WINDOW)
    
    score3, f1s3, f2s3, f3s3, f4s3 = compute_harvest_score(window_df)
    status3, color3, icon3 = get_harvest_status(score3)
    
    st.markdown("#### 🌾 Sistem Penilaian Kelayakan Panen Berbasis Sensor CNN-1D")
    st.markdown("""
<div class="neo-card" style="padding:14px;margin-bottom:16px;">
<p style="font-size:12px;color:#6EA882;font-family:'Share Tech Mono',monospace;">Sistem ini menghitung skor kelayakan panen dari 4 faktor sensor dalam 6 window waktu terakhir, disesuaikan dengan model penyiraman otomatis. Skor ≥65% = Siap Panen, 40–64% = Tunda Panen, &lt;40% = Belum Layak.</p>
</div>
""", unsafe_allow_html=True)
    
    # Big status display
    badge_class = {"SIAP PANEN": "status-badge-panen", "TUNDA PANEN": "status-badge-tunda", "BELUM LAYAK": "status-badge-belum"}[status3]
    st.markdown(f"""
<div class="{badge_class}" style="margin:0 0 20px;">
  <div style="font-family:'Share Tech Mono',monospace;font-size:2rem;color:{color3};">{icon3} {status3}</div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:1rem;color:{color3};opacity:0.7;">Skor Kelayakan: {score3}%</div>
</div>
""", unsafe_allow_html=True)
    
    # Score gauge + factors
    ga1, ga2 = st.columns([1, 1])
    with ga1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score3,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "SKOR KELAYAKAN PANEN", 'font': {'color': '#D6F5E3', 'family': 'Share Tech Mono', 'size': 13}},
            number={'suffix': '%', 'font': {'color': color3, 'family': 'Share Tech Mono', 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#6EA882', 'tickfont': {'color': '#6EA882', 'size': 10}},
                'bar': {'color': color3},
                'bgcolor': '#071A0E',
                'bordercolor': '#0A2416',
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(255,59,48,0.15)'},
                    {'range': [40, 65], 'color': 'rgba(255,214,10,0.12)'},
                    {'range': [65, 100], 'color': 'rgba(0,255,135,0.12)'}
                ],
                'threshold': {'line': {'color': color3, 'width': 3}, 'thickness': 0.75, 'value': score3}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(7,26,14,0.6)', font=dict(color='#D6F5E3'), height=300, margin=dict(l=30, r=30, t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with ga2:
        st.markdown("#### 🔬 Faktor Penilaian (6 Window Terakhir)")
        
        factors = [
            ("Konsistensi Kelembapan Optimal (60-70%)", f1s3, 35),
            ("Suhu Lingkungan Ideal (25-30°C)", f2s3, 25),
            ("Stabilitas Sensor Kelembapan", f3s3, 20),
            ("Rata-rata Mendekati 65% Optimal", f4s3, 20),
        ]
        for name, val, weight in factors:
            bar_color = "#00FF87" if val >= 70 else "#FFD60A" if val >= 40 else "#FF3B30"
            st.markdown(f"""
<div style="margin:10px 0;">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
    <span style="font-size:11px;color:#D6F5E3;">{name}</span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:11px;color:{bar_color};">{val:.1f}% <span style="color:#6EA882;">(bobot {weight}%)</span></span>
  </div>
  <div style="background:#0A2416;border-radius:4px;height:8px;overflow:hidden;">
    <div style="width:{val}%;height:100%;background:{bar_color};border-radius:4px;transition:width 0.6s;"></div>
  </div>
</div>
""", unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("#### 📋 Rekomendasi Sistem")
    if status3 == "SIAP PANEN":
        st.markdown("""
<div class="neo-card" style="border-color:rgba(0,255,135,0.5);">
<p style="color:#00FF87;font-family:'Share Tech Mono',monospace;font-size:14px;margin-bottom:10px;">✅ KONDISI OPTIMAL — SEGERA LAKUKAN PANEN</p>
<p style="font-size:13px;">1. <b>Panen Selektif:</b> Potong daun mulai dari lapisan luar, sisakan 3–4 helai daun bawah agar tanaman terus bertunas.</p>
<p style="font-size:13px;">2. <b>Waktu Terbaik:</b> Lakukan panen pada pagi hari (06:00–09:00) untuk kandungan fitokimia dan antioksidan tertinggi.</p>
<p style="font-size:13px;">3. <b>Pasca Panen:</b> Kurangi frekuensi penyiraman 20% selama 3 hari untuk stimulasi pertumbuhan tunas baru.</p>
<p style="font-size:13px;">4. <b>Estimasi Hasil:</b> Kerapatan klorofil tinggi mengindikasikan biomassa daun dalam kondisi puncak komersial.</p>
</div>
""", unsafe_allow_html=True)
    elif status3 == "TUNDA PANEN":
        st.markdown("""
<div class="neo-card" style="border-color:rgba(255,214,10,0.5);">
<p style="color:#FFD60A;font-family:'Share Tech Mono',monospace;font-size:14px;margin-bottom:10px;">⚠️ KONDISI CUKUP — TUNDA 3–5 HARI</p>
<p style="font-size:13px;">1. <b>Stabilisasi Kelembapan:</b> Pertahankan kelembapan di zona 60–70% secara konsisten selama minimal 72 jam ke depan.</p>
<p style="font-size:13px;">2. <b>Monitor Suhu:</b> Pastikan suhu tidak melebihi 30°C. Jika perlu, tambahkan naungan parsial (shading net 30%).</p>
<p style="font-size:13px;">3. <b>Nutrisi:</b> Aplikasikan pupuk cair daun berbasis nitrogen untuk mempercepat pembentukan klorofil optimal.</p>
<p style="font-size:13px;">4. <b>Re-evaluasi:</b> Sistem akan otomatis re-evaluasi setiap 6 siklus pembacaan sensor berikutnya.</p>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="neo-card" style="border-color:rgba(255,59,48,0.5);">
<p style="color:#FF3B30;font-family:'Share Tech Mono',monospace;font-size:14px;margin-bottom:10px;">❌ KONDISI KRITIS — JANGAN PANEN DULU</p>
<p style="font-size:13px;">1. <b>Perbaikan Irigasi Darurat:</b> Aktifkan pompa irigasi. Target kelembapan minimum 60% dalam 24 jam.</p>
<p style="font-size:13px;">2. <b>Cek Sistem Sensor:</b> Validasi bacaan sensor — kelembapan ekstrem mungkin disebabkan oleh kalibrasi sensor yang meleset.</p>
<p style="font-size:13px;">3. <b>Hentikan Panen Total:</b> Memanen saat kondisi kritis akan menyebabkan tanaman stress dan gagal bertunas kembali.</p>
<p style="font-size:13px;">4. <b>Recovery Plan:</b> Butuh 7–10 hari normalisasi sebelum dapat dievaluasi kembali untuk kelayakan panen.</p>
</div>
""", unsafe_allow_html=True)
    
    # Historical harvest score trend
    st.markdown("#### 📈 Tren Skor Kelayakan Panen (50 Pembacaan Terakhir)")
    scores_history = []
    step = max(1, idx3 // 50)
    for i in range(WINDOW, min(idx3, len(df)), step):
        win = df.iloc[i-WINDOW:i]
        s, *_ = compute_harvest_score(win)
        scores_history.append({"No": i, "Skor": s})
    
    if scores_history:
        df_scores = pd.DataFrame(scores_history)
        fig_hs = go.Figure()
        fig_hs.add_hrect(y0=65, y1=100, fillcolor="rgba(0,255,135,0.06)", line_width=0)
        fig_hs.add_hrect(y0=40, y1=65, fillcolor="rgba(255,214,10,0.04)", line_width=0)
        fig_hs.add_hrect(y0=0, y1=40, fillcolor="rgba(255,59,48,0.04)", line_width=0)
        fig_hs.add_trace(go.Scatter(x=df_scores["No"], y=df_scores["Skor"], mode="lines", line=dict(color='#00FF87', width=2.5), fill="tozeroy", fillcolor="rgba(0,255,135,0.06)"))
        fig_hs.add_hline(y=65, line_dash="dot", line_color="rgba(0,255,135,0.5)", annotation_text="Siap Panen (65%)")
        fig_hs.add_hline(y=40, line_dash="dot", line_color="rgba(255,214,10,0.5)", annotation_text="Tunda Panen (40%)")
        fig_hs.update_layout(title="Skor Kelayakan Panen (%)", **PLOT_THEME, height=250, showlegend=False)
        st.plotly_chart(fig_hs, use_container_width=True)

# ============================================================
# TAB 4 — COMPUTER VISION
# ============================================================
with tab4:
    st.markdown("#### 📷 NEO-VISION: Analisis Citra Daun Bayam Brazil")
    st.markdown("""
<div class="neo-card" style="padding:14px;margin-bottom:16px;">
<p style="font-size:12px;color:#6EA882;">Unggah foto daun Bayam Brazil untuk analisis kelayakan panen berbasis computer vision. Sistem mengekstrak indeks warna piksel RGB untuk menilai kesehatan klorofil daun dan menentukan status: <span style="color:#FF3B30;">Tidak Layak</span>, <span style="color:#FFD60A;">Tunda Panen</span>, atau <span style="color:#00FF87;">Siap Panen</span>.</p>
</div>
""", unsafe_allow_html=True)
    
    # === ALUR CV ===
    st.markdown("##### 🔄 Alur Analisis Computer Vision")
    st.markdown("""
<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:20px;">
  <div class="pipeline-step"><span class="step-num">01</span>Upload foto daun</div>
  <div style="color:#00FF87;">→</div>
  <div class="pipeline-step"><span class="step-num">02</span>Ekstrak matriks piksel RGB</div>
  <div style="color:#00FF87;">→</div>
  <div class="pipeline-step"><span class="step-num">03</span>Segmentasi warna (hijau sehat / kuning-cokelat sakit)</div>
  <div style="color:#00FF87;">→</div>
  <div class="pipeline-step"><span class="step-num">04</span>Hitung indeks klorofil estimasi</div>
  <div style="color:#00FF87;">→</div>
  <div class="pipeline-step"><span class="step-num">05</span>Scoring + keputusan panen</div>
</div>
""", unsafe_allow_html=True)
    
    file_img = st.file_uploader("Unggah Foto Daun (.png, .jpg, .jpeg)", type=["png","jpg","jpeg"])
    
    if file_img is not None:
        img = Image.open(file_img).convert("RGB")
        img_np = np.array(img)
        
        r = img_np[:,:,0].astype(float)
        g = img_np[:,:,1].astype(float)
        b = img_np[:,:,2].astype(float)
        total_px = img_np.shape[0] * img_np.shape[1]
        
        # Segmentation masks
        green_mask = (g > r) & (g > b) & (g > 50) & (g > 80)           # Hijau dominan sehat
        dark_green  = (g > r) & (g > b) & (g <= 80) & (g > 40)         # Hijau tua / rimbun
        yellow_mask = (r > b) & (g > b) & (r > 80) & (~green_mask) & (~dark_green)  # Klorosis
        brown_mask  = (r > g) & (r > b) & (r > 60) & (g < 80)          # Bercak cokelat
        
        p_sehat   = (np.sum(green_mask) + np.sum(dark_green)) / total_px * 100
        p_klorosis = np.sum(yellow_mask) / total_px * 100
        p_cokelat = np.sum(brown_mask) / total_px * 100
        p_lain    = max(0, 100 - p_sehat - p_klorosis - p_cokelat)
        
        # Scoring
        # Raw cv score based on pixel health ratio
        cv_score = (p_sehat * 0.7) + (p_klorosis * -0.3) + (p_cokelat * -0.5)
        cv_score = max(0, min(100, cv_score))
        
        # Harvest decision
        if cv_score >= 55:
            cv_status, cv_color, cv_icon = "SIAP PANEN", "#00FF87", "✅"
        elif cv_score >= 30:
            cv_status, cv_color, cv_icon = "TUNDA PANEN", "#FFD60A", "⚠️"
        else:
            cv_status, cv_color, cv_icon = "TIDAK LAYAK PANEN", "#FF3B30", "❌"
        
        # SPAD estimation
        spad_est = p_sehat * 0.68 + (100 - p_sehat) * 0.12
        
        cv1, cv2 = st.columns([1, 1.1])
        
        with cv1:
            st.image(img, caption="📸 Citra Input Daun", use_container_width=True)
            
            # Donut chart
            fig_donut = go.Figure(go.Pie(
                labels=['Hijau Sehat (Klorofil Tinggi)', 'Klorosis (Kuning)', 'Bercak Cokelat', 'Latar/Lainnya'],
                values=[max(p_sehat, 0.1), max(p_klorosis, 0.1), max(p_cokelat, 0.1), max(p_lain, 0.1)],
                hole=0.5,
                marker=dict(colors=['#00FF87', '#FFD60A', '#8B4513', '#4A4A4A']),
                textfont=dict(color='white', size=10)
            ))
            fig_donut.update_layout(
                title="🎨 Komposisi Warna Piksel Daun",
                **PLOT_THEME, height=300,
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#D6F5E3', size=10))
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        
        with cv2:
            # Status badge
            badge_cv = {"SIAP PANEN": "status-badge-panen", "TUNDA PANEN": "status-badge-tunda", "TIDAK LAYAK PANEN": "status-badge-belum"}[cv_status]
            st.markdown(f"""
<div class="{badge_cv}" style="margin-bottom:16px;">
  <div style="font-family:'Share Tech Mono',monospace;font-size:1.6rem;color:{cv_color};">{cv_icon} {cv_status}</div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:12px;color:{cv_color};opacity:0.7;">Computer Vision Score: {cv_score:.1f}%</div>
</div>
""", unsafe_allow_html=True)
            
            st.markdown("##### 🔬 Hasil Analisis Piksel")
            pixel_data = [
                ("Piksel Hijau Sehat (Klorofil Aktif)", p_sehat, "#00FF87"),
                ("Piksel Klorosis (Daun Menguning)", p_klorosis, "#FFD60A"),
                ("Piksel Bercak Cokelat (Penyakit/Hama)", p_cokelat, "#8B4513"),
                ("Latar Belakang / Lainnya", p_lain, "#4A4A4A"),
            ]
            for label, val, bar_c in pixel_data:
                st.markdown(f"""
<div style="margin:8px 0;">
  <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
    <span style="font-size:11px;color:#D6F5E3;">{label}</span>
    <span style="font-family:'Share Tech Mono',monospace;font-size:11px;color:{bar_c};">{val:.1f}%</span>
  </div>
  <div style="background:#0A2416;border-radius:3px;height:7px;">
    <div style="width:{min(val,100)}%;height:100%;background:{bar_c};border-radius:3px;"></div>
  </div>
</div>
""", unsafe_allow_html=True)
            
            st.markdown("")
            m1, m2, m3 = st.columns(3)
            m1.metric("SPAD Est.", f"{spad_est:.1f}")
            m2.metric("Keparahan Hama", f"{p_klorosis + p_cokelat:.1f}%")
            m3.metric("CV Score", f"{cv_score:.1f}%")
            
            # Recommendations
            st.markdown("##### 📋 Rekomendasi Agronomi")
            if cv_status == "SIAP PANEN":
                st.markdown("""
<div class="neo-card" style="border-color:rgba(0,255,135,0.4);padding:14px;">
<p style="font-size:12px;">
✅ Kerapatan klorofil sangat tinggi. Daun siap dipanen secara selektif.<br>
🌿 Potong menggunakan gunting steril, sisakan 3–4 helai daun bawah.<br>
🕕 Waktu optimal: pagi hari sebelum jam 10.00.<br>
💧 Kurangi irigasi 20% pasca panen untuk merangsang tunas baru.
</p>
</div>
""", unsafe_allow_html=True)
            elif cv_status == "TUNDA PANEN":
                st.markdown("""
<div class="neo-card" style="border-color:rgba(255,214,10,0.4);padding:14px;">
<p style="font-size:12px;">
⚠️ Terdeteksi indikasi klorosis ringan. Tunda panen 3–5 hari.<br>
💉 Aplikasikan pupuk cair nitrogen (NPK 30-10-10) dosis 2 ml/L.<br>
🌡️ Pastikan suhu siang tidak melebihi 30°C — tambahkan naungan jika perlu.<br>
🔍 Re-scan daun setelah 72 jam untuk validasi pemulihan.
</p>
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown("""
<div class="neo-card" style="border-color:rgba(255,59,48,0.4);padding:14px;">
<p style="font-size:12px;">
❌ Klorosis parah dan/atau infeksi hama terdeteksi. Jangan panen.<br>
🚨 Karantina tanaman dari koloni sehat — risiko penularan patogen tinggi.<br>
💊 Aplikasikan fungisida organik (Trichoderma sp.) + insektisida nabati.<br>
🔧 Cek sistem irigasi: ketidakseimbangan kelembapan memperparah klorosis.<br>
⏳ Estimasi recovery: 10–14 hari dengan perawatan intensif.
</p>
</div>
""", unsafe_allow_html=True)
    else:
        # Placeholder upload guide
        st.markdown("""
<div style="border:2px dashed rgba(0,255,135,0.3);border-radius:12px;padding:40px;text-align:center;margin-top:20px;">
<div style="font-size:3rem;">🌿</div>
<p style="font-family:'Share Tech Mono',monospace;color:#6EA882;margin:10px 0;">UNGGAH FOTO DAUN BAYAM BRAZIL</p>
<p style="font-size:12px;color:#4A4A4A;">Format: PNG, JPG, JPEG | Resolusi minimal 200×200px<br>Foto dari jarak dekat (makro) menghasilkan analisis lebih akurat</p>
<div style="display:flex;justify-content:center;gap:20px;margin-top:20px;flex-wrap:wrap;">
  <div style="text-align:center;">
    <div style="width:60px;height:60px;background:rgba(255,59,48,0.2);border:2px solid #FF3B30;border-radius:8px;display:inline-flex;align-items:center;justify-content:center;font-size:1.5rem;">❌</div>
    <p style="font-size:11px;color:#FF3B30;margin:6px 0;font-family:'Share Tech Mono',monospace;">Tidak Layak<br><span style="color:#6EA882;">CV Score &lt;30%</span></p>
  </div>
  <div style="text-align:center;">
    <div style="width:60px;height:60px;background:rgba(255,214,10,0.2);border:2px solid #FFD60A;border-radius:8px;display:inline-flex;align-items:center;justify-content:center;font-size:1.5rem;">⚠️</div>
    <p style="font-size:11px;color:#FFD60A;margin:6px 0;font-family:'Share Tech Mono',monospace;">Tunda Panen<br><span style="color:#6EA882;">CV Score 30–55%</span></p>
  </div>
  <div style="text-align:center;">
    <div style="width:60px;height:60px;background:rgba(0,255,135,0.2);border:2px solid #00FF87;border-radius:8px;display:inline-flex;align-items:center;justify-content:center;font-size:1.5rem;">✅</div>
    <p style="font-size:11px;color:#00FF87;margin:6px 0;font-family:'Share Tech Mono',monospace;">Siap Panen<br><span style="color:#6EA882;">CV Score ≥55%</span></p>
  </div>
</div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 📋 50 Data Awal Master Dataset")
    st.dataframe(df.head(50)[["NO","HARI","TANGGAL","WAKTU","KELEMBAPAN","SUHU","STATUS_TANAH"]], use_container_width=True, height=300)
