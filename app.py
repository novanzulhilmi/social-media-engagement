import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
import json
import requests
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu  # <-- LIBRARY BARU

# --- Kamus Bahasa & Bendera ---
LANG_MAP = {
    'pt': 'Portuguese 🇵🇹',
    'ru': 'Russian 🇷🇺',
    'ar': 'Arabic 🇸🇦',
    'ja': 'Japanese 🇯🇵',
    'fr': 'French 🇫🇷',
    'en': 'English 🇬🇧',
    'es': 'Spanish 🇪🇸',
    'de': 'German 🇩🇪',
    'zh': 'Chinese 🇨🇳',
    'hi': 'Hindi 🇮🇳',
    'ko': 'Korean 🇰🇷',
    'id': 'Indonesian 🇮🇩',
    'it': 'Italian 🇮🇹'
}
REVERSE_LANG_MAP = {v: k for k, v in LANG_MAP.items()}

# Mengabaikan warning spesifik dari sklearn
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Analisis Engagement", page_icon="🚀", layout="wide")

# --- CSS KUSTOM untuk TAMPILAN ---
CSS_STYLE = """
<style>
/* ... (CSS Keyframes, Global, Tombol Tetap Sama) ... */
@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Tombol Primary (Prakiraan & Navigasi) */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
    color: white;
    border-radius: 20px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
    transition: all 0.3s ease;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    box-shadow: 0 5px 15px rgba(0,0,200,0.3);
    transform: translateY(-2px);
}

/* --- Sidebar --- */
/* Hapus gradient, biarkan option_menu yang mengatur */
[data-testid="stSidebar"] {
    background: var(--secondary-background-color);
    border-right: 1px solid var(--background-color);
}

/* --- Tabs --- */
div[data-testid="stTabs"] button {
    border-radius: 8px;
    padding: 10px 15px;
    color: #2575fc; /* Warna teks tab */
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: #e0eaff;
    color: #6a11cb;
    font-weight: bold;
    border-bottom: 3px solid #6a11cb;
}

/* --- Metric/KPI --- */
div[data-testid="stMetric"] {
    background-color: var(--secondary-background-color); 
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    animation: fadeIn 0.5s ease-out; /* Animasi Fade-in BARU */
}
div[data-testid="stMetric"] > div:nth-child(1) {
    color: var(--text-color); 
    opacity: 0.8;
}
div[data-testid="stMetric"] > div:nth-child(2) {
    font-size: 2.2rem; /* Value */
    font-weight: bold;
    color: var(--text-color); 
}

/* --- Info Box --- */
div[data-testid="stInfo"] {
    background-color: rgba(37, 117, 252, 0.1);
    border-color: #2575fc;
    border-radius: 10px;
    padding: 15px;
    color: var(--text-color); 
}

/* --- CSS untuk Gambar Responsif (BARU) --- */
.responsive-image {
    max-width: 100%; /* Sesuai permintaan Anda */
    height: auto;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

/* --- CSS untuk Kartu Presentasi (BARU) --- */
.presentation-card {
    background-color: var(--secondary-background-color);
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    animation: fadeIn 0.7s ease-out;
    height: 100%;
    border-left: 5px solid #6a11cb; /* <-- Tambahan untuk visual */
    transition: all 0.3s ease; /* BARU: transisi hover */
    margin-bottom: 15px; /* BARU: Menambah jarak antar elemen vertikal */
}

/* BARU: Efek hover untuk kartu presentasi */
.presentation-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}

/* BARU: Class untuk highlight teks */
.highlight-text {
    color: #2575fc;
    font-weight: bold;
}
</style>
"""
st.markdown(CSS_STYLE, unsafe_allow_html=True)


# --- Fungsi Lottie ---
@st.cache_data
def load_lottieurl(url: str):
    """
    Mengambil file JSON Lottie dari URL.
    """
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# LOTTIE BARU untuk Halaman Presentasi
LOTTIE_PRESENTATION_URL = "https://assets3.lottiefiles.com/packages/lf20_96bovlqg.json"


# --- Fungsi Load Data ---
@st.cache_data
def load_data():
    """
    Memuat dan memproses dataset dari file CSV.
    """
    try:
        df = pd.read_csv("Social Media Engagement Dataset.csv")
        
        # --- PERBAIKAN PENTING: Normalisasi Rate ---
        df['engagement_rate'] = df['engagement_rate'].apply(lambda x: x / 100 if x > 2 else x)
        df['toxicity_score'] = df['toxicity_score'].apply(lambda x: x / 100 if x > 2 else x)
        # --- AKHIR PERBAIKAN ---

        # 'Explode' hashtags dan keywords
        df_hashtags = df.assign(hashtag=df['hashtags'].str.split(',')).explode('hashtag')
        df_hashtags['hashtag'] = df_hashtags['hashtag'].str.strip().str.lower()
        
        df_keywords = df.assign(keyword=df['keywords'].str.split(',')).explode('keyword')
        df_keywords['keyword'] = df_keywords['keyword'].str.strip().str.lower()
        
        return df, df_hashtags, df_keywords
    except FileNotFoundError:
        st.error("File 'Social Media Engagement Dataset.csv' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama.")
        return None, None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")
        return None, None, None

# --- Fungsi Training Model ---
@st.cache_resource
def train_models(_df):
    """
    Melatih model Regresi dan Klasifikasi.
    """
    # Pra-pemrosesan data untuk model
    _df['keyword_model'] = _df['keywords'].str.split(',').str[0].str.strip().str.lower()
    _df['hashtag_model'] = _df['hashtags'].str.split(',').str[0].str.strip().str.lower()
    
    features = ['day_of_week', 'language', 'platform', 'keyword_model', 'hashtag_model', 'campaign_name']
    targets_reg = ['likes_count', 'shares_count', 'comments_count', 'toxicity_score', 'impressions', 'engagement_rate']
    target_clf = 'emotion_type'

    _df_cleaned = _df.dropna(subset=features + targets_reg + [target_clf])

    X = _df_cleaned[features]
    y_reg = _df_cleaned[targets_reg]
    y_clf = _df_cleaned[target_clf]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features)
        ],
        remainder='passthrough'
    )

    # Model Regresi
    pipeline_reg = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    pipeline_reg.fit(X, y_reg)
    
    # Model Klasifikasi
    pipeline_clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    pipeline_clf.fit(X, y_clf)
    
    unique_values = {col: _df_cleaned[col].unique().tolist() for col in features}
    
    return pipeline_reg, pipeline_clf, unique_values

# --- Fungsi Metrik Saran ---
@st.cache_data
def get_advanced_metrics(_df, _df_keywords):
    """
    Menghitung metrik lanjutan untuk saran yang lebih cerdas.
    """
    metrics = {}
    
    # 1. Platform Metrics (rata-rata per platform)
    metrics['platform'] = _df.groupby('platform').agg(
        avg_engagement=('engagement_rate', 'mean'),
        avg_toxicity=('toxicity_score', 'mean'),
        top_day=('day_of_week', lambda x: x.value_counts().idxmax())
    ).to_dict('index')
    
    # 2. Day Metrics (rata-rata per platform, per hari)
    metrics['day'] = _df.groupby(['platform', 'day_of_week'])['engagement_rate'].mean().to_dict()
    
    # 3. Language Metrics (rata-rata per platform, per bahasa)
    metrics['lang'] = _df.groupby(['platform', 'language'])['engagement_rate'].mean().to_dict()
    
    # 4. Keyword Metrics (rata-rata global per keyword)
    keyword_df_cleaned = _df_keywords[_df_keywords['keyword'].notna()]
    # Perbaikan dari error sebelumnya: langsung gunakan df yang sudah di-explode
    metrics['keyword'] = keyword_df_cleaned.groupby('keyword')['engagement_rate'].mean().to_dict()
    
    # 5. Golden Combo (Kombinasi Emas)
    try:
        golden_combo_df = _df.groupby(['platform', 'day_of_week', 'language'])['engagement_rate'].mean().nlargest(1)
        if not golden_combo_df.empty:
            metrics['golden_combo'] = golden_combo_df.index[0]
            metrics['golden_avg'] = golden_combo_df.values[0]
    except Exception:
        pass # Abaikan jika gagal (misal: data terlalu sedikit)
            
    return metrics

# --- Memuat Data dan Model ---
df, df_hashtags, df_keywords = load_data()

if df is not None:
    pipeline_reg, pipeline_clf, unique_values = train_models(df.copy())
    # Menghitung metrik lanjutan untuk saran
    advanced_metrics = get_advanced_metrics(df, df_keywords)
    
    # Metrik global sebagai fallback
    avg_engagement = df['engagement_rate'].mean()
    avg_toxicity = df['toxicity_score'].mean()
    top_day = df['day_of_week'].value_counts().idxmax()
    
    # URL Gambar Placeholder (untuk contoh di halaman presentasi)
    PLACEHOLDER_IMG_URL = "https://placehold.co/600x300/6a11cb/white?text=Contoh+Gambar+Anda&font=lato"


    # --- ======================== NAVIGASI SIDEBAR (BARU) ======================== ---
    with st.sidebar:
        st.markdown(f"<h3>Analisis Media Sosial</h3>", unsafe_allow_html=True)
        
        selected_page = option_menu(
            menu_title=None,  # Hapus judul menu
            options=["Beranda", "Presentasi", "Analisis Rangking", "Prakiraan"],
            icons=["house-door-fill", "easel2-fill", "bar-chart-line-fill", "robot"],
            menu_icon="cast", 
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "var(--secondary-background-color)"},
                "icon": {"color": "#2575fc", "font-size": "20px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#e0eaff",
                },
                "nav-link-selected": {"background-color": "linear-gradient(90deg, #6a11cb 0%, #2575fc 100%)", "color": "white", "font-weight": "bold"},
            }
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info("Dashboard ini dibuat untuk menganalisis dan memprediksi data engagement media sosial Anda.")


    # --- ======================== HALAMAN BERANDA ======================== ---
    if selected_page == "Beranda":
        
        # --- PERMINTAAN #2: Ganti Lottie dengan Gambar Lokal ---
        col_anim, col_text = st.columns([1, 2])
        
        with col_anim:
            try:
                # Coba muat gambar lokal 'beranda.png'
                st.image(
                    "logo.png",
                    use_container_width=True, # <-- PERBAIKAN (Poin 3): dari use_column_width
                    caption="Visualisasi Analisis Data"
                )
            except FileNotFoundError:
                # Fallback jika gambar tidak ditemukan
                st.info("Letakkan file 'beranda.png' di folder yang sama dengan file .py ini untuk menampilkan gambar kustom di sini.")
                # Anda bisa mengaktifkan Lottie lagi sebagai fallback jika mau
                # lottie_animation = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_s9algjvi.json") 
                # if lottie_animation:
                #     st_lottie(lottie_animation, height=300, key="analytics_fallback")

        
        with col_text:
            st.title("Selamat Datang di Dashboard Analisis Engagement 🚀")
            st.markdown("""
            Aplikasi ini membantu Anda memahami dan memprediksi engagement media sosial. 
            Gunakan **AI** kami untuk mendapatkan prakiraan performa konten atau jelajahi 
            data historis Anda untuk menemukan tren teratas.
            
            Pilih salah satu menu di **Sidebar** untuk memulai:
            - **Presentasi:** Lihat penjelasan visual proyek ini.
            - **Analisis Rangking:** Jelajahi performa konten historis.
            - **Prakiraan:** Dapatkan prediksi AI untuk konten baru.
            """)
            
    # --- ======================== HALAMAN PRESENTASI (ROMBAK TOTAL) ======================== ---
    elif selected_page == "Presentasi":
        st.title("💡 Presentasi Proyek: Analisis Engagement")
        
        # --- PERBAIKAN: Menghapus st.columns ---
        # Animasi Lottie sekarang akan menjadi full-width
        lottie_pres = load_lottieurl(LOTTIE_PRESENTATION_URL)
        if lottie_pres:
            st_lottie(lottie_pres, height=300)
        
        # Kartu "Selamat Datang" sekarang akan menjadi full-width
        st.markdown("""
        <div class="presentation-card" style="text-align: center;"> <!-- PERBAIKAN (Poin 1): text-align: center -->
        <h3>Selamat datang di presentasi proyek ini.</h3>
        Aplikasi ini dirancang sebagai <span class="highlight-text">Alat Bantu Pengambilan Keputusan (Decision Support Tool)</span> untuk strategi konten media sosial Anda.
        <br><br>
        <strong>Tujuannya adalah mengubah data mentah menjadi wawasan yang dapat ditindaklanjuti.</strong>
        </div>
        """, unsafe_allow_html=True)
        # --- AKHIR PERBAIKAN ---
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # --- PERBAIKAN (Poin 2): Penjelasan Dataset Lebih Rinci ---
        st.subheader("1. Dataset: Bahan Bakar Kita")
        st.markdown("Aplikasi ini ditenagai oleh dataset `Social Media Engagement Dataset.csv`. Mari kita bedah data ini:")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Postingan", f"{len(df):,}")
        c2.metric("Platform Teratas", df['platform'].mode()[0])
        c3.metric("Total Bahasa", df['language'].nunique())
        c4.metric("Hari Teraktif", df['day_of_week'].mode()[0])
        
        st.markdown("**Pratinjau Data Mentah:**")
        st.dataframe(df.head())

        # PERBAIKAN (Poin 2): Penjelasan kolom yang lebih rinci
        st.markdown("**Penjelasan Lengkap Seluruh Kolom Dataset:**")
        
        # Buat daftar deskripsi kolom
        column_descriptions = {
            "day_of_week": "Hari (Senin, Selasa, dll.) saat konten diposting.",
            "platform": "Platform media sosial (Instagram, Twitter, dll.) tempat konten diposting.",
            "location": "Lokasi geografis (biasanya kota/negara) yang terkait dengan postingan.",
            "language": "Kode bahasa (pt, ru, en, dll.) dari teks konten.",
            "text_content": "Teks mentah aktual dari postingan tersebut.",
            "hashtags": "Daftar hashtag (dipisahkan koma) yang digunakan dalam postingan.",
            "keywords": "Daftar keyword (dipisahkan koma) yang diekstrak dari teks.",
            "topic_category": "Kategori topik yang dibahas (Produk, Harga, dll.).",
            "sentiment_score": "Skor numerik sentimen (-1 Negatif hingga +1 Positif).",
            "sentiment_label": "Label sentimen (Positif, Negatif, Netral).",
            "emotion_type": "Emosi spesifik yang terdeteksi (Senang, Marah, Bingung, dll.).",
            "toxicity_score": "Skor numerik (0-1) yang menunjukkan seberapa toksik/negatif konten tersebut.",
            "likes_count": "Jumlah 'Likes' yang diterima postingan.",
            "shares_count": "Jumlah 'Shares' yang diterima postingan.",
            "comments_count": "Jumlah 'Comments' yang diterima postingan.",
            "impressions": "Jumlah total berapa kali postingan ditampilkan kepada pengguna.",
            "engagement_rate": "Metrik kunci (biasanya (Likes+Comments+Shares)/Impressions) dalam format desimal (0-1).",
            "brand_name": "Nama brand (Google, Nike, dll.) yang terkait dengan postingan.",
            "product_name": "Nama produk spesifik (Chromebook, Epic React, dll.) yang disebutkan.",
            "campaign_name": "Nama kampanye pemasaran (BlackFriday, PowerRelease, dll.) yang terkait."
        }

        # Tampilkan dalam dua kolom agar lebih rapi
        col1_desc, col2_desc = st.columns(2)
        
        # Membagi daftar kolom
        all_columns = list(column_descriptions.items())
        mid_point = len(all_columns) // 2 + (len(all_columns) % 2)
        
        with col1_desc:
            for col, desc in all_columns[:mid_point]:
                st.markdown(f"- **{col}**: {desc}")

        with col2_desc:
            for col, desc in all_columns[mid_point:]:
                st.markdown(f"- **{col}**: {desc}")
        
        st.markdown("**Ringkasan Statistik Data Numerik:**")
        st.dataframe(df.describe())
        
        st.markdown("**Distribusi Platform:**")
        platform_dist = df['platform'].value_counts().reset_index()
        platform_dist.columns = ['Platform', 'Jumlah Postingan']
        fig_pie = px.pie(platform_dist, 
                         names='Platform', 
                         values='Jumlah Postingan', 
                         title='Distribusi Postingan di Seluruh Platform',
                         hole=0.3)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # --- PERMINTAAN #3: Penjelasan Misi Lebih Rinci ---
        st.subheader("2. Misi & Tujuan")
        st.markdown("Berdasarkan permintaan awal Anda, misi aplikasi ini terbagi menjadi dua tujuan utama:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="presentation-card" style="background-color: rgba(37, 117, 252, 0.1);">
            <h4>Menganalisa Data Historis (Melihat ke Belakang)</h4>
            <p><strong>Permintaan:</strong> "Saya ingin menganalisa... rangking... top three day... top engagement... top likes... top language... top hashtag... top keyword."</p>
            <p><strong>Tujuan:</strong> Kita perlu memahami apa yang <strong class="highlight-text">telah berhasil</strong> di masa lalu. Pola apa yang muncul? Platform, hari, atau keyword mana yang paling menguntungkan? Ini adalah dasar dari semua strategi.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="presentation-card" style="background-color: rgba(106, 27, 203, 0.1);">
            <h4>Memprediksi Performa Masa Depan (Melihat ke Depan)</h4>
            <p><strong>Permintaan:</strong> "Saya ingin... Prakiraan... prediksi engagement dibuat berdasarkan: Hari, Bahasa, Platform, Keyword, Hashtag, dan Campaign."</p>
            <p><strong>Tujuan:</strong> Menganalisa saja tidak cukup. Kita perlu menggunakan data historis untuk <strong class="highlight-text">melatih model AI (Machine Learning)</strong> yang dapat memprediksi performa konten yang <strong>belum ada</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # --- PERMINTAAN #3: Penjelasan Hasil (Menu) Lebih Rinci ---
        st.subheader("3. Hasil Akhir: Penjelasan Fitur Aplikasi")
        st.markdown("Untuk memenuhi kedua misi tersebut, aplikasi ini dibagi menjadi beberapa menu fungsional:")

        st.markdown("""
        <div class="presentation-card">
        <h4>Beranda</h4>
        <p>Halaman ini adalah pintu gerbang utama Anda. Ini memberikan sambutan dan navigasi visual ke fitur-fitur utama aplikasi, serta menampilkan visual utama (gambar yang Anda letakkan).</p>
        </div>
        
        <div class="presentation-card">
        <h4>Analisis Rangking</h4>
        <p>Ini adalah jawaban untuk misi 'Menganalisa'. Halaman ini berisi 6 tab terpisah, masing-masing dengan <strong>visualisasi diagram batang</strong> untuk:
        <ul>
            <li>Hari Upload Terpopuler</li>
            <li>Top 10 Postingan (Engagement Rate)</li>
            <li>Top 10 Postingan (Likes)</li>
            <li>Bahasa Paling Sering Digunakan</li>
            <li>Top 10 Hashtag</li>
            <li>Top 10 Keyword</li>
        </ul>
        Setiap diagram dilengkapi dengan <strong>kesimpulan dan saran</strong> otomatis berdasarkan data yang ditampilkan.
        </p>
        </div>
        
        <div class="presentation-card">
        <h4>Prakiraan</h4>
        <p>Ini adalah jawaban untuk misi 'Memprediksi'. Halaman ini adalah alat AI interaktif Anda:
        <ol>
            <li>Anda memasukkan 6 parameter konten baru (Hari, Bahasa, Platform, dll.).</li>
            <li>Model AI <i>(Random Forest)</i> akan memprediksi 7 metrik performa secara instan (Likes, Shares, Comments, Engagement Rate, dll.).</li>
            <li>Sistem kemudian memberikan <strong>Analisis & Saran Tingkat Lanjut</strong> yang membandingkan prediksi Anda dengan data historis, mengidentifikasi tujuan konten, dan mencari "titik terlemah" untuk dioptimalkan.</li>
        </ol>
        </p>
        </div>
        
        <div class="presentation-card">
        <h4>Presentasi</h4>
        <p>Halaman yang sedang Anda lihat sekarang. Ini berfungsi sebagai dokumentasi dan penjelasan proyek secara keseluruhan, mulai dari dataset, tujuan, hingga hasil akhir.</p>
        </div>
        """, unsafe_allow_html=True)


        st.markdown("<hr>", unsafe_allow_html=True)
            
    # --- ======================== HALAMAN ANALISIS RANGKING ======================== ---
    elif selected_page == "Analisis Rangking":
        st.title("🏆 Analisis Rangking Engagement")
        st.markdown("Berikut adalah rangking teratas berdasarkan data Anda. Semuanya dalam format diagram batang **vertikal** untuk perbandingan visual.")

        # --- PERBAIKAN: Menghapus emoji dari nama tab untuk menghindari SyntaxError ---
        tab_names = [
            "Hari Upload", 
            "Top 10 Engagement Rate", 
            "Top 10 Likes",
            "Bahasa", 
            "Top 10 Hashtag", 
            "Top 10 Keyword"
        ]
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_names)

        # SEMUA TAB MENGGUNAKAN DIAGRAM BATANG VERTIKAL
        with tab1: # Hari Upload
            st.subheader("Popularitas Hari untuk Upload")
            day_counts = df['day_of_week'].value_counts().reset_index()
            day_counts.columns = ['Hari', 'Jumlah Post']
            day_counts = day_counts.sort_values(by="Jumlah Post", ascending=False)
            fig = px.bar(day_counts, 
                         x='Hari', y='Jumlah Post',  # <-- Vertikal
                         title="Jumlah Postingan Berdasarkan Hari",
                         color='Jumlah Post', text_auto=True,
                         color_continuous_scale='Viridis', # <-- PERMINTAAN #1
                         labels={'Hari': 'Hari dalam Seminggu', 'Jumlah Post': 'Jumlah Postingan'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # KESIMPULAN (DISEMPURNAKAN)
            if not day_counts.empty:
                top_day_data = day_counts.iloc[0]
                bottom_day = day_counts.iloc[-1]['Hari']
                st.info(
                    f"💡 **Analisis Singkat:** Hari **{top_day_data['Hari']}** adalah hari tersibuk ({top_day_data['Jumlah Post']} postingan). "
                    f"Ini berarti audiens Anda paling aktif, TAPI juga **persaingan tertinggi**. "
                    f"**Saran:** Jika performa Anda rendah di hari ini, coba posting di hari yang lebih 'tenang' (seperti **{bottom_day}**) untuk melihat apakah konten Anda lebih menonjol.",
                    icon="💡"
                )

        with tab2: # Top 10 Engagement Rate
            st.subheader("Top 10 Postingan dengan Engagement Rate Tertinggi")
            top_eng = df.nlargest(10, 'engagement_rate')[['text_content', 'engagement_rate', 'platform']]
            top_eng['text_display'] = top_eng['text_content'].str.slice(0, 60) + '...'
            top_eng = top_eng.sort_values(by="engagement_rate", ascending=False) # Descending untuk vertikal
            fig = px.bar(top_eng,
                         x='text_display', y='engagement_rate',  # <-- Vertikal
                         title="Top 10 Postingan: Engagement Rate",
                         color='engagement_rate', color_continuous_scale='Plotly3', # <-- PERMINTAAN #1
                         labels={'engagement_rate': 'Engagement Rate', 'text_display': 'Judul Konten'},
                         hover_data={'text_content': True, 'platform': True, 'engagement_rate': ':.2%'} 
                         )
            fig.update_layout(yaxis_tickformat='.1%') 
            st.plotly_chart(fig, use_container_width=True)
            
            # KESIMPULAN (DISEMPURNAKAN)
            if not top_eng.empty:
                top_eng_post_data = top_eng.iloc[0] 
                st.info(
                    f"💡 **Analisis Singkat:** Postingan di **{top_eng_post_data['platform']}** dengan rate **{top_eng_post_data['engagement_rate']:.2%}** adalah *benchmark* (standar emas) Anda. "
                    f"**Saran:** Pelajari **format**, **nada bicara (tone)**, dan **topik** dari postingan ini ({top_eng_post_data['text_content'][:40]}...). Apakah itu video? Pertanyaan? Gunakan ini sebagai template untuk konten berkinerja tinggi.",
                    icon="💡"
                )

        with tab3: # Top 10 Likes
            st.subheader("Top 10 Postingan dengan Likes Terbanyak")
            top_likes = df.nlargest(10, 'likes_count')[['text_content', 'likes_count', 'platform']]
            top_likes['text_display'] = top_likes['text_content'].str.slice(0, 60) + '...'
            top_likes = top_likes.sort_values(by="likes_count", ascending=False) # Descending untuk vertikal
            fig = px.bar(top_likes,
                         x='text_display', y='likes_count',  # <-- Vertikal
                         title="Top 10 Postingan: Likes",
                         color='likes_count', text_auto=True, color_continuous_scale='OrRd', # <-- PERMINTAAN #1
                         labels={'likes_count': 'Jumlah Likes', 'text_display': 'Judul Konten'},
                         hover_data={'text_content': True, 'platform': True}
                         )
            st.plotly_chart(fig, use_container_width=True)
            
            # KESIMPULAN (DISEMPURNAKAN)
            if not top_likes.empty:
                top_like_post_data = top_likes.iloc[0]
                st.info(
                    f"💡 **Analisis Singkat:** Postingan di **{top_like_post_data['platform']}** ({int(top_like_post_data['likes_count']):,} likes) adalah 'juara viralitas' Anda. "
                    f"**Saran:** Konten seperti ini sangat bagus untuk **Brand Awareness**. Gunakan format ({top_like_post_data['text_content'][:40]}...) untuk kampanye yang bertujuan menjangkau audiens baru yang belum mengenal Anda.",
                    icon="💡"
                )

        with tab4: # Bahasa
            st.subheader("Popularitas Bahasa yang Digunakan")
            lang_counts = df['language'].value_counts().reset_index()
            lang_counts.columns = ['Bahasa', 'Jumlah']
            lang_counts['Bahasa_Display'] = lang_counts['Bahasa'].map(LANG_MAP).fillna(lang_counts['Bahasa'])
            lang_counts = lang_counts.sort_values(by="Jumlah", ascending=False) # Descending untuk vertikal
            fig = px.bar(lang_counts, 
                         x='Bahasa_Display', y='Jumlah',  # <-- Vertikal
                         title="Jumlah Postingan Berdasarkan Bahasa",
                         color='Jumlah', text_auto=True,
                         color_continuous_scale='Plasma') # <-- PERMINTAAN #1
            fig.update_layout(xaxis_title="Bahasa")
            st.plotly_chart(fig, use_container_width=True)
            
            # KESIMPULAN (DISEMPURNAKAN)
            if not lang_counts.empty:
                top_lang_data = lang_counts.iloc[0]
                second_lang = lang_counts.iloc[1]['Bahasa_Display']
                st.info(
                    f"💡 **Analisis Singkat:** Bahasa **{top_lang_data['Bahasa_Display']}** adalah audiens utama Anda ({top_lang_data['Jumlah']} postingan). "
                    f"**Saran:** Pertimbangkan untuk membuat konten spesifik atau menerjemahkan konten unggulan ke dalam bahasa kedua terpopuler Anda (**{second_lang}**) untuk memperluas jangkauan ke segmen baru.",
                    icon="💡"
                )

        with tab5: # Top 10 Hashtag
            st.subheader("Top 10 Hashtag Paling Populer")
            hash_counts = df_hashtags['hashtag'].value_counts().nlargest(10).reset_index()
            hash_counts.columns = ['Hashtag', 'Jumlah']
            hash_counts = hash_counts.dropna(subset=['Hashtag']) 
            hash_counts = hash_counts.sort_values('Jumlah', ascending=False) # Descending untuk vertikal
            fig = px.bar(hash_counts, 
                         x='Hashtag', y='Jumlah',  # <-- Vertikal
                         title="Top 10 Hashtag",
                         color='Jumlah', text_auto=True,
                         color_continuous_scale='Turbo') # <-- PERMINTAAN #1
            st.plotly_chart(fig, use_container_width=True)
            
            # KESIMPULAN (DISEMPURNAKAN)
            if not hash_counts.empty:
                top_hash_data = hash_counts.iloc[0]
                second_hash = hash_counts.iloc[1]['Hashtag']
                st.info(
                    f"💡 **Analisis Singkat:** Hashtag **#{top_hash_data['Hashtag']}** adalah tema sentral Anda ({top_hash_data['Jumlah']} kali). "
                    f"**Saran:** Untuk menghindari kejenuhan, kombinasikan hashtag utama ini dengan hashtag *niche* atau *trending* (seperti **#{second_hash}**) untuk menjangkau audiens yang lebih spesifik namun tetap relevan.",
                    icon="💡"
                )

        with tab6: # Top 10 Keyword
            st.subheader("Top 10 Keyword Paling Populer")
            key_counts = df_keywords['keyword'].value_counts().nlargest(10).reset_index()
            key_counts.columns = ['Keyword', 'Jumlah']
            key_counts = key_counts.dropna(subset=['Keyword']) 
            key_counts = key_counts.sort_values('Jumlah', ascending=False) # Descending untuk vertikal
            fig = px.bar(key_counts, 
                         x='Keyword', y='Jumlah',  # <-- Vertikal
                         title="Top 10 Keyword",
                         color='Jumlah', text_auto=True,
                         color_continuous_scale='Electric') # <-- PERMINTAAN #1
            st.plotly_chart(fig, use_container_width=True)
            
            # KESIMPULAN (DISEMPURNAKAN)
            if not key_counts.empty:
                top_key_data = key_counts.iloc[0]
                st.info(
                    f"💡 **Analisis Singkat:** Keyword **'{top_key_data['Keyword']}'** adalah fokus utama dari strategi konten Anda ({top_key_data['Jumlah']} kali). "
                    f"**Saran:** Gunakan halaman 'Prakiraan' untuk menguji keyword ini di platform yang berbeda. Sangat mungkin keyword ini sangat laku di **Instagram**, tetapi kinerjanya biasa saja di **Twitter** (atau sebaliknya).",
                    icon="💡"
                )

    # --- ======================== HALAMAN PRAKIRAAN ======================== ---
    elif selected_page == "Prakiraan":
        st.title("🔮 Prakiraan Engagement Konten")
        st.markdown("Masukkan detail konten yang akan Anda upload untuk mendapatkan prakiraan engagement.")

        with st.form("prediction_form"):
            st.subheader("Form Input Konten")
            
            # --- PERBAIKAN: Menambahkan 'placeholder_text' ---
            placeholder_text = "Pilih Opsi..." 
            
            col1, col2, col3 = st.columns(3)
            with col1:
                # --- PERBAIKAN: Menambahkan placeholder dan index=0 ---
                day_options = [placeholder_text] + sorted(unique_values['day_of_week'])
                day = st.selectbox("Hari Upload:", day_options, index=0)
                
                lang_codes_from_data = sorted(unique_values['language'])
                lang_display_options = [placeholder_text] + sorted([LANG_MAP.get(code, code) for code in lang_codes_from_data if code in LANG_MAP])
                lang_display_selection = st.selectbox("Bahasa:", lang_display_options, index=0)
                
            with col2:
                # --- PERBAIKAN: Menambahkan placeholder dan index=0 ---
                platform_options = [placeholder_text] + sorted(unique_values['platform'])
                platform = st.selectbox("Platform:", platform_options, index=0)
                
                campaign_options = [placeholder_text] + sorted(unique_values['campaign_name'])
                campaign = st.selectbox("Campaign:", campaign_options, index=0)
            with col3:
                # --- PERBAIKAN: Menambahkan placeholder dan index=0 ---
                keyword_options = [placeholder_text] + sorted([k for k in unique_values['keyword_model'] if pd.notna(k)])
                keyword = st.selectbox("Keyword Utama:", keyword_options, index=0)
                
                hashtag_options = [placeholder_text] + sorted([h for h in unique_values['hashtag_model'] if pd.notna(h)])
                hashtag = st.selectbox("Hashtag Utama:", hashtag_options, index=0)

            submit_button = st.form_submit_button("Dapatkan Prakiraan 🚀", type="primary")

        if submit_button:
            # --- PERBAIKAN: Menambahkan blok validasi ---
            if (day == placeholder_text or 
                lang_display_selection == placeholder_text or 
                platform == placeholder_text or 
                campaign == placeholder_text or 
                keyword == placeholder_text or 
                hashtag == placeholder_text):
                
                st.warning("⚠️ Mohon lengkapi semua 6 pilihan untuk mendapatkan prakiraan.")
            
            else: 
                # --- PERBAIKAN: Memastikan sisa kode di-indentasi (digeser ke kanan) di dalam 'else' ---
                lang_code = REVERSE_LANG_MAP.get(lang_display_selection, lang_display_selection)
                input_data = pd.DataFrame({
                    'day_of_week': [day],
                    'language': [lang_code], 
                    'platform': [platform],
                    'keyword_model': [keyword],
                    'hashtag_model': [hashtag],
                    'campaign_name': [campaign]
                }) # <-- Ini adalah ')' yang hilang dari error Anda
                
                with st.spinner("Menganalisis & Memproses Prakiraan..."):
                    pred_reg = pipeline_reg.predict(input_data)[0]
                    pred_clf = pipeline_clf.predict(input_data)[0]
                    
                    results_reg = {
                        'Likes': (pred_reg[0], "❤️"),
                        'Shares': (pred_reg[1], "🔁"),
                        'Comments': (pred_reg[2], "💬"),
                        'Impressions': (pred_reg[4], "👁️"),
                        'Toxicity Rate': (pred_reg[3], "☣️"),
                        'Engagement Rate': (pred_reg[5], "🔥")
                    }
                    
                    emotion_emoji_map = {
                        'Positive': '😄', 'Negative': '😠', 'Neutral': '😐',
                        'Happy': '😊', 'Sad': '😢', 'Angry': '😠', 'Excited': '🤩',
                        'Confused': '🤔', 'Surprised': '😲', 'Fear': '😨'
                    }
                    emotion_emoji = emotion_emoji_map.get(pred_clf, "❓")

                    st.subheader("🎉 Hasil Prakiraan:")
                    cols = st.columns(4)
                    cols[0].metric(label=f"Tipe Emosi", value=f"{emotion_emoji} {pred_clf}")
                    
                    i = 1 
                    for key, (value, emoji) in results_reg.items():
                        col = cols[i % 4]
                        if key in ['Toxicity Rate', 'Engagement Rate']:
                            formatted_val = f"{value * 100:.2f}%"
                            col.metric(label=f"{emoji} {key}", value=formatted_val)
                        else:
                            formatted_val = f"{int(value):,}"
                            col.metric(label=f"{emoji} {key}", value=formatted_val)
                        i += 1
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                    
                    # --- BAGIAN BARU: KESIMPULAN & SARAN (LOGIKA SANGAT DISEMPURNAKAN) ---
                    st.subheader("💡 Analisis & Saran (Tingkat Lanjut)")
                    
                    # Mendapatkan data prediksi
                    engagement_pred = pred_reg[5]
                    toxicity_pred = pred_reg[3]
                    impressions_pred = pred_reg[4]
                    shares_pred = pred_reg[1]
                    comments_pred = pred_reg[2]
                    suggestions = []

                    # --- Mendapatkan Metrik Kontekstual ---
                    platform_metrics_dict = advanced_metrics.get('platform', {})
                    platform_metrics = platform_metrics_dict.get(platform, {})
                    
                    avg_eng_platform = platform_metrics.get('avg_engagement', avg_engagement)
                    avg_tox_platform = platform_metrics.get('avg_toxicity', avg_toxicity)
                    top_day_platform = platform_metrics.get('top_day', top_day)

                    avg_eng_day_choice = advanced_metrics.get('day', {}).get((platform, day), 0)
                    avg_eng_top_day = advanced_metrics.get('day', {}).get((platform, top_day_platform), 0)
                    
                    avg_eng_keyword_choice = advanced_metrics.get('keyword', {}).get(keyword, 0)
                    
                    # --- (BARU) Analisis Tujuan Konten / Persona ---
                    if toxicity_pred > 0.6 and pred_clf in ['Angry', 'Negative']:
                        suggestions.append(f"🎯 **Tujuan Teridentifikasi: Konten Provokatif/Risiko Tinggi.** "
                                           f"Emosi '{pred_clf}' dan Toksisitas {toxicity_pred:.2%} sangat tinggi. Ini akan memicu reaksi, tapi mungkin negatif. Gunakan HANYA jika ini disengaja (misal: debat panas, kritik). Risiko *bad buzz* tinggi.")
                    elif engagement_pred > avg_eng_platform and shares_pred > (df['shares_count'].mean() * 1.2):
                        suggestions.append(f"🎯 **Tujuan Teridentifikasi: Viralitas & Jangkauan.** "
                                           f"Prediksi 'Shares' dan 'Engagement' Anda tinggi. Konten ini berpotensi besar untuk menjangkau audiens baru (viral). Sangat baik untuk kampanye *awareness*.")
                    elif engagement_pred > avg_eng_platform and comments_pred > (df['comments_count'].mean() * 1.2):
                        suggestions.append(f"🎯 **Tujuan Teridentifikasi: Membangun Komunitas.** "
                                           f"Prediksi 'Comments' tinggi menunjukkan konten ini memicu diskusi. Sangat baik untuk membangun komunitas dan mendapatkan *feedback* langsung dari audiens setia Anda.")
                    elif impressions_pred > (df['impressions'].mean() * 1.5) and engagement_pred < avg_eng_platform:
                        suggestions.append(f"🎯 **Tujuan Teridentifikasi: Jangkauan Luas (Awareness).** "
                                           f"**Peringatan:** Konten Anda diprediksi akan **dilihat** banyak orang (Impresi tinggi), tapi **tidak menarik** (Engagement rendah). Ini disebut 'Scroll-by'. **Saran:** Perbaiki *hook* visual atau *Call-to-Action* (CTA) Anda agar lebih memikat.")
                    else:
                        suggestions.append(f"🎯 **Tujuan Teridentifikasi: Performa Standar/Brand-Building.** "
                                           f"Konten ini diprediksi akan berjalan sesuai standar. Ini adalah konten 'aman' yang baik untuk menjaga konsistensi brand Anda.")


                    # --- Analisis Performa Engagement (Sudah ada, tetap relevan) ---
                    if engagement_pred > avg_eng_platform * 1.1:
                        suggestions.append(f"📈 **Performa Unggul:** Prediksi engagement Anda ({engagement_pred:.2%}) **jauh di atas rata-rata** untuk **{platform}** (rata-rata: {avg_eng_platform:.2%}). Kombinasi Anda terlihat sangat kuat!")
                    elif engagement_pred < avg_eng_platform * 0.9:
                        suggestions.append(f"📉 **Performa Kurang:** Prediksi engagement Anda ({engagement_pred:.2%}) **di bawah rata-rata** untuk **{platform}** (rata-rata: {avg_eng_platform:.2%}). Mari kita lihat mengapa:")
                    else:
                        suggestions.append(f"📊 **Performa Rata-rata:** Prediksi engagement Anda ({engagement_pred:.2%}) **sesuai rata-rata** untuk **{platform}** (rata-rata: {avg_eng_platform:.2%}). Ada ruang untuk optimalisasi.")

                    # --- Analisis "Weakest Link" (Hari) (Sudah ada, tetap relevan) ---
                    if avg_eng_day_choice > 0 and avg_eng_top_day > 0 and avg_eng_day_choice < avg_eng_top_day:
                        suggestions.append(
                            f"  - **Peluang Hari:** Anda memilih **{day}**, yang di **{platform}** memiliki rata-rata engagement ({avg_eng_day_choice:.2%}). "
                            f"Hari terkuat di **{platform}** adalah **{top_day_platform}** (rata-rata: {avg_eng_top_day:.2%}). "
                            f"**Saran:** Jika topiknya fleksibel, pertimbangkan beralih ke **{top_day_platform}** untuk potensi peningkatan."
                        )
                    
                    # --- Analisis Keyword (Sudah ada, tetap relevan) ---
                    if avg_eng_keyword_choice > 0 and avg_eng_keyword_choice > avg_engagement:
                        suggestions.append(f"  - **Pilihan Keyword Baik:** Keyword Anda ('{keyword}') adalah pilihan kuat! Secara historis, keyword ini memiliki rata-rata engagement {avg_eng_keyword_choice:.2%}.")
                    elif avg_eng_keyword_choice > 0:
                        suggestions.append(f"  - **Peringatan Keyword:** Keyword Anda ('{keyword}') secara historis memiliki engagement ({avg_eng_keyword_choice:.2%}) di bawah rata-rata global. Pastikan konten Anda sangat menonjol untuk mengatasi ini.")
                    
                    # --- Analisis Toksisitas & Emosi (Sintesis) (Sudah ada, tetap relevan) ---
                    if pred_clf in ['Negative', 'Angry', 'Sad', 'Fear'] and toxicity_pred < 0.6: # Filter out high-risk
                        suggestions.append(
                            f"  - **Analisis Emosi:** Anda mendapat prediksi emosi **{pred_clf}**. "
                            f"Jika ini *sengaja* (misal: konten sedih/serius), ini wajar. "
                            f"Jika *tidak disengaja*, emosi negatif ini bisa menjadi alasan utama prediksi engagement Anda (jika rendah). Pertimbangkan melembutkan bahasa/keyword."
                        )
                    elif toxicity_pred > avg_tox_platform and toxicity_pred < 0.6: # Filter out high-risk
                        suggestions.append(f"  - **Peringatan Toksisitas:** Emosi Anda **{pred_clf}** (positif/netral), tetapi toksisitas Anda ({toxicity_pred:.2%}) masih **di atas rata-rata** {platform} ({avg_tox_platform:.2%}). "
                                           f"Ini mungkin karena keyword/hashtag ('{keyword}', '{hashtag}') yang bisa disalahartikan. Cek ulang.")
                    
                    # --- Golden Combo Insight (Sudah ada, tetap relevan) ---
                    if 'golden_combo' in advanced_metrics:
                        g_plat, g_day, g_lang_code = advanced_metrics['golden_combo']
                        g_lang_display = LANG_MAP.get(g_lang_code, g_lang_code)
                        g_avg = advanced_metrics['golden_avg']
                        suggestions.append(f"  - **Insight Tambahan:** Hanya sebagai info, 'kombinasi emas' di data Anda (engagement tertinggi) adalah: **{g_plat}** + **{g_day}** + **{g_lang_display}**, dengan rata-rata engagement {g_avg:.2%}.")
                    

                    # Tampilkan semua saran
                    if suggestions:
                        st.info("Berdasarkan data historis Anda:", icon="ℹ️")
                        for suggestion in suggestions:
                            st.markdown(f"- {suggestion}")
                    
                    st.info("ℹ️ **Disclaimer:** Prakiraan dan saran ini dibuat berdasarkan model Machine Learning dari data historis pada website Kaggle. Hasil data ini dibuat pada tahun 2025.")

else:
    st.error("Gagal memuat data. Aplikasi tidak dapat dijalankan.")
