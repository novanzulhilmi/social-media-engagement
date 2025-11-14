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

# URL untuk animasi Lottie
LOTTIE_URL = "https://assets9.lottiefiles.com/packages/lf20_s9algjvi.json"
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
        
        lottie_animation = load_lottieurl(LOTTIE_URL)
        col_anim, col_text = st.columns([1, 2])
        
        with col_anim:
            if lottie_animation:
                st_lottie(
                    lottie_animation,
                    speed=1,
                    reverse=False,
                    loop=True,
                    quality="high",
                    height=300,
                    width=300,
                    key="analytics_animation",
                )
        
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
            
    # --- ======================== HALAMAN PRESENTASI (BARU) ======================== ---
    elif selected_page == "Presentasi":
        st.title("💡 Presentasi Proyek: Analisis Engagement")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            lottie_pres = load_lottieurl(LOTTIE_PRESENTATION_URL)
            if lottie_pres:
                st_lottie(lottie_pres, height=300)
        with col2:
            st.markdown("""
            <div classclass="presentation-card">
            Selamat datang di presentasi proyek ini. 
            Aplikasi ini dirancang sebagai **Alat Bantu Pengambilan Keputusan (Decision Support Tool)** untuk strategi konten media sosial Anda.
            
            **Tujuannya adalah mengubah data mentah menjadi wawasan yang dapat ditindaklanjuti.**
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.subheader("1. Dataset: Bahan Bakar Kita")
        st.markdown("Visualisasi data mentah yang kita gunakan:")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Postingan", f"{len(df):,}")
        c2.metric("Platform Teratas", df['platform'].mode()[0])
        c3.metric("Total Bahasa", df['language'].nunique())
        c4.metric("Hari Teraktif", df['day_of_week'].mode()[0])
        
        st.subheader("2. Misi & Tujuan")
        st.markdown("Berdasarkan permintaan Anda, misi aplikasi ini terbagi menjadi dua bagian:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="presentation-card" style="background-color: rgba(37, 117, 252, 0.1);">
            <h4>📊 Analisis Historis (Rangking)</h4>
            <p><strong>Permintaan:</strong> "Saya ingin menampilkan rangking berdasarkan dataset... terpisah berdasarkan masing-masing rangkingnya."</p>
            <p><strong>Solusi:</strong> Halaman 'Analisis Rangking' dengan 6 tab visual untuk melihat data teratas (Hari, Likes, Engagement, Bahasa, dll.) secara instan.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="presentation-card" style="background-color: rgba(106, 27, 203, 0.1);">
            <h4>🔮 Prakiraan AI (Prediksi)</h4>
            <p><strong>Permintaan:</strong> "Tambahkan fitur untuk Prakiraan... prediksi engagement dibuat berdasarkan: Hari, Bahasa, Platform, Keyword, dll."</p>
            <p><strong>Solusi:</strong> Halaman 'Prakiraan' yang menggunakan <strong>Random Forest Model</strong> untuk memprediksi 7 metrik berbeda dan memberikan saran cerdas.</p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("3. Hasil Akhir & Contoh Visual")
        st.markdown("Hasilnya adalah aplikasi interaktif ini. Berikut adalah contoh bagaimana Anda bisa menampilkan gambar dengan lebar maksimal (responsif) di Streamlit, sesuai permintaan Anda:")
        
        # Contoh Kode untuk Menampilkan Gambar Responsif
        st.markdown(f"""
        <p style="text-align: center;">Ini adalah gambar yang diatur dengan <code>max-width: 100%</code>.</p>
        <img src='logo.png' class='responsive-image' alt='Contoh Gambar'>
        """, unsafe_allow_html=True)


            
    # --- ======================== HALAMAN ANALISIS RANGKING ======================== ---
    elif selected_page == "Analisis Rangking":
        st.title("🏆 Analisis Rangking Engagement")
        st.markdown("Berikut adalah rangking teratas berdasarkan data Anda. Semuanya dalam format diagram batang **vertikal** untuk perbandingan visual.")

        tab_names = [
            "☀️ Hari Upload", 
            "🔥 Top 10 Engagement Rate", 
            "❤️ Top 10 Likes",
            "🌍 Bahasa", 
            "🏷️ Top 10 Hashtag", 
            "🔑 Top 10 Keyword"
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
                         color_continuous_scale='Blues',
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
                    f"**Saran:** Jika performa Anda rendah di hari ini, coba posting di hari yang lebih 'tenang' (seperti **{bottom_day}**) untuk melihat apakah konten Anda lebih menonjol."
                )

        with tab2: # Top 10 Engagement Rate
            st.subheader("Top 10 Postingan dengan Engagement Rate Tertinggi")
            top_eng = df.nlargest(10, 'engagement_rate')[['text_content', 'engagement_rate', 'platform']]
            top_eng['text_display'] = top_eng['text_content'].str.slice(0, 60) + '...'
            top_eng = top_eng.sort_values(by="engagement_rate", ascending=False) # Descending untuk vertikal
            fig = px.bar(top_eng,
                         x='text_display', y='engagement_rate',  # <-- Vertikal
                         title="Top 10 Postingan: Engagement Rate",
                         color='engagement_rate', color_continuous_scale='Greens',
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
                    f"**Saran:** Pelajari **format**, **nada bicara (tone)**, dan **topik** dari postingan ini ({top_eng_post_data['text_content'][:40]}...). Apakah itu video? Pertanyaan? Gunakan ini sebagai template untuk konten berkinerja tinggi."
                )

        with tab3: # Top 10 Likes
            st.subheader("Top 10 Postingan dengan Likes Terbanyak")
            top_likes = df.nlargest(10, 'likes_count')[['text_content', 'likes_count', 'platform']]
            top_likes['text_display'] = top_likes['text_content'].str.slice(0, 60) + '...'
            top_likes = top_likes.sort_values(by="likes_count", ascending=False) # Descending untuk vertikal
            fig = px.bar(top_likes,
                         x='text_display', y='likes_count',  # <-- Vertikal
                         title="Top 10 Postingan: Likes",
                         color='likes_count', text_auto=True, color_continuous_scale='Reds',
                         labels={'likes_count': 'Jumlah Likes', 'text_display': 'Judul Konten'},
                         hover_data={'text_content': True, 'platform': True}
                         )
            st.plotly_chart(fig, use_container_width=True)
            
            # KESIMPULAN (DISEMPURNAKAN)
            if not top_likes.empty:
                top_like_post_data = top_likes.iloc[0]
                st.info(
                    f"💡 **Analisis Singkat:** Postingan di **{top_like_post_data['platform']}** ({int(top_like_post_data['likes_count']):,} likes) adalah 'juara viralitas' Anda. "
                    f"**Saran:** Konten seperti ini sangat bagus untuk **Brand Awareness**. Gunakan format ({top_like_post_data['text_content'][:40]}...) untuk kampanye yang bertujuan menjangkau audiens baru yang belum mengenal Anda."
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
                         color_continuous_scale='Cividis')
            fig.update_layout(xaxis_title="Bahasa")
            st.plotly_chart(fig, use_container_width=True)
            
            # KESIMPULAN (DISEMPURNAKAN)
            if not lang_counts.empty:
                top_lang_data = lang_counts.iloc[0]
                second_lang = lang_counts.iloc[1]['Bahasa_Display']
                st.info(
                    f"💡 **Analisis Singkat:** Bahasa **{top_lang_data['Bahasa_Display']}** adalah audiens utama Anda ({top_lang_data['Jumlah']} postingan). "
                    f"**Saran:** Pertimbangkan untuk membuat konten spesifik atau menerjemahkan konten unggulan ke dalam bahasa kedua terpopuler Anda (**{second_lang}**) untuk memperluas jangkauan ke segmen baru."
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
                         color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            # KESIMPULAN (DISEMPURNAKAN)
            if not hash_counts.empty:
                top_hash_data = hash_counts.iloc[0]
                second_hash = hash_counts.iloc[1]['Hashtag']
                st.info(
                    f"💡 **Analisis Singkat:** Hashtag **#{top_hash_data['Hashtag']}** adalah tema sentral Anda ({top_hash_data['Jumlah']} kali). "
                    f"**Saran:** Untuk menghindari kejenuhan, kombinasikan hashtag utama ini dengan hashtag *niche* atau *trending* (seperti **#{second_hash}**) untuk menjangkau audiens yang lebih spesifik namun tetap relevan."
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
                         color_continuous_scale='Plasma')
            st.plotly_chart(fig, use_container_width=True)
            
            # KESIMPULAN (DISEMPURNAKAN)
            if not key_counts.empty:
                top_key_data = key_counts.iloc[0]
                st.info(
                    f"💡 **Analisis Singkat:** Keyword **'{top_key_data['Keyword']}'** adalah fokus utama dari strategi konten Anda ({top_key_data['Jumlah']} kali). "
                    f"**Saran:** Gunakan halaman 'Prakiraan' untuk menguji keyword ini di platform yang berbeda. Sangat mungkin keyword ini sangat laku di **Instagram**, tetapi kinerjanya biasa saja di **Twitter** (atau sebaliknya)."
                )

    # --- ======================== HALAMAN PRAKIRAAN ======================== ---
    elif selected_page == "Prakiraan":
        st.title("🔮 Prakiraan Engagement Konten")
        st.markdown("Masukkan detail konten yang akan Anda upload untuk mendapatkan prakiraan engagement.")

        with st.form("prediction_form"):
            st.subheader("Form Input Konten")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                day = st.selectbox("Hari Upload:", sorted(unique_values['day_of_week']))
                lang_codes_from_data = sorted(unique_values['language'])
                lang_display_options = sorted([LANG_MAP.get(code, code) for code in lang_codes_from_data if code in LANG_MAP])
                lang_display_selection = st.selectbox("Bahasa:", lang_display_options)
                
            with col2:
                platform = st.selectbox("Platform:", sorted(unique_values['platform']))
                campaign = st.selectbox("Campaign:", sorted(unique_values['campaign_name']))
            with col3:
                keyword = st.selectbox("Keyword Utama:", sorted([k for k in unique_values['keyword_model'] if pd.notna(k)]))
                hashtag = st.selectbox("Hashtag Utama:", sorted([h for h in unique_values['hashtag_model'] if pd.notna(h)]))

            submit_button = st.form_submit_button("Dapatkan Prakiraan 🚀", type="primary")

        if submit_button:
            lang_code = REVERSE_LANG_MAP.get(lang_display_selection, lang_display_selection)
            input_data = pd.DataFrame({
                'day_of_week': [day],
                'language': [lang_code], 
                'platform': [platform],
                'keyword_model': [keyword],
                'hashtag_model': [hashtag],
                'campaign_name': [campaign]
            })
            
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
                
                st.info("ℹ️ **Disclaimer:** Prakiraan dan saran ini dibuat berdasarkan model Machine Learning dari data historis Anda. Hasil ini adalah estimasi dan bukan jaminan performa.")

else:
    st.error("Gagal memuat data. Aplikasi tidak dapat dijalankan.")