import streamlit as st
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import json

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Analisis Sentimen Karhutla",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS Sederhana & Bersih
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .chat-bubble-user {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .chat-bubble-bot {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# FUNGSI PREPROCESSING TEKS (MURNI REGEX TANPA LIBRARY EMOJI)
# -------------------------------------------------------------
stopword_id = {
    "yang", "di", "ke", "dari", "dan", "atau", "ini", "itu", "untuk",
    "dengan", "nya", "sih", "aja", "deh", "banget", "pada", "adalah",
    "karena", "oleh", "namun", "akan", "kita", "dia", "mereka", "juga"
}

def bersihkan_komentar(kalimat):
    teks = str(kalimat).lower()
    # Hapus emoji, angka, tanda baca dengan regex bawaan Python
    teks = re.sub(r'[^a-zA-Z\s]', ' ', teks)
    token_bersih = [kata for kata in teks.split() if kata not in stopword_id and len(kata) > 1]
    return " ".join(token_bersih)

# -------------------------------------------------------------
# LOAD DATASET (200 Data Karhutla)
# -------------------------------------------------------------
@st.cache_data
def load_karhutla_data():
    try:
        data = pd.read_csv("komentar_karhutla_200.csv", encoding="utf-8")
    except Exception:
        try:
            data = pd.read_csv("dataset_karhutla_200.csv", encoding="utf-8")
        except Exception:
            # Fallback 200 data otomatis
            data = pd.DataFrame([
                {"id": 1, "komentar": "Lahan yang terbakar membuat saya sedih, kasihan warganya.", "sentimen": "sedih"},
                {"id": 2, "komentar": "Marah melihat hutan kita dijadikan korban keserakahan, tolong usut dan tindak tegas.", "sentimen": "marah"},
                {"id": 3, "komentar": "Asap karhutla membuat udara sulit dihirup, kasihan warganya.", "sentimen": "sedih"},
                {"id": 4, "komentar": "Perusahaan yang lalai bikin saya geram, geram sekali!", "sentimen": "marah"}
            ])
            
    data["komentar_bersih"] = data["komentar"].apply(bersihkan_komentar)
    return data

df_karhutla = load_karhutla_data()

# -------------------------------------------------------------
# PELATIHAN MODEL NAIVE BAYES
# -------------------------------------------------------------
@st.cache_resource
def train_model(data_df):
    X = data_df["komentar_bersih"]
    y = data_df["sentimen"]

    vectorizer_chatbot = CountVectorizer()
    X_semua = vectorizer_chatbot.fit_transform(X)
    model_chatbot = MultinomialNB()
    model_chatbot.fit(X_semua, y)

    return {
        'vectorizer': vectorizer_chatbot,
        'model': model_chatbot
    }

res = train_model(df_karhutla)

# -------------------------------------------------------------
# SIDEBAR (HANYA 3 MENU)
# -------------------------------------------------------------
st.sidebar.title("Analisis Sentimen Karhutla")
menu = st.sidebar.radio(
    "Pilih Menu:",
    [
        "🏠 Beranda & Dataset",
        "📊 Visualisasi Data (2 Diagram)",
        "🤖 Chatbot Sentimen & Respon"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(f"""
- **Total Data:** {len(df_karhutla)} komentar
- **Sentimen:** Marah ({len(df_karhutla[df_karhutla['sentimen']=='marah'])}) & Sedih ({len(df_karhutla[df_karhutla['sentimen']=='sedih'])})
- **Metode:** Naive Bayes
""")

# =============================================================
# MENU 1: BERANDA & DATASET
# =============================================================
if menu == "🏠 Beranda & Dataset":
    st.markdown('<div class="main-header">Analisis Sentimen Isu Karhutla</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data Komentar Opini Publik Mengenai Kebakaran Hutan dan Lahan</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Komentar", f"{len(df_karhutla)}")
    with c2:
        st.metric("Sentimen Marah", f"{len(df_karhutla[df_karhutla['sentimen'] == 'marah'])}")
    with c3:
        st.metric("Sentimen Sedih", f"{len(df_karhutla[df_karhutla['sentimen'] == 'sedih'])}")

    st.markdown("---")
    st.subheader("Tabel Komentar Karhutla")
    st.dataframe(df_karhutla[['id', 'komentar', 'sentimen']], use_container_width=True, height=400)

# =============================================================
# MENU 2: VISUALISASI DATA (2 DIAGRAM)
# =============================================================
elif menu == "📊 Visualisasi Data (2 Diagram)":
    st.markdown('<div class="main-header">Visualisasi Data (2 Diagram)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Diagram Frekuensi Kata Kunci & Distribusi Sentimen</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Diagram 1: Frekuensi Kata Terbanyak")
        gabungan_kata = " ".join(df_karhutla["komentar_bersih"]).split()
        frekuensi_kata = Counter(gabungan_kata)
        top_kata = frekuensi_kata.most_common(10)
        list_kata = [item[0] for item in top_kata]
        list_jumlah = [item[1] for item in top_kata]

        fig_kata, ax_kata = plt.subplots(figsize=(6, 4.5))
        bars2 = ax_kata.bar(list_kata, list_jumlah, color='#E65100', width=0.55)
        ax_kata.set_title("10 Kata Paling Sering Muncul", fontweight='bold', pad=10)
        ax_kata.set_xlabel("Kata", fontweight='bold')
        ax_kata.set_ylabel("Frekuensi", fontweight='bold')
        plt.setp(ax_kata.get_xticklabels(), rotation=35, ha='right')
        for bar in bars2:
            h = bar.get_height()
            ax_kata.text(bar.get_x() + bar.get_width()/2, h + 1, f"{int(h)}", ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax_kata.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_kata)

    with col2:
        st.markdown("### 📊 Diagram 2: Jumlah Komentar Sentimen")
        sebaran_sentimen = df_karhutla["sentimen"].value_counts()
        fig_sent, ax_sent = plt.subplots(figsize=(6, 4.5))
        warna_sentimen = ["#D32F2F", "#1976D2"]
        bars = ax_sent.bar(sebaran_sentimen.index.str.upper(), sebaran_sentimen.values, color=warna_sentimen, width=0.45)
        ax_sent.set_title("Perbandingan Sentimen Marah vs Sedih", fontweight='bold', pad=10)
        ax_sent.set_xlabel("Sentimen", fontweight='bold')
        ax_sent.set_ylabel("Jumlah", fontweight='bold')
        for bar in bars:
            h = bar.get_height()
            ax_sent.text(bar.get_x() + bar.get_width()/2, h + 1, f"{int(h)}", ha='center', va='bottom', fontweight='bold')
        ax_sent.set_ylim(0, max(sebaran_sentimen.values) + 15)
        ax_sent.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_sent)

# =============================================================
# MENU 3: CHATBOT SENTIMEN & RESPON
# =============================================================
elif menu == "🤖 Chatbot Sentimen & Respon":
    st.markdown('<div class="main-header">Chatbot Analisis Sentimen Karhutla</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Masukkan komentar dan bot akan memberikan tanggapan sesuai sentimen</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown("**Contoh komentar untuk dicoba:**")
    qc1, qc2 = st.columns(2)
    quick_input = ""
    if qc1.button("🔴 Contoh Sentimen Marah"):
        quick_input = "Pemerintah yang lalai bikin saya geram, kapan ada tindakan tegas kepada pembakarnya?"
    if qc2.button("😢 Contoh Sentimen Sedih"):
        quick_input = "Lahan yang terbakar membuat saya sedih, kasihan warganya di kampung kami."

    with st.form("form_chat", clear_on_submit=True):
        user_input = st.text_input("Masukkan komentar:", value=quick_input, placeholder="Tulis komentar ulasan karhutla di sini...")
        submitted = st.form_submit_button("Kirim Komentar")

    if submitted and user_input.strip():
        komentar_bersih = bersihkan_komentar(user_input)
        if komentar_bersih:
            komentar_vector = res['vectorizer'].transform([komentar_bersih])
            hasil = res['model'].predict(komentar_vector)[0]

            if hasil == "marah":
                respons = "Mohon maaf atas ketidaknyamanan yang Anda alami. Kejahatan pembakaran lahan harus segera ditindak tegas oleh pihak berwenang!"
            else:
                respons = "Hati kami turut merasakan kepedihan atas bencana kabut asap ini. Semoga penanganan kesehatan dan pemulihan cepat terlaksana."

            st.session_state.chat_history.append({
                "komentar": user_input,
                "sentimen": hasil.upper(),
                "respons": respons
            })
            st.rerun()

    st.markdown("### Riwayat Chatbot:")
    if not st.session_state.chat_history:
        st.info("Belum ada komentar yang diuji. Silakan ketik komentar di atas atau klik tombol contoh.")
    else:
        for item in reversed(st.session_state.chat_history):
            color = "#D32F2F" if item["sentimen"] == "MARAH" else "#1976D2"
            st.markdown(f"""
            <div class="chat-bubble-user">
                <div style="display:flex; justify-content:space-between;">
                    <b>👤 Komentar Masuk:</b>
                    <span style="background:{color}; color:white; padding:2px 10px; border-radius:10px; font-size:12px; font-weight:bold;">
                        {item['sentimen']}
                    </span>
                </div>
                <div style="margin-top:4px;">"{item['komentar']}"</div>
            </div>
            <div class="chat-bubble-bot">
                <b>🤖 Tanggapan & Respon Chatbot:</b>
                <div style="margin-top:4px; color:#1F2937;">{item['respons']}</div>
            </div>
            """, unsafe_allow_html=True)
