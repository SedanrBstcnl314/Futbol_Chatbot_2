import streamlit as st
import pandas as pd
import os
from rapidfuzz import process, fuzz
import matplotlib.pyplot as plt
import unicodedata
import re 

import subprocess


# ===============================
# 📚 RAG KÜTÜPHANELERİ VE API YÖNETİMİ
# ===============================
from dotenv import load_dotenv

# LangChain'in güncel ve doğru içe aktarmaları
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate  # langchain_core DEĞİL!
from langchain.chains import RetrievalQA

# Vektör Veritabanı Oluşturmak İçin Ek Gerekli Importlar
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# .env dosyasını yükle ve API anahtarını kontrol et
load_dotenv() 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

if not GEMINI_API_KEY:
    st.error("❌ HATA: .env dosyasında **GEMINI_API_KEY** bulunamadı!")
    st.info("🔑 Google AI Studio'dan (https://aistudio.google.com/) bir API key alıp .env dosyasına ekleyin.")
    st.stop() 


# ===============================
# 🔤 Yardımcı Fonksiyonlar: Türkçe karakterleri normalize et
# ===============================

def normalize_text(text: str) -> str:
    """Türkçe karakterleri ve özel işaretleri RAG/arama için normalize eder."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    replacements = {
        "ç": "c", "ğ": "g", "ı": "i", "ö": "o",
        "ş": "s", "ü": "u", ".": "", "-": " "
    }
    for tr, en in replacements.items():
        text = text.replace(tr, en)
    text = re.sub(r'\s+', ' ', text).strip()
    return unicodedata.normalize("NFKD", text)

# ===============================
# 📂 Veri Yükleme ve Ön İşleme
# ===============================

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "players.csv")

@st.cache_data
def load_data() -> pd.DataFrame:
    """Oyuncu verilerini yükler ve RAG için hazırlar."""
    if not os.path.exists(DATA_PATH):
        st.error(f"HATA: {DATA_PATH} dosyası bulunamadı. Lütfen veri dosyasını kontrol edin.")
        return pd.DataFrame() 
        
    df = pd.read_csv(DATA_PATH)
    
    # NaN değerleri RAG ve istatistiksel hesaplamalar için uygun hale getir
    df["team_name"] = df["team_name"].fillna("Bilinmiyor")
    df["player_name"] = df["player_name"].fillna("Bilinmiyor")
    
    # Sayısal sütunları tam sayıya dönüştürür (eğer mümkünse)
    numeric_cols = ['age', 'goals', 'assists', 'appearances', 'minutes_played', 'yellow_cards', 'red_cards', 'height_cm']
    for column_name in numeric_cols:
         if column_name in df.columns:
              df[column_name] = df[column_name].fillna(0).astype(int)
    
    df["team_name_norm"] = df["team_name"].apply(normalize_text)
    df["player_name_norm"] = df["player_name"].apply(normalize_text)
    return df

df = load_data()

# ===============================
# 🧠 Teknik Direktörler ve Arama Fonksiyonları
# ===============================
# Not: Normalize edilmiş anahtarlar kullanmak erişimi hızlandırır ve garanti eder.
COACHES = {
    "galatasaray": "Okan Buruk", "fenerbahce": "Jose Mourinho", "besiktas": "Giovanni van Bronckhorst", 
    "trabzonspor": "Abdullah Avcı", "basaksehir": "Çağdaş Atan", "kasimpasa": "Sami Uğurlu", 
    "sivasspor": "Bülent Uygun", "alanyaspor": "Fatih Tekke", "antalyaspor": "Sergen Yalçın", 
    "adana demirspor": "Hikmet Karaman", "rizespor": "İlhan Palut", "hatayspor": "Özhan Pulat", 
    "konyaspor": "Ali Çamdalı", "gaziantep fk": "Selçuk İnan", "samsunspor": "Markus Gisdol", 
    "kayserispor": "Burak Yılmaz", "goztepe": "Stanimir Stoilov", "bodrum fk": "İsmet Taşdemir", 
    "istanbulspor": "Hakan Yakın", "pendikspor": "İbrahim Üzülmez",
}

def find_player_name(text: str) -> str | None:
    """Oyuncu adını rapidfuzz kullanarak bulur."""
    query_text = normalize_text(text)
    names = df["player_name_norm"].tolist()
    
    best_match = process.extractOne(query_text, names, scorer=fuzz.partial_ratio) 
    
    if best_match and best_match[1] > 75: 
        return df.loc[df["player_name_norm"] == best_match[0], "player_name"].values[0]
        
    return None

def find_team_name(text: str) -> str | None:
    """Takım adını bulur."""
    query_text = normalize_text(text)
    teams_dict = {normalize_text(original_team): original_team for original_team in df["team_name"].dropna().unique()}
    
    best_match = process.extractOne(query_text, teams_dict.keys(), scorer=fuzz.partial_ratio)
    if best_match and best_match[1] > 60:
        return teams_dict[best_match[0]] 
    return None

# ===============================
# ⚔️ KURAL TABANLI CEVAPLAR (Kesin ve Hızlı Yanıtlar)
# ===============================

def compare_players(player1_name: str, player2_name: str) -> str:
    """İki oyuncuyu temel istatistiklere göre karşılaştırır."""
    try:
        player1_row = df[df["player_name"] == player1_name].iloc[0]
        player2_row = df[df["player_name"] == player2_name].iloc[0]
    except IndexError:
        return "Hata: Kıyaslama yapılacak oyunculardan birinin verisi eksik veya bozuk. Lütfen isimleri kontrol edin."

    comparison_stats = [
        ("Gol", "goals", "⚽"), ("Asist", "assists", "🎯"), ("Maç", "appearances", "🏃"),
        ("Oynama Süresi (dk)", "minutes_played", "⏱️"), ("Pas İsabeti", "pass_accuracy", "✅"),
    ]

    result = f"**{player1_name} ({player1_row['team_name']}) vs. {player2_name} ({player2_row['team_name']}) Kıyaslaması**\n\n"
    
    player1_score_total = 0
    player2_score_total = 0

    for label, column_name, icon in comparison_stats:
        player1_value = player1_row[column_name]
        player2_value = player2_row[column_name]
        
        # Sayısal karşılaştırma
        if player1_value > player2_value:
            winner = player1_name
            player1_score_total += 1
        elif player2_value > player1_value:
            winner = player2_name
            player2_score_total += 1
        else:
            winner = "Berabere"

        # Formatlama
        if column_name == "pass_accuracy":
            player1_value_str = f"{player1_value:.2f}%"
            player2_value_str = f"{player2_value:.2f}%"
        else:
            player1_value_str = str(int(player1_value))
            player2_value_str = str(int(player2_value))

        result += f"- {icon} **{label}:** {player1_value_str} (vs {player2_value_str}) → Kazanan: **{winner}**\n"
    
    # Genel kazananı belirle 
    if player1_score_total > player2_score_total:
        overall_winner = player1_name
    elif player2_score_total > player1_score_total:
        overall_winner = player2_name
    else:
        overall_winner = "İstatistiksel olarak Berabere"

    result += f"\n🏆 **Genel Karşılaştırma Kazananı:** {overall_winner} ({player1_score_total} - {player2_score_total})"
    return result

def handle_ranking_query(query_norm: str) -> str | None:
    """En çok/en az gibi sıralama sorgularını işler."""
    
    if not any(keyword in query_norm for keyword in ["en cok", "en az", "top 5", "en fazla"]):
        return None
        
    stat_map = {
        "gol": "goals", "asist": "assists", "sari kart": "yellow_cards", 
        "kirmizi kart": "red_cards", "mac": "appearances", "yas": "age", 
        "dakika": "minutes_played", "pas isabeti": "pass_accuracy"
    }
    
    target_column = None
    target_label = ""
    for key_term, column_name in stat_map.items():
        if key_term in query_norm:
            target_column = column_name
            target_label = key_term
            break
            
    if target_column:
        
        ascending_order = "en az" in query_norm
        team = find_team_name(query_norm)
        filtered_df = df.copy()
        
        if team:
            filtered_df = filtered_df[filtered_df["team_name_norm"] == normalize_text(team)]
            
        top_players = filtered_df.sort_values(target_column, ascending=ascending_order).head(5)
        
        if top_players.empty:
            return f"{team or 'Genel Ligde'} için **{target_label}** verisi bulunamadı."
                
        team_info = f" ({team})" if team else ""
        rank_type = "En Az" if ascending_order else "En Çok"
        answer = f"🥇 **{rank_type} {target_label.upper()} Yapan 5 Oyuncu{team_info}:**\n"
        
        rank_list = []
        for _, row_data in top_players.iterrows():
            value_string = f"{row_data[target_column]:.2f}%" if target_column == "pass_accuracy" else str(int(row_data[target_column]))
            rank_list.append(f"{row_data['player_name']} ({row_data['team_name']}): {value_string}")
        
        answer += "\n".join(rank_list)
        return answer
            
    return None

def handle_conditional_query(query_norm: str) -> str | None:
    """Belirli koşullara (yaş > 30, gol > 5 vb.) göre filtreleme yapar."""
    
    df_filtered = df.copy()
    filters_applied = []
    
    # 1. Takım Filtresi Tespiti
    team = find_team_name(query_norm)
    if team:
        df_filtered = df_filtered[df_filtered['team_name_norm'] == normalize_text(team)]
        filters_applied.append(f"Takım: {team}")

    # 2. Pozisyon Filtresi Tespiti
    position_map = {"forvet": "Forward", "defans": "Defender", "orta saha": "Midfielder", "kaleci": "Goalkeeper"}
    position_match = next((pos_name for query_term, pos_name in position_map.items() if query_term in query_norm), None)
    
    if position_match:
        df_filtered = df_filtered[df_filtered['position'] == position_match]
        filters_applied.append(f"Pozisyon: {position_match}")

    # 3. İstatistiksel Kriterler Tespiti (Yaş, Gol, Asist, vb.)
    stat_keywords = {"yas": "age", "gol": "goals", "asist": "assists"}  # Sayı ve kelime eşleştirmesi arar: Örn: "30 yas", "5 gol"
    matches = re.findall(r'(\d+)\s*(\w+)', query_norm) 

    is_greater = any(keyword in query_norm for keyword in ['fazla', 'cok', 'buyuk', 'den buyuk', 'den fazla'])
    is_less = any(keyword in query_norm for keyword in ['az', 'kucuk', 'den az', 'den kucuk'])
    
    for value_str, condition_word in matches:
        if not value_str.isdigit(): continue
        value = int(value_str)
        
        for key_term, column_name in stat_keywords.items():
            if key_term in condition_word:
                if is_greater and not is_less:
                    df_filtered = df_filtered[df_filtered[column_name] >= value]
                    filters_applied.append(f"{key_term.capitalize()} ≥ {value}")
                elif is_less and not is_greater:
                    df_filtered = df_filtered[df_filtered[column_name] <= value]
                    filters_applied.append(f"{key_term.capitalize()} ≤ {value}")
                else: 
                    # Eşittir durumu (eğer > veya < yoksa)
                    df_filtered = df_filtered[df_filtered[column_name] == value]
                    filters_applied.append(f"{key_term.capitalize()} = {value}")

    # --- Sonuç Oluşturma ---
    if len(filters_applied) > 0:
        if not df_filtered.empty:
            filter_str = " | ".join(filters_applied)
            result_list = df_filtered.sort_values('goals', ascending=False)['player_name'].tolist()
            count = len(result_list)
            
            answer = f"🔍 **Filtreler:** ({filter_str})\n"
            answer += f"Belirtilen kriterlere uyan **{count} oyuncu** bulundu. İşte ilk 10'u:\n"
            answer += ", ".join(result_list[:10])
            
            return answer
        else:
            filter_str = " | ".join(filters_applied)
            return f"Üzgünüm, belirtilen kriterlere ({filter_str}) uyan hiçbir oyuncu bulunamadı. 😥"
            
    return None


# ===============================
# 🤖 RAG MİMARİSİ KURULUMU (Analitik ve Yorumlayıcı Yanıtlar)
# ===============================

@st.cache_resource(show_spinner="Vektör Veritabanı Yükleniyor/Oluşturuluyor...")
def setup_rag_pipeline():
    """Vektör veritabanını yükler ve RAG zincirini kurar."""
    
    ##if not os.path.isdir(CHROMA_DIR):
    ##    st.error(f"❌ Vektör veritabanı bulunamadı: {CHROMA_DIR}")
    ##    st.warning("⚠️ Lütfen önce terminalde şunu çalıştırın: python src/vector_store.py")
    ##    st.stop()
    ##    return None
    
    
    # 1. Embedding Modeli ve Vektör Depolama
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=GEMINI_API_KEY
    )
    
    # --- 2. Vektör Veritabanı Kontrolü ve Yükleme/Oluşturma (Cloud'a Uyumlu Kısım) ---
    if not os.path.isdir(CHROMA_DIR):
        # ⚠️ Veritabanı YOK! Streamlit Cloud ortamında veritabanı oluşturulacak
        
        if not os.path.exists(DATA_PATH):
            st.error(f"FATAL HATA: Veri dosyası bulunamadı: `{DATA_PATH}`. Lütfen `data/players.csv` dosyasının GitHub'da olduğundan emin olun.")
            st.stop()
            return None
            
        st.info(f"Vektör Veritabanı ({CHROMA_DIR}) bulunamadı. Lütfen bekleyin, veritabanı oluşturuluyor...")

        try:
            # Buradaki Mantık, sizin src/vector_store.py dosyanızın içindekiyle aynıdır!
            loader = CSVLoader(file_path=DATA_PATH, encoding="utf-8")
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(documents)

            # Veritabanını oluştur ve diske kaydet
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings_model, 
                persist_directory=CHROMA_DIR
            )
            st.success("✅ Vektör Veritabanı başarıyla oluşturuldu!")
        
        except Exception as e:
            st.error(f"Vektör Veritabanı oluşturulurken hata oluştu: {e}")
            st.stop()
            return None
    else:
        # 🟢 Veritabanı VAR (Lokalde veya Cloud'un cache'inde), SADECE YÜKLÜYORUZ.
        st.info(f"Vektör Veritabanı bulundu, yükleniyor...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR, 
            embedding_function=embeddings_model 
        )
        st.success("✅ Vektör Veritabanı yüklendi!")
    
    # 3. LLM Modelini Tanımlama
    llm_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    # 4. Prompt Template
    prompt_template = """
    Sen bir Türkiye Süper Ligi Analiz Botusun. YALNIZCA sağlanan futbolcu verilerini (context) kullanarak soruları TÜRKÇE yanıtla. Context, oyuncuların istatistiklerini içerir.
    
    ÖZEL KURALLAR:
    1. CONTEXT KULLANIMI: Cevabı mutlaka Context içindeki verileri kullanarak destekle.
    2. ANALİZ: Karşılaştırma, neden-sonuç veya sıralama gerektiren sorular için verileri analitik bir şekilde özetle.
    3. BULAMAMA: Cevabı Context içinde bulamazsan, 'Üzgünüm, bu spesifik bilgiyi elimdeki Süper Lig verilerinde bulamadım.' şeklinde nazikçe yanıtla.
    
    Context: {context}
    Soru: {question}
    Cevap:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # 5. RAG Zincirini Kurma
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff", 
        retriever=vectorstore.as_retriever(search_kwargs={"k": 8}), # 8 en alakalı belgeyi çeker
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False 
    )
    return qa_chain


# RAG zincirini uygulama başlatıldığında yükle
qa_rag_chain = setup_rag_pipeline() 


# ===============================
# 📊 Streamlit Arayüzü ve Dashboard
# ===============================

st.title("⚽ Türkiye Süper Ligi Chatbot ve Analiz Paneli")
st.markdown("Oyuncular, takımlar ve teknik direktörler hakkında soru sorabilirsiniz!")

if not df.empty:  # Dashboard Metrikleri
    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Oyuncu", len(df))
    col2.metric("Ortalama Yaş", f"{df['age'].mean():.1f}")
    col3.metric("Toplam Takım", df["team_name"].nunique())

    # --- En çok gol atanlar Grafiği (Panel için) ---
    st.subheader("🏅 En Çok Gol Atan 5 Futbolcu")
    top_scorers = df.sort_values("goals", ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top_scorers["player_name"], top_scorers["goals"], color="orange")
    ax.set_xlabel("Gol Sayısı")
    ax.set_ylabel("Futbolcu")
    ax.invert_yaxis()
    st.pyplot(fig)

    # --- Takım Detayları Paneli ---
    st.subheader("🏟️ Takım Detay Analizi")
    team_list = sorted(df['team_name'].dropna().unique().tolist())
    selected_team = st.selectbox("Analiz etmek istediğiniz takımı seçin:", team_list)
    
    if selected_team:
        team_df = df[df['team_name'] == selected_team]
        
        if not team_df.empty: 
            avg_age = team_df['age'].mean()
            total_goals = team_df['goals'].sum()
            total_assists = team_df['assists'].sum()
            
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            t_col1.metric("Oyuncu Sayısı", team_df.shape[0])
            t_col2.metric("Ortalama Yaş", f"{avg_age:.1f}")
            t_col3.metric("Toplam Gol", int(total_goals))
            t_col4.metric("Toplam Asist", int(total_assists))
            
            st.markdown("#### Takımın Liderleri")
            l_col1, l_col2, l_col3 = st.columns(3)
            
            # Liderleri güvenli bir şekilde çekme
            if not team_df.sort_values('goals', ascending=False).empty:
                top_goal = team_df.sort_values('goals', ascending=False).iloc[0]
                l_col1.metric("En Çok Gol Atan", top_goal['player_name'], f"{int(top_goal['goals'])} gol")
            
            if not team_df.sort_values('assists', ascending=False).empty:
                top_assist = team_df.sort_values('assists', ascending=False).iloc[0]
                l_col2.metric("En Çok Asist Yapan", top_assist['player_name'], f"{int(top_assist['assists'])} asist")

            if not team_df.sort_values('minutes_played', ascending=False).empty:
                top_min = team_df.sort_values('minutes_played', ascending=False).iloc[0]
                l_col3.metric("En Çok Süre Alan (dk)", top_min['player_name'], f"{int(top_min['minutes_played'])} dk")
            
            st.markdown("#### Takım Kadrosu (Seçili İstatistikler)")
            display_cols = ['player_name', 'position', 'age', 'height_cm', 'goals', 'assists', 'appearances', 'minutes_played', 'yellow_cards']
            
            display_df = team_df[display_cols].copy()
            for col in ['age', 'height_cm', 'goals', 'assists', 'appearances', 'minutes_played', 'yellow_cards']:
                 if col in display_df.columns:
                     display_df[col] = display_df[col].astype(int)
            
            st.dataframe(display_df.sort_values('goals', ascending=False), hide_index=True)


# ===============================
# 💬 Chatbot Alanı (HİBRİT ÇAĞRI MANTIĞI)
# ===============================

st.subheader("💬 Chatbot")
user_input = st.text_input("Sorunuzu yazın:", key="user_input")

if user_input and not df.empty:
    user_input_norm = normalize_text(user_input)
    answer = None
    
    # 1. KURAL TABANLI/KESİN CEVAPLAR (En Yüksek Öncelik)
    
    # A. Sıralama ve Koşullu Sorular
    ranking_answer = handle_ranking_query(user_input_norm)
    if not answer and ranking_answer:
        answer = ranking_answer
        
    conditional_answer = handle_conditional_query(user_input_norm)
    if not answer and conditional_answer:
        answer = conditional_answer

    # B. Kıyaslama Soruları (player1 mi player2 mi?)
    if not answer and any(word in user_input_norm for word in ["mi", "vs", "karsilastir", "daha"]):
        words = user_input.replace(",", "").split()
        player_candidates = []
        for term in words:
            match = find_player_name(term) 
            if match and match not in player_candidates:
                player_candidates.append(match)
        
        if len(player_candidates) >= 2:
            player1 = player_candidates[0]
            player2 = player_candidates[1]
            answer = compare_players(player1, player2)
        elif len(player_candidates) == 1:
            # Tek oyuncu bulunduysa LLM'e göndermek daha iyi olabilir
            pass 
        else:
            answer = "Lütfen kıyaslamak istediğiniz iki oyuncunun adını açıkça belirtin. (Örn: 'Icardi mi Džeko mu?')"
    
    # C. Kesin Takım/Oyuncu Bilgileri (TD, Toplam Gol, Yaş)
    if not answer:
        team_name = find_team_name(user_input_norm) 
        player_name = find_player_name(user_input_norm)
        
        if team_name:
            team_data = df[df["team_name"] == team_name]
            
            if "teknik direktor" in user_input_norm or "hoca" in user_input_norm:
                coach = COACHES.get(normalize_text(team_name))
                answer = f"{team_name} takımının teknik direktörü **{coach}** 👔" if coach else f"{team_name} teknik direktör bilgisi bulunamadı."
            elif "toplam gol" in user_input_norm or "kac gol atti" in user_input_norm:
                total_goals = int(team_data["goals"].sum())
                answer = f"{team_name} takımı toplamda **{total_goals}** gol attı ⚽"
            elif "ortalama yas" in user_input_norm:
                 avg_age = team_data["age"].mean()
                 answer = f"{team_name} takımının ortalama yaşı **{avg_age:.1f}** 👶"
        
        if player_name and not answer:
            player_data = df[df["player_name"] == player_name].iloc[0]
            if "gol" in user_input_norm:
                 answer = f"{player_name} bu sezon **{int(player_data['goals'])}** gol attı ⚽"
            elif "takim" in user_input_norm:
                 answer = f"{player_name} şu anda **{player_data['team_name']}** takımında oynuyor 🏟️"


    # 2. RAG TABANLI CEVAPLAR (Analitik ve Yorumlayıcı Cevaplar için Fallback)
    # Kural tabanlı sistem cevap veremezse LLM analizi devreye girer.
    if not answer and qa_rag_chain:
        with st.spinner("Cevap aranıyor... (Veri tabanında arama yapılıyor ve Gemini LLM ile analiz ediliyor)"):
            try:            # LLM'den veri çekme işlemi
                result = qa_rag_chain.invoke(user_input)
                llm_answer = result["result"]
                
                # LLM'in genel bulamama cevabını kontrol et (Örn: "Üzgünüm...")
                if "bulamadım" not in llm_answer.lower() and "geçen sezon" not in llm_answer.lower():
                    answer = llm_answer
                    
            except Exception as e:
                # LLM API veya bağlantı hatası
                st.warning(f"RAG sorgulamasında bir sorun oluştu. API veya bağlantı hatası olabilir. Detay: {e}")
                answer = None
                
    
    # 3. GENEL BULAMAMA DURUMU (Graceful Exit)
    if answer is None:
        answer = "Üzgünüm, bu spesifik bilgiyi elimdeki Süper Lig verilerinde bulamadım. Lütfen soru biçimini değiştirip tekrar deneyebilir veya daha açık bir takım/oyuncu adı yazabilirsin. 😔"

    st.success(f"🤖 **Analiz Botu:** {answer}")