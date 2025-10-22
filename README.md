# ⚽ Türkiye Süper Ligi Chatbot ve Analiz Paneli

Bu proje, **GAIH GenAI Bootcamp** kapsamında geliştirilmiş bir **RAG (Retrieval-Augmented Generation)** tabanlı futbol analiz chatbotudur.  
Kullanıcılar, Süper Lig futbolcuları ve takımları hakkında doğal dilde sorular sorarak istatistiksel bilgilere ulaşabilirler.  

## 📘 Proje Özeti
Bu proje, **Türkiye Süper Ligi futbolcuları ve takımları** hakkında verileri kullanarak istatistiksel analizler ve akıllı sorgulamalar yapabilen bir **Streamlit tabanlı Chatbot** uygulamasıdır.

Chatbot; LangChain, ChromaDB(vektör veritabanı), LLM (Large Language Model) ve embedding modeli Gemini API (Google Generative AI) tabanlı bir **RAG (Retrieval-Augmented Generation)** mimarisi kullanarak futbol istatistiklerini analiz eder, karşılaştırmalar yapar ve doğal dilde verilen sorulara yanıt verir.

---

## 🚀 Özellikler
- 📊 **Takım ve oyuncu istatistikleri** sorgulama  
- ⚔️ **İki oyuncu karşılaştırması** (gol, asist, pas isabeti vb.)  
- 🧠 **RAG tabanlı cevaplama** (Gemini API + Chroma Vector Store)  
- 🔍 **Filtreli sorgular:** "30 yaş üstü forvetler", "en çok gol atan 5 futbolcu" vb.  
- 🏟️ **Takım bazlı analiz panelleri ve grafik gösterimleri**  
- 🤖 **Gemini (LLM) ile doğal dilde yanıt üretme**

---

## 🧩 Kullanılan Teknolojiler
| Bileşen | Açıklama |
|:--|:--|
| Python 3.11 | Ana programlama dili |
| Streamlit | Web arayüzü geliştirme |
| LangChain | RAG pipeline ve LLM orkestrasyonu,Framework |
| Google Gemini API | LLM + Embedding üretimi |
| ChromaDB | Vektör veritabanı (embedding’leri saklar) |
| HuggingFace Embeddings | Yedek embedding modeli |
| Sentence Transformers | Gemini erişilemezse fallback embedding üretici |
| dotenv | API anahtarlarını yönetme |
| pandas / numpy | Veri temizleme ve analiz |
| matplotlib | Görselleştirme (pozisyon, yaş dağılımı vb.) |
| RapidFuzz | Yardımcı |

---

## 🏗️ Proje Mimarisi 

```mermaid
graph TD
    A[📂 CSV Veri players_raw.csv] -->|(players.csv)| --> B[🧹 process_players.py<br/>Veri temizleme ve dönüştürme]
    B|Embedding oluştur| --> C[📦 embed_players.py<br/>Gemini + SentenceTransformer Embedding]
    C|Vektör deposu| --> D[🧠 ChromaDB<br/>Vektör veritabanı oluşturma]
    D|Sorgu + RAG işlemi| --> E[🤖 app.py<br/>LangChain + RAG + Streamlit UI]
    E --> F[👤 Kullanıcı<br/>Doğal dil sorguları]
    F --> E
```

---

## 💻 Kurulum (⚙️Yerel Çalıştırma)

### 1️⃣ Ortam oluşturma
```
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Gerekli kütüphaneleri yükleme
```
pip install -r requirements.txt
```

### 3️⃣ Ortam değişkenleri
Proje köküne bir `.env` dosyası ekleyin ve içine Google Gemini API anahtarınızı yazın:
```
GEMINI_API_KEY="your_gemini_api_key_here"
```

### 4️⃣ Veri hazırlama
Ham veriyi temizleyip kullanılabilir hale getirin:
```
python src/process_players.py
```

### 5️⃣ Embedding / Vektör veritabanı(ChromaDB) oluşturma
```
python src/vector_store.py
```

### 6️⃣ Uygulamayı başlat
```
streamlit run app.py
```

---

## 🌐 Deploy (Streamlit Cloud)

1. Projeyi GitHub’a yükle (`app.py`, `requirements.txt`, `data/players.csv`, `src/` klasörü olmalı).
2. [Streamlit Cloud](https://share.streamlit.io/) adresine git.
3. “New app” seç → repo, branch ve `app.py` yolunu belirt.
4. **Secrets** kısmına API anahtarını ekle:
   ```
   GEMINI_API_KEY="your_actual_key_here"
   ```
5. Deploy et 🎉

> ✅ Deploy sonrası link buraya eklenecek:  
> 🔗 **Canlı Uygulama:** [Streamlit App Linki](https://share.streamlit.io/...)

---

## ✅ Proje Aşamaları (GAIH Bootcamp PDF Karşılaştırması)

| Aşama | Durum | Açıklama |
|-------|--------|----------|
| Veri Toplama ve Temizleme | ✅ | `process_players.py` ile yapıldı |
| Embedding Oluşturma | ✅ | `embed_players.py` ve `vector_store.py` |
| RAG Pipeline (LangChain + Chroma) | ✅ | `app.py` içinde |
| LLM Entegrasyonu (Gemini) | ✅ | `ChatGoogleGenerativeAI` kullanıldı |
| Streamlit Arayüzü | ✅ | Dashboard + Chat kısmı |
| Model Değerlendirme | 🔄 | Yanıt doğruluğu iyileştirilecek |
| Deploy | 🔄 | Şu anda yapılmakta |

---

## 🧠 Geliştirici Notları
- `vector_store.py` yalnızca ilk seferde çalıştırılmalı.  
- Chroma veritabanı çok büyükse `.gitignore` içinde hariç tutulmalı.  
- Gemini API kota limitine dikkat edilmelidir.  
- Geliştirme sonrası model doğruluk testi (örnek sorularla) yapılmalıdır.

---

## ✨ Örnek Sorgular
- "Galatasaray’da kaç futbolcu var?"  
- "Icardi mi Džeko mu daha çok gol attı?"  
- "30 yaş üstü forvetler kimler?"  
- "Fenerbahçe’nin ortalama yaşı kaç?"  
- "En çok asist yapan 5 futbolcu kim?"
- “Trabzonspor’un teknik direktörü kim?”

---

## 👩‍💻 Geliştirici
**Sedanur Bostancıoğlu**  
📧 sedanurbostancioglu@example.com  
📍 Türkiye  
🚀 GAIH Generative AI Bootcamp - Final Projesi
