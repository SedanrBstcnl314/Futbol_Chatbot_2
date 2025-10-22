# ⚽ Türkiye Süper Ligi Chatbot ve Analiz Paneli

Bu proje, **Akbank GenAI Bootcamp** kapsamında,popüler futbol ligleri ve takımları hakkında güncel ve doğru bilgi sağlayabilen **RAG (Retrieval-Augmented Generation)** tabanlı futbol analiz chatbotudur.  
Kullanıcılar, Süper Lig futbolcuları ve takımları hakkında doğal dilde sorular sorarak istatistiksel bilgilere ulaşabilirler.  

## 📘 Proje Özeti
Bu proje, **Türkiye Süper Ligi futbolcuları ve takımları** hakkında verileri kullanarak istatistiksel analizler ve akıllı sorgulamalar yapabilen bir **Streamlit tabanlı Chatbot** uygulamasıdır.

Chatbot; LangChain, ChromaDB(vektör veritabanı), LLM (Large Language Model) ve embedding modeli Gemini API (Google Generative AI) tabanlı bir **RAG (Retrieval-Augmented Generation)** mimarisi kullanarak futbol istatistiklerini analiz eder, karşılaştırmalar yapar ve doğal dilde verilen sorulara yanıt verir.

---

Bu projede, Türkiye Süper Lig istatistiklerini içeren yapılandırılmış bir veri seti kullanılmıştır.

**Kaynak:** Veri seti, [Kaggle - Turkish Super League](https://www.kaggle.com/datasets/edacelikeloglu/turkish-super-league?utm_source=chatgpt.com) adresinden alınmıştır.

**İçerik:** 2023-2024 sezonu Türkiye Süper Ligi takımlarının puan durumları, maç sonuçları, oyuncu performans metrikleri ve genel lig istatistiklerini içeren CSV formatındaki veriler kullanılmıştır. Veriler, RAG pipeline'ına beslenmeden önce pandas kütüphanesi ile temizlenmiş ve düzenlenmiştir.

**Hazırlanış Metodolojisi (Detaylı):**

Ön İşleme (src/process_players.py): Ham CSV verisi (players_raw.csv), pandas kullanılarak okunmuş ve standartlaştırılmıştır.

Sütun Seçimi ve Yeniden Adlandırma: Yalnızca RAG için gerekli olan 20'den fazla temel oyuncu istatistiği sütunu (Örn: GamesAppearences -> appearances, ShotsGoals -> goals) seçilmiş ve isimlendirilmiştir.

Temizlik ve Tip Dönüşümü: Eksik değerler (NaN) ilgili tiplere göre doldurulmuş (sayısal alanlar için 0), veri tipleri tam sayı (int) ve ondalıklı (float) olarak dönüştürülmüştür.

Veri Normalizasyonu: Takım adları küçük harfe çevrilmiş, Türkçe karakterler (ş,ı,ö,ü,ç,ğ) normalize edilmiş ve bazı takım adlarındaki kısaltmalar (A.Ş., J.K.) temizlenerek tutarlı hale getirilmiştir (Örn: galatasaray a.s. -> galatasaray).

Temizlenmiş nihai veri seti (players.csv) RAG işlemine hazır hale getirilmiştir.

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

## Kullanılan Yöntemler ve Çözüm Mimarisi (RAG)

Proje, Google'ın güçlü Gemini modellerini kullanarak LangChain ve doğrudan ChromaDB/Gemini SDK entegrasyonuyla kurulu bir RAG mimarisi ile hayata geçirilmiştir.

Bileşen | Kullanılan Teknoloji | Amaç

Büyük Dil Modeli (LLM) | Google Gemini API (gemini-2.5-flash) | Kullanıcının sorusunu anlama ve bağlamı zenginleştirilmiş yanıtı üretme.

Vektörleştirme (Embedding) | Çift Katmanlı Model: Gemini API (models/embedding-001) ve Sentence-Transformers (all-MiniLM-L6-v2) | Veri setindeki metin parçalarını sayısal vektörlere dönüştürerek anlamsal benzerlik aramasını mümkün kılma. Gemini başarısız olursa Sentence-Transformers fallback olarak kullanılır.

Vektör Veritabanı | ChromaDB | Vektörleri depolama ve sorgu vektörüne en yakın (benzer) doküman parçalarını hızlıca getirme (Retrieval).

RAG Akışı | LangChain (Retriever ve Chain oluşturma) & Doğrudan SDK (Embedding oluşturma) | Tüm RAG bileşenlerini verimli bir şekilde bir araya getirme.

Web Arayüzü | Streamlit | Kullanıcı dostu ve hızlı bir arayüz ile chatbot'u yayınlama (Deployment).

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

🖥️ Geliştirme Ortamı

Projenin tüm kodlama, temizleme, embedding ve Streamlit arayüz geliştirme aşamaları Visual Studio Code (VS Code) ortamında gerçekleştirilmiştir.

---

---

## 💻 Kurulum (⚙️Yerel Çalıştırma)

### 1️⃣ Ortam oluşturma (Sanal)
```
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate   # Linux/macOS
```

### 2️⃣ Gerekli kütüphaneleri yükleme
Projenin bağımlılıkları requirements.txt dosyasında listelenmiştir.
```
pip install -r requirements.txt
```

### 3️⃣ Ortam değişkenleri
Proje, Gemini API'ye bağlanmak için bir ortam değişkenine ihtiyaç duyar.
Proje köküne bir `.env` dosyası ekleyin ve içine Google Gemini API anahtarınızı yazın:
```
GEMINI_API_KEY="your_gemini_api_key_here"
```

### 4️⃣ Veri hazırlama
Kaggle'dan indirilen ham CSV dosyasını (players_raw.csv adıyla) projenizin data/ klasörüne yerleştirin. Ardından.Ham veriyi temizleyip kullanılabilir hale getirin:
```
python src/process_players.py
```
Bu işlem, RAG için hazır olan nihai players.csv dosyasını oluşturacaktır.

### 5️⃣ Embedding / Vektör veritabanı(ChromaDB) oluşturma
Veri setini vektörlere dönüştürüp ChromaDB'ye yüklemek için gerekli script'i çalıştırın:
```
python src/vector_store.py
```

### 6️⃣ Uygulamayı başlat
```
streamlit run app.py
```
Tarayıcınızda otomatik olarak açılacaktır (genellikle http://localhost:8501).

---
 
> 🔗 **Canlı Uygulama:** [Streamlit App Linki](https://futbolchatbot2-kmop9icjrdb6vb5rhb3pmf.streamlit.app/)

---

## 📁 Proje Yapısı
Futbol_Chatbot_2/
│
├── app.py         # Ana uygulama (Streamlit arayüzü)
├── requirements.txt         # Gerekli kütüphaneler
│── .gitignore
|── .env
|── README.md
|
├── data/
│ ├── players_raw.csv       # Orijinal veri seti (Kaggle'dan)
│ └── players.csv            # Temizlenmiş veri seti 
│
├── src/
│ └── process_players.py     # Veri temizleme ve dönüştürme işlemleri
│└── embed_players.py
|└── vector_store.py
|
|── chroma_db/
|── chroma/
| 
└── venv/ # Sanal ortam (otomatik oluşturulur)

---

## ✅Elde Edilen Sonuçlar (Özet)

Yüksek Doğruluk: RAG mimarisi sayesinde, modelin futbolla ilgili spekülatif veya yanlış bilgi verme oranı önemli ölçüde düşürülmüştür.

Bağlam Odaklılık: Chatbot, sadece yüklenen veri setindeki bilgilere dayanarak yanıt vermektedir, bu da cevapların bağlam dışına çıkmasını engellemiştir.

Hızlı Yanıt: gemini-2.5-flash modelinin düşük gecikme süresi (latency) ve Streamlit'in performansı ile hızlı bir kullanıcı deneyimi sunulmuştur.

Gelişmiş Veri İşleme: Ham lig istatistiklerinin temizlenmesi ve takım adlarının normalize edilmesi, veri tabanındaki tutarlılığı artırarak RAG sonuçlarının kalitesini yükseltmiştir.

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
📧 s.bostancioglu4@gmail.com  
📍 Türkiye  
🚀 GAIH Generative AI Bootcamp - Final Projesi

---

## Web Linki (Canlı Uygulama)

Projemizin çalışan canlı versiyonuna aşağıdaki adresten ulaşabilirsiniz:

👉 [Futbol Chatbotu (Streamlit)](https://futbolchatbot2-kmop9icjrdb6vb5rhb3pmf.streamlit.app/)

GitHub Repo Linki:

Projenin tüm kod dosyalarına bu depodan erişebilirsiniz: [Futbol_Chatbot_2](https://github.com/SedanrBstcnl314/Futbol_Chatbot_2)