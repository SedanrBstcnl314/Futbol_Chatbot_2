import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# --- Ortam değişkenlerini yükle (.env) ---
load_dotenv()

# --- Veri ve veritabanı yolları ---
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "players.csv")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# --- Embedding modeli ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- CSV dosyasını yükle ---
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Veri dosyası bulunamadı: {DATA_PATH}")

print(f"✅ Veri dosyası bulundu: {DATA_PATH}")

# HATA DÜZELTMESİ: CSVLoader'a encoding="utf-8" eklendi.
loader = CSVLoader(file_path=DATA_PATH, encoding="utf-8")
documents = loader.load()

# --- Metinleri parçalara ayır (split) ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# --- Chroma veritabanı oluştur ---
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=CHROMA_DIR
)

print(f"✅ Chroma veritabanı başarıyla oluşturuldu: {CHROMA_DIR}")
print("🎉 Hazırsın! Şimdi 'streamlit run app.py' komutuyla uygulamayı başlat!")