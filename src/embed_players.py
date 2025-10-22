import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer

# .env dosyasını yükle
load_dotenv()

# Gemini API key’i al
api_key=os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ _API_KEY bulunamadı! Lütfen .env dosyasını kontrol et.")
else:
    genai.configure(api_key=api_key)

# Veri seti yolu
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "players.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Veri dosyası bulunamadı: {DATA_PATH}")

# Veri setini yükle
df = pd.read_csv(DATA_PATH)
df = df.fillna("")

# Chroma veritabanı başlat
client = chromadb.PersistentClient(path="chroma")
collection = client.get_or_create_collection("players_info")

# SentenceTransformer (backup olarak)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_gemini_embedding(text):
    """Gemini API ile embedding üretir (fallback ile birlikte)"""
    try:
        model = genai.embed_content(
            model="models/embedding-001",
            content=text
        )
        return model["embedding"]
    except Exception as e:
        print(f"Gemini hatası ({e}), SentenceTransformer ile devam ediliyor...")
        return embedder.encode(text).tolist()

# Embedding işlemi
documents = []
metadatas = []
ids = []

for idx, row in df.iterrows():
    player_info = f"{row['player_name']} - {row['team_name']} ({row['position']}) {row['age']} yaşında, {row['goals']} gol, {row['assists']} asist"
    documents.append(player_info)
    metadatas.append({
        "player_name": row["player_name"],
        "team_name": row["team_name"],
        "position": row["position"],
        "age": row["age"]
    })
    ids.append(str(idx))

embeddings = [get_gemini_embedding(doc) for doc in documents]

# ChromaDB’ye kaydet
collection.upsert(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print("✅ Embedding işlemi tamamlandı ve Chroma veritabanına kaydedildi!")
