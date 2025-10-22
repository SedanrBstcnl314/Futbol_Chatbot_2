# src/process_players.py
import pandas as pd
import os

# Ana dizin ve data klasörü yolları
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Girdi dosyası: data/players_raw.csv
raw_path = os.path.join(DATA_DIR, "players_raw.csv")

if not os.path.exists(raw_path):
    print(f"ERROR: {raw_path} bulunamadı. Lütfen CSV'yi data/ içine koyun ve adını players_raw.csv yapın.")
    raise SystemExit(1)

# CSV'yi oku
df = pd.read_csv(raw_path, low_memory=False, dtype=str)

print("Orijinal sütunlar📊:", df.columns.tolist())
print("Örnek 5 satır:")
print(df.head(5))
print("✅ Veri okundu, toplam satır:", len(df))
print("📊 Sütunlar:", len(df.columns))

# Temel sütun isimleri
wanted_cols = [
    "Name", "First Name", "Last Name","Age",
    "Birth Date", "Birth Country", "Nationality",
    "Height", "Weight",
    "TeamName", "GamesPosition", "GamesCaptain",
    "GamesAppearences", "GamesLineups", "GamesMin",
    "ShotsGoals", "ShotsAssists", "ShotsSaves",
    "PassesAccuracy", "CardsYellow", "CardsRed", "CardsYellowRed",
    "PenaltyWon", "PenaltyCommited", 
    "PenaltyScored", "PenaltyMissed", "PenaltySaved"
]

# Bazı datasetlerde bu sütunlar farklı yazılabiliyor, o yüzden var mı kontrol et
available_cols = [col for col in wanted_cols if col in df.columns]
clean = df[available_cols].copy()

# Kolon adlarını sadeleştir
rename_map = {
    "Name": "player_name",
    "First Name": "first_name",
    "Last Name": "last_name",
    "Age": "age",
    "Birth Date": "birth_date",
    "Birth Country": "birth_country",
    "Nationality": "nationality",
    "Height": "height_cm",
    "Weight": "weight_kg",
    "TeamName": "team_name",
    "GamesPosition": "position",
    "GamesCaptain": "is_captain",
    "GamesAppearences": "appearances",
    "GamesLineups": "starting_lineups",
    "GamesMin": "minutes_played",
    "ShotsGoals": "goals",
    "ShotsAssists": "assists",
    "ShotsSaves": "shots_saves",
    "PassesAccuracy": "pass_accuracy",
    "CardsYellow": "yellow_cards",
    "CardsRed": "red_cards",
    "CardsYellowRed": "yellow_red_cards",
    "PenaltyWon": "penalty_won",
    "PenaltyCommited": "penalty_committed", 
    "PenaltyScored": "penalty_scored",
    "PenaltyMissed": "penalty_missed",
    "PenaltySaved": "penalty_saved"
}
clean.rename(columns=rename_map, inplace=True)

# Eksik değerleri doldur
clean.fillna({
    "player_name": "Bilinmiyor",
    "first_name": "",
    "last_name": "",
    "age": 0,
    "birth_country": "",
    "birth_date": "",
    "nationality": "",
    "team_name": "",
    "position": "",
    "is_captain": False,
    "goals": 0,
    "assists": 0,
    "shots_saves": 0,
    "yellow_cards": 0,
    "red_cards": 0,
    "yellow_red_cards": 0,
    "penalty_won": 0,
    "penalty_committed": 0,
    "penalty_scored": 0,
    "penalty_missed": 0,
    "penalty_saved": 0,

}, inplace=True)

# Tip dönüşümleri için tam sayı sütunları
int_cols = ["age", "appearances", "starting_lineups", "minutes_played", "height_cm", "weight_kg",
            "goals", "assists", "shots_saves", "yellow_cards", "red_cards", "yellow_red_cards",
            "penalty_won", "penalty_committed", "penalty_scored", "penalty_missed", "penalty_saved"]

for col in int_cols:
    if col in clean.columns:
        # İnt sütunları için dönüşüm (NaN'ları 0 ile doldurarak)
        clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0).astype(int)

# Pass Accuracy (Pas İsabeti) genellikle yüzde olduğu için float (ondalıklı) tutulmalı
if "pass_accuracy" in clean.columns:
    clean["pass_accuracy"] = pd.to_numeric(
        clean["pass_accuracy"], errors="coerce"
    ).fillna(0).round(2) # 2 ondalık basamağa yuvarla

# String temizliği
for col in clean.select_dtypes(include=["object"]).columns:
    clean[col] = clean[col].astype(str).str.strip()
    clean["pass_accuracy"] = pd.to_numeric(clean["pass_accuracy"], errors="coerce").fillna(0)

# Doğum tarihlerini biçimlendir
if "birth_date" in clean.columns:
    clean["birth_date"] = pd.to_datetime(
        clean["birth_date"], errors="coerce", infer_datetime_format=True
    ).dt.strftime("%Y-%m-%d") 
    # Dönüşümde hata alanlar (NaT) buraya gelir. Onları boş string yapalım.
    clean["birth_date"] = clean["birth_date"].fillna("")   

# Takım isimlerini temizle ve normalize et
clean["team_name"] = clean["team_name"].astype(str).str.strip().str.lower()

# Türkçe karakterleri İngilizce'ye çevir
replacements = {
    "ş": "s", "ı": "i", "ö": "o", "ü": "u", "ç": "c", "ğ": "g", ".": "", "-": " " # Nokta ve tireleri de temizleyelim
}
for tr, en in replacements.items():
    clean["team_name"] = clean["team_name"].str.replace(tr, en)

# Bazı takımların alternatif adlarını düzelt
clean["team_name"] = clean["team_name"].replace({
    "galatasaray a.s.": "galatasaray",
    "fenerbahce a.s.": "fenerbahce",
    "besiktas jk": "besiktas",
    "trabzonspor a.s.": "trabzonspor",
    "mke ankaragucu": "ankaragucu",
    "vavacars fatih karagumruk": "fatih karagumruk",
    "yilport samsunspor": "samsunspor",
    "ems yapispor": "ems spor",
    "istanbulspor a.s": "istanbulspor",
    "corendon alanyaspor": "alanyaspor",
    "atıkalpler konyaspor": "konyaspor"
})


# Yeni CSV olarak kaydet
out_path = os.path.join(DATA_DIR, "players.csv")
clean.to_csv(out_path, index=False, encoding="utf-8")
print(f"Temiz veri kaydedildi: {out_path}")
print(clean.head(10).to_string(index=False))
