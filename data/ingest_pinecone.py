import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
INDEX_NAME = "airial-clone-index"
BATCH_SIZE = 100

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Ensure Index exists
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    print(f"Creating index '{INDEX_NAME}'... this might take a minute.")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Helper Functions ---
def get_val(row, possible_keys, default_val=""):
    """Searches a row for a list of possible column names safely."""
    for key in possible_keys:
        if key in row:
            val = row[key]
            return default_val if pd.isna(val) else val
    return default_val

def safe_float(val, default=0.0):
    """Safely converts a value to float, handling 'N/A' or strings."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def ingest_data():
    # --- Load Data ---
    print("Loading datasets...")
    # Original Data
    pois_df = pd.read_csv("pois_completed.csv")
    routes_df = pd.read_csv("routes.csv")
    with open("city_pics.json", "r") as f:
        city_pics = json.load(f)
        
    # New Data
    hotels_df = pd.read_csv("hotels_populated.csv")
    with open("pexels_media_db.json", "r") as f:
        pexels_media = json.load(f)

    # STANDARDIZE COLUMNS: Lowercase and strip hidden spaces
    pois_df.columns = pois_df.columns.str.strip().str.lower()
    routes_df.columns = routes_df.columns.str.strip().str.lower()
    hotels_df.columns = hotels_df.columns.str.strip().str.lower()

    # --- Step 1: Process POIs ---
    print("Processing POIs...")
    poi_vectors = []
    for idx, row in pois_df.iterrows():
        name = get_val(row, ['name', 'attraction_name', 'poi'], 'Unknown POI')
        city = get_val(row, ['city', 'location'], 'Unknown City')
        category = get_val(row, ['category', 'type', 'vibe'], 'General')
        desc = get_val(row, ['description', 'desc', 'details'], 'No description available.')
        rating = get_val(row, ['rating', 'score'], 0.0)

        text_content = f"{name} in {city}. Category: {category}. Description: {desc}"
        embedding = model.encode(text_content).tolist()
        
        image_url = city_pics.get(city, {}).get('image_url', "https://via.placeholder.com/400x300?text=No+Image")
        
        poi_vectors.append({
            "id": f"poi_{idx}",
            "values": embedding,
            "metadata": {
                "record_type": "poi",
                "name": str(name),
                "city": str(city),
                "category": str(category),
                "rating": safe_float(rating),
                "image_url": str(image_url)
            }
        })

    # --- Step 2: Process Routes ---
    print("Processing Routes...")
    route_vectors = []
    for idx, row in routes_df.iterrows():
        origin = get_val(row, ['origin', 'origin_iata', 'from'], 'Unknown')
        dest = get_val(row, ['destination', 'destination_iata', 'to'], 'Unknown')
        price = get_val(row, ['min_price', 'price', 'cost'], 0.0)
        duration = get_val(row, ['duration_hours', 'duration', 'time'], 'N/A')

        route_text = f"Flight route from {origin} to {dest}"
        embedding = model.encode(route_text).tolist()
        
        route_vectors.append({
            "id": f"route_{idx}",
            "values": embedding,
            "metadata": {
                "record_type": "route",
                "origin": str(origin),
                "destination": str(dest),
                "price": safe_float(price),
                "duration": str(duration)
            }
        })

    # --- Step 3: Process Hotels ---
    print("Processing Hotels...")
    hotel_vectors = []
    for idx, row in hotels_df.iterrows():
        name = get_val(row, ['hotel_name', 'name'], 'Unknown Hotel')
        city = get_val(row, ['city', 'location'], 'Unknown City')
        amenities = get_val(row, ['amenities', 'features'], 'N/A')
        rating = get_val(row, ['rating', 'score'], 0.0)
        price = get_val(row, ['price_per_night', 'price'], 'N/A')

        text_content = f"Hotel: {name} in {city}. Amenities: {amenities}"
        embedding = model.encode(text_content).tolist()

        hotel_vectors.append({
            "id": f"hotel_{idx}",
            "values": embedding,
            "metadata": {
                "record_type": "hotel",
                "name": str(name),
                "city": str(city),
                "amenities": str(amenities),
                "rating": safe_float(rating),
                "price": str(price)
            }
        })

    # --- Step 4: Process Pexels Media ---
    print("Processing Pexels Media...")
    media_vectors = []
    for loc_id, data in pexels_media.items():
        loc_name = data.get("location_name", "Unknown Location")
        # Pinecone accepts Lists of Strings for metadata filtering/storage. Limit to 5 for efficiency.
        videos = [str(v) for v in data.get("videos", [])[:5]]
        photos = [str(p) for p in data.get("photos", [])[:5]]

        text_content = f"Visual travel media, vertical videos, and scenic photos for {loc_name}"
        embedding = model.encode(text_content).tolist()

        media_vectors.append({
            "id": f"media_{loc_id}",
            "values": embedding,
            "metadata": {
                "record_type": "media",
                "location_name": str(loc_name),
                "videos": videos,
                "photos": photos
            }
        })

    # --- Step 5: Batch Upload ---
    all_data = poi_vectors + route_vectors + hotel_vectors + media_vectors
    print(f"Total records to upsert: {len(all_data)}")
    
    for i in range(0, len(all_data), BATCH_SIZE):
        batch = all_data[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i//BATCH_SIZE + 1} of {(len(all_data) + BATCH_SIZE - 1)//BATCH_SIZE}")

if __name__ == "__main__":
    ingest_data()
    print("Data Ingestion to Pinecone Complete!")