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

# Ensure Index exists (all-MiniLM-L6-v2 uses 384 dimensions)
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

def ingest_hotels():
    # --- Load Data ---

    file_path = os.path.join(os.path.dirname(__file__), "hotels_populated.csv")
    
    print(f"Loading {file_path}...")
    try:
        hotels_df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return

    # STANDARDIZE COLUMNS: Lowercase and strip hidden spaces
    hotels_df.columns = hotels_df.columns.str.strip().str.lower()

    # --- Step: Process Hotels ---
    print("Processing Hotel Embeddings...")
    hotel_vectors = []
    
    for idx, row in hotels_df.iterrows():
        # Extraction
        name = get_val(row, ['hotel_name', 'name'], 'Unknown Hotel')
        city = get_val(row, ['city', 'location'], 'Unknown City')
        amenities = get_val(row, ['amenities', 'features'], 'N/A')
        rating = get_val(row, ['rating', 'score'], 0.0)
        price = get_val(row, ['price_per_night', 'price'], 'N/A')
        image_url = get_val(row, ['image_url'], "https://via.placeholder.com/400x300?text=No+Image")

        # Create semantic text block for the vector
        # We include amenities so users can search for "Pet friendly hotels in London"
        text_content = f"Hotel: {name} in {city}. Amenities: {amenities}. Rating: {rating}"
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
                "price": str(price),
                "image_url": str(image_url)
            }
        })

    # --- Step: Batch Upload ---
    print(f"Total hotel records to upsert: {len(hotel_vectors)}")
    
    for i in range(0, len(hotel_vectors), BATCH_SIZE):
        batch = hotel_vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i//BATCH_SIZE + 1} of {(len(hotel_vectors) + BATCH_SIZE - 1)//BATCH_SIZE}")

if __name__ == "__main__":
    ingest_hotels()
    print("Hotel Ingestion to Pinecone Complete!")