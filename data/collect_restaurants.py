import csv
import requests
import time
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# CONFIGURATION & INITIALIZATION
# ==========================================
# Load API key securely from .env file
load_dotenv()
API_KEY = os.getenv("SEARCHAPI_KEY")

# Get directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define input and output paths relative to the script directory
INPUT_CSV = os.path.join(script_dir, "IATAcodes.csv")
OUTPUT_CSV = os.path.join(script_dir, "restaurants_detailed.csv")
RESTAURANTS_PER_IATA = 5

# --- SAFETY TIMERS ---
BASE_SLEEP = 0.5                     # Delay between successful calls
MAX_RETRIES = 3                      # Retries for rate limits or connection drops
RETRY_PENALTY_SLEEP = 5.0            # Wait 5 seconds if rate-limited

if not API_KEY:
    raise ValueError("API Key not found. Please ensure it is set in your .env file.")

def make_api_request(params):
    """Handles API requests with robust retry and rate-limit logic."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get("https://www.searchapi.io/api/v1/search", params=params, timeout=30)
            
            if response.status_code == 429:
                print(f"[429 Rate Limit! Retrying in {RETRY_PENALTY_SLEEP}s...]", end=" ", flush=True)
                time.sleep(RETRY_PENALTY_SLEEP)
                continue 
                
            elif response.status_code != 200:
                print(f"[Error {response.status_code}]", end=" ", flush=True)
                time.sleep(BASE_SLEEP)
                return None
                
            time.sleep(BASE_SLEEP)
            return response.json()
            
        except requests.exceptions.RequestException:
            print(f"[Connection Error! Retrying in {RETRY_PENALTY_SLEEP}s...]", end=" ", flush=True)
            time.sleep(RETRY_PENALTY_SLEEP)
    return None

def main():
    print("Starting Detailed Google Maps Restaurant Search.")
    print("Resume logic active. Auto-Retry active.\n")

    # 1. Read input data
    try:
        with open(INPUT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = list(reader)
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found.")
        return

    rows_to_process = lines[1:]

    # 2. Check existing progress
    processed_iatas = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                next(reader) # Skip header
                for row in reader:
                    if row:
                        processed_iatas.add(row[0]) # Column 0 is IATA
            except StopIteration:
                pass

    print(f"Found {len(processed_iatas)} already processed IATA codes. Resuming...\n")

    # Define robust output header
    output_header = [
        "iata_code", "airport_name", "search_city", 
        "restaurant_name", "data_id", "address", "latitude", "longitude", 
        "rating", "reviews_count", "price_level", "phone_number", 
        "website", "open_state", "types", "primary_image"
    ]

    # 3. Open in Append mode
    file_mode = 'a' if processed_iatas else 'w'
    with open(OUTPUT_CSV, file_mode, newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        if file_mode == 'w':
            writer.writerow(output_header)

        api_calls_made = 0

        for line_index, row in enumerate(rows_to_process, start=1):
            if not row or len(row) < 4: continue
            
            iata_code = row[0]
            airport_name = row[1]
            city = row[3]

            if iata_code in processed_iatas:
                continue

            # ==========================================
            # CALL 1: SEARCH PHASE (google_maps)
            # ==========================================
            search_query = f"restaurants near {airport_name} {city}"
            print(f"\n[{line_index}/{len(rows_to_process)}] Searching for: {search_query}...", end=" ", flush=True)

            search_params = {
                "engine": "google_maps",
                "q": search_query,
                "gl": "us",
                "hl": "en",
                "api_key": API_KEY
            }

            search_data = make_api_request(search_params)
            api_calls_made += 1

            if not search_data or not search_data.get("local_results"):
                print("No restaurants found.")
                # Save empty state to avoid re-running this failed location on resume
                writer.writerow([iata_code, airport_name, city, "No results", "", "", "", "", "", "", "", "", "", "", "", ""])
                out_file.flush()
                continue

            top_restaurants = search_data["local_results"][:RESTAURANTS_PER_IATA]
            print(f"Found {len(top_restaurants)}. Fetching details...")

            # ==========================================
            # CALL 2-6: DETAIL PHASE (google_maps_place)
            # ==========================================
            for idx, restaurant in enumerate(top_restaurants, start=1):
                data_id = restaurant.get("data_id")
                if not data_id:
                    continue

                print(f"  -> [{idx}/{len(top_restaurants)}] Fetching details for {restaurant.get('title', 'Unknown')}...", end=" ", flush=True)

                detail_params = {
                    "engine": "google_maps_place",
                    "data_id": data_id,
                    "gl": "us",
                    "hl": "en",
                    "api_key": API_KEY
                }

                detail_data = make_api_request(detail_params)
                api_calls_made += 1

                # Parse the rich data
                place = detail_data.get("place", {}) if detail_data else {}
                
                r_name = place.get("title", restaurant.get("title", "Unknown"))
                r_address = place.get("address", "N/A")
                r_lat = place.get("gps_coordinates", {}).get("latitude", "N/A")
                r_lng = place.get("gps_coordinates", {}).get("longitude", "N/A")
                r_rating = place.get("rating", "N/A")
                r_reviews = place.get("reviews", "N/A")
                r_price = place.get("price", "N/A")
                r_phone = place.get("phone", "N/A")
                r_website = place.get("website", "N/A")
                r_open_state = place.get("open_state", "N/A")
                r_types = " | ".join(place.get("types", []))
                
                # Extract at least 1 image
                r_image = "N/A"
                images = place.get("images", [])
                if images and isinstance(images, list):
                    r_image = images[0].get("link", "N/A")

                # Write directly to CSV
                writer.writerow([
                    iata_code, airport_name, city, 
                    r_name, data_id, r_address, r_lat, r_lng, 
                    r_rating, r_reviews, r_price, r_phone, 
                    r_website, r_open_state, r_types, r_image
                ])
                
                # IMMEDIATE DISK FLUSH
                out_file.flush()
                print("Saved.")

    print(f"\n✅ Session Finished. Total API calls made: {api_calls_made}.")

if __name__ == "__main__":
    main()