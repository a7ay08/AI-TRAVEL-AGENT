import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
SEARCHAPI_KEY = os.getenv("SEARCHAPI_KEY")

if not SEARCHAPI_KEY:
    raise ValueError("CRITICAL: SEARCHAPI_KEY not found in .env file!")

# File Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "hotels_populated.csv")
    
# Future dates are required for Google Hotels to return property data
CHECK_IN = "2026-06-15" 
CHECK_OUT = "2026-06-16"

def fetch_single_hotel_image(hotel_name, city):
    """Searches for a specific hotel to grab its high-quality image."""
    query = f"{hotel_name} {city}"
    
    params = {
        "engine": "google_hotels",
        "q": query,
        "check_in_date": CHECK_IN,
        "check_out_date": CHECK_OUT,
        "currency": "USD",
        "api_key": SEARCHAPI_KEY
    }
    
    try:
        response = requests.get("https://www.searchapi.io/api/v1/search", params=params)
        
        if response.status_code != 200:
            print(f"  [!] API Error {response.status_code}")
            return "https://via.placeholder.com/400x300?text=Error"

        data = response.json()
        properties = data.get("properties", [])
        
        if properties:
            # Grab the top matched property
            target_hotel = properties[0]
            images_array = target_hotel.get("images", [])
            
            if images_array and len(images_array) > 0:
                # Prioritize original_image, fallback to thumbnail
                return images_array[0].get("original_image") or images_array[0].get("thumbnail")
            elif "thumbnail" in target_hotel:
                return target_hotel["thumbnail"]
                
        # Fallback if no images are found for this specific hotel
        return "https://via.placeholder.com/400x300?text=No+Image+Found"

    except Exception as e:
        print(f"  [!] Request failed: {e}")
        return "https://via.placeholder.com/400x300?text=Error"

def main():
    print(f"Loading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_CSV}")
        return

    # Add the 'image_url' column if it doesn't exist yet
    if 'image_url' not in df.columns:
        df['image_url'] = None

    total_rows = len(df)
    print(f"Found {total_rows} hotels to process.\n")

    for idx, row in df.iterrows():
        hotel_name = str(row.get('hotel_name', '')).strip()
        city = str(row.get('city', '')).strip()
        current_image = str(row.get('image_url', '')).strip()

        # Skip rows that already have a valid image URL (Resume functionality)
        if current_image and current_image.startswith("http") and "placeholder" not in current_image:
            continue
            
        if not hotel_name or hotel_name.lower() == 'nan':
            continue

        print(f"[{idx + 1}/{total_rows}] Fetching image for: {hotel_name} ({city})")
        
        # Fetch the image
        img_url = fetch_single_hotel_image(hotel_name, city)
        
        # Update the dataframe
        df.at[idx, 'image_url'] = img_url
        
        # Save the CSV immediately after every successful fetch (Atomic Save)
        # This overwrites the file safely so your single CSV gets updated in real-time
        df.to_csv(INPUT_CSV, index=False)
        
        # Polite delay to respect API rate limits
        time.sleep(1.0)

    print("\n✅ Data Enrichment Complete! All images have been added to your CSV.")

if __name__ == "__main__":
    main()