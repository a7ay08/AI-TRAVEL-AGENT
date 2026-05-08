import csv
import requests
import time
import os
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION & PLACEHOLDERS
# ==========================================
API_KEY = "YOUR_API_KEY"  # <-- Paste your SearchApi key here
INPUT_CSV = "IATAcodes.csv"
OUTPUT_CSV = "hotels_populated.csv"
MAX_CREDITS_TO_USE = 1500           # Total API calls to allow for this run
DAYS_IN_ADVANCE = 30                 # Start searching 30 days from today
STAY_DURATION = 1                    # Number of nights for the query

# --- SAFETY TIMERS ---
BASE_SLEEP = 0.5                     # Standard 0.5-second delay between successful calls
MAX_RETRIES = 3                      # How many times to retry if the server blocks you
RETRY_PENALTY_SLEEP = 5.0            # Wait 5 seconds if rate-limited before trying again

def main():
    print("Starting Robust Google Hotels Search.")
    print(f"Resume logic active. Auto-Retry active. Max credits: {MAX_CREDITS_TO_USE}\n")

    # 1. Read the raw input file
    try:
        with open(INPUT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            lines = list(reader)
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found. Please ensure it's in the same directory.")
        return

    input_header = lines[0]
    rows_to_process = lines[1:]

    # 2. Check for existing progress to resume (Set based to handle 1-to-many rows)
    processed_iatas = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                next(reader) # Skip the header
                for row in reader:
                    if row:
                        processed_iatas.add(row[0]) # Column 0 is the IATA code
            except StopIteration:
                pass

    print(f"Found {len(processed_iatas)} already processed IATA codes. Resuming...\n")

    # Define Output Header
    output_header = [
        "iata_code", "airport_name", "city",
        "hotel_name", "price_per_night", "rating", "reviews", "hotel_class", "amenities"
    ]

    # 3. Open in Append mode
    file_mode = 'a' if processed_iatas else 'w'
    with open(OUTPUT_CSV, file_mode, newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        if file_mode == 'w':
            writer.writerow(output_header)

        api_calls_made = 0
        reached_limit = False

        for line_index, row in enumerate(rows_to_process, start=1):
            if reached_limit: break
            
            # Ensure row has enough columns based on your IATAcodes.csv structure
            if not row or len(row) < 4: continue
            
            iata_code = row[0]
            airport_name = row[1]
            city = row[3]

            # Resume Logic: Skip if we already fetched hotels for this IATA
            if iata_code in processed_iatas:
                continue

            if api_calls_made >= MAX_CREDITS_TO_USE:
                print("\nReached total credit limit.")
                reached_limit = True
                break

            # Date calculation required by Google Hotels API
            check_in_date = (datetime.now() + timedelta(days=DAYS_IN_ADVANCE)).strftime("%Y-%m-%d")
            check_out_date = (datetime.now() + timedelta(days=DAYS_IN_ADVANCE + STAY_DURATION)).strftime("%Y-%m-%d")
            
            # Create a highly targeted query based on the metadata in your CSV
            query = f"hotels near {airport_name} {city}"
            print(f"[{line_index}/{len(rows_to_process)}] Fetching {iata_code} ({query})...", end=" ", flush=True)

            params = {
                "engine": "google_hotels",
                "q": query,
                "check_in_date": check_in_date,
                "check_out_date": check_out_date,
                "gl": "us",
                "hl": "en",
                "currency": "USD",
                "api_key": API_KEY
            }

            # ==========================================
            # ROBUST RETRY LOGIC
            # ==========================================
            request_successful = False
            for attempt in range(MAX_RETRIES):
                try:
                    response = requests.get("https://www.searchapi.io/api/v1/search", params=params, timeout=30)
                    
                    # Handle 429 Rate Limit
                    if response.status_code == 429:
                        print(f"[429 Rate Limit! Retrying in {RETRY_PENALTY_SLEEP}s...]", end=" ", flush=True)
                        time.sleep(RETRY_PENALTY_SLEEP)
                        continue 
                        
                    # Handle other bad errors
                    elif response.status_code != 200:
                        print(f"[Error {response.status_code}]", end=" ", flush=True)
                        time.sleep(BASE_SLEEP)
                        break 
                        
                    # If 200 OK, we are successful
                    latency = response.elapsed.total_seconds()
                    print(f"({latency:.2f}s latency)", end=" ", flush=True)
                    api_calls_made += 1
                    time.sleep(BASE_SLEEP)
                    request_successful = True
                    break 
                    
                except requests.exceptions.RequestException:
                    print(f"[Connection Error! Retrying in {RETRY_PENALTY_SLEEP}s...]", end=" ", flush=True)
                    time.sleep(RETRY_PENALTY_SLEEP)
            
            if not request_successful:
                print("Failed after retries.")
                continue

            # ==========================================
            # PARSE THE DATA (TOP 5 HOTELS)
            # ==========================================
            try:
                data = response.json()
                properties = data.get("properties", [])
                
                if not properties:
                    print("No hotels found.")
                    # Write an empty row so we know we searched it but found nothing
                    writer.writerow([iata_code, airport_name, city, "No hotels found", "", "", "", "", ""])
                else:
                    hotels_written = 0
                    
                    # Slice the list to get a maximum of 5 hotels
                    for prop in properties[:5]:
                        h_name = prop.get("name", "Unknown")
                        h_price = prop.get("rate_per_night", {}).get("lowest", "N/A")
                        h_rating = prop.get("rating", "N/A")
                        h_reviews = prop.get("reviews", "N/A")
                        h_class = prop.get("hotel_class", "N/A")
                        h_amenities = " | ".join(prop.get("amenities", []))
                        
                        writer.writerow([
                            iata_code, airport_name, city, 
                            h_name, h_price, h_rating, h_reviews, h_class, h_amenities
                        ])
                        hotels_written += 1
                        
                    print(f"Success! {hotels_written} hotels saved.")
                    
            except Exception as e:
                print(f"Data parsing failed: {e}")
                continue

            # Safely flush to disk after every successful location chunk
            out_file.flush() 

    print(f"\n✅ Session Finished. {api_calls_made} API credits used.")

if __name__ == "__main__":
    main()