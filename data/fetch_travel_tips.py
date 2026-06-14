import csv
import json
import requests
import time
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv()

# Free Wikivoyage Action API
WIKIVOYAGE_API_URL = "https://en.wikivoyage.org/w/api.php"

# Configure the Free Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env file.")
    exit()

genai.configure(api_key=GEMINI_API_KEY)

# Using Gemini 3 Flash - It is extremely fast and specifically optimized for tasks like JSON extraction
# We enforce JSON output directly in the config to guarantee no markdown errors
model = genai.GenerativeModel(
    'gemini-3-flash-preview',
    generation_config={"response_mime_type": "application/json"}
)

INPUT_CSV = "C:\\Users\\ajayv\\OneDrive\\Desktop\\Chatbot_project\\AI-TRAVEL-AGENT\\data\\IATAcodes.csv"
OUTPUT_JSONL = "travel_tips.jsonl"

# --- SAFETY TIMERS ---
# Gemini Free Tier allows 15 Requests Per Minute (RPM). 
# Sleeping for 4 seconds between calls guarantees we stay under the limit safely.
BASE_SLEEP = 10.0                   
MAX_RETRIES = 3

def extract_niche_sections(raw_text):
    """Extracts only the highly relevant 'niche' sections from the Wikivoyage text."""
    relevant_text = ""
    sections_to_find = ["Stay safe", "Respect", "Cope", "Understand", "Get around"]
    
    for section in sections_to_find:
        pattern = rf"==\s*{section}\s*==\n(.*?)(?=\n==\s*[A-Z]|$)"
        match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            relevant_text += f"\n--- {section} ---\n" + match.group(1).strip()
            
    return relevant_text[:8000] # Gemini has a huge context window, we can send more text!

def generate_10_tips_via_llm(city, raw_context):
    """Uses Google Gemini to parse the raw wiki text into exactly 10 formatted tips."""
    if not raw_context.strip():
        return []

    prompt = f"""
    Act as a local expert for {city}. Based on the following Wikivoyage data, extract exactly 10 niche, highly-specific travel tips. 
    Focus on local etiquette, safety warnings, and transport hacks. Do not include generic tourist advice.
    
    You must output a JSON array of objects with the schema:
    [{{"category": "String", "tip": "String"}}]
    
    Wikivoyage Data:
    {raw_context}
    """

    try:
        response = model.generate_content(prompt)
        tips_json = response.text.strip()
        
        return json.loads(tips_json)
    except Exception as e:
        print(f"  [LLM Error] Failed to parse tips: {e}")
        return []

def main():
    print("Starting Wikivoyage Niche Tips Extractor (Gemini Free Tier)...")
    print("Format: JSON Lines (Progressive Save Active)\n")

    # 1. Check existing progress (Resume Logic)
    processed_iatas = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_iatas.add(data.get("iata_code"))
                except json.JSONDecodeError:
                    pass

    # 2. Read IATA Codes
    try:
        with open(INPUT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            cities_to_process = list(reader)
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found. Please ensure it is in the same directory.")
        return

    # 3. Open output file in Append mode
    with open(OUTPUT_JSONL, 'a', encoding='utf-8') as out_file:
        
        api_calls_made = 0

        for row in cities_to_process:
            iata = row.get('iata_code', '').strip()
            city = row.get('city', '').strip()
            country = row.get('country', '').strip()

            if not iata or not city:
                continue

            if iata in processed_iatas:
                continue # Skip already processed cities

            print(f"[{iata}] Fetching raw data for {city}, {country}...", end=" ", flush=True)

            # Step 1: Call free Wikivoyage API for the page text
            params = {
                "action": "query",
                "prop": "extracts",
                "titles": city,
                "explaintext": 1,
                "redirects": 1,
                "format": "json"
            }

            try:
                # Wikimedia requires a descriptive User-Agent
                headers = {
                    "User-Agent": "AITravelBotProject/1.0 (contact@yourdomain.com) Python-requests"
                }
                
                # API Call to Wikipedia/Wikivoyage with headers included
                response = requests.get(WIKIVOYAGE_API_URL, params=params, headers=headers, timeout=15)
                
                # Check if the response is actually 200 OK before parsing JSON
                if response.status_code != 200:
                    print(f"Server rejected request with status code {response.status_code}")
                    continue
                    
                data = response.json()
                
                pages = data.get("query", {}).get("pages", {})
                page = list(pages.values())[0]

                if "missing" in page:
                    print("Page not found on Wikivoyage.")
                    continue

                raw_text = page.get("extract", "")
                
                # Step 2: Filter out the noise
                niche_context = extract_niche_sections(raw_text)
                if len(niche_context) < 100: 
                    niche_context = raw_text[:8000]

                print("Parsing into 10 tips via Gemini...", end=" ", flush=True)

                # Step 3: Use Gemini LLM to structure the JSON
                tips_array = generate_10_tips_via_llm(city, niche_context)

                if tips_array:
                    # Construct the final object
                    final_record = {
                        "iata_code": iata,
                        "city": city,
                        "country": country,
                        "tips": tips_array
                    }

                    # Step 4: Write as a single JSON line and flush immediately
                    out_file.write(json.dumps(final_record) + "\n")
                    out_file.flush() 
                    
                    print(f"Success! Saved {len(tips_array)} tips.")
                    api_calls_made += 1
                else:
                    print("Failed to generate tips.")

                # Delay to respect the 15 RPM Free Tier limit
                time.sleep(BASE_SLEEP)

            except requests.exceptions.RequestException as e:
                print(f"Connection Error: {e}. Retrying later.")
                time.sleep(MAX_RETRIES)

    print(f"\n✅ Finished. Gemini API calls made: {api_calls_made}")

if __name__ == "__main__":
    main()