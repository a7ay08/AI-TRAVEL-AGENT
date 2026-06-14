import os
import re
import json
import random
from datasets import load_dataset
from tqdm import tqdm

# Constants
SYSTEM_PROMPT = "You are Etihad Al Chat, an advanced AI travel assistant. You help travelers discover destinations, find flights, hotels, and local attractions, using interactive widgets to personalize their experience."

FLIGHT_TARGET = 7000
HOTEL_ATTRACTION_TARGET = 3000
TOTAL_TARGET = 10000

print("Initializing build_finetuning_dataset.py...")

# Helper: Clean turn text from prefix
def clean_prefix(text):
    text = text.strip()
    # Remove customer: / agent: / user: / assistant: / User: / Agent: / customer: / agent:
    text = re.sub(r'^(customer|agent|user|assistant|User|Agent|Customer|Assistant)\s*:\s*', '', text)
    return text.strip()

# Helper: Extract name from text
def extract_capitalized_phrase(text, keywords):
    for kw in keywords:
        match = re.search(r'([A-Z][a-zA-Z\s\']+(?:\b' + kw + r'\b))', text)
        if match:
            return match.group(1).strip()
    return None

# Helper: Extract price from text
def extract_price(text):
    # Match $123 or 123 dollars or 123 USD
    match = re.search(r'\$(\d+(?:,\d+)?)|\b(\d+)\s*(?:dollars|usd|aed)\b', text, re.IGNORECASE)
    if match:
        val = match.group(1) or match.group(2)
        val = int(val.replace(',', ''))
        # Convert USD to AED if USD
        if '$' in text or 'dollar' in text.lower() or 'usd' in text.lower():
            val = int(val * 3.67)
        return f"AED {val:,}"
    return None

# Parser for Air Dialogue Data (google/air_dialogue)
def parse_air_dialogue():
    print("Loading Air Dialogue dataset...")
    ds = load_dataset("google/air_dialogue", "air_dialogue_data", split="train", streaming=True)
    
    parsed_dialogues = []
    count = 0
    
    for row in tqdm(ds, desc="Parsing Air Dialogue"):
        if count >= FLIGHT_TARGET:
            break
            
        dialogue = row.get("dialogue", [])
        intent = row.get("intent", {})
        action = row.get("action", {})
        
        if not dialogue or len(dialogue) < 3:
            continue
            
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        is_valid = True
        
        for idx, turn in enumerate(dialogue):
            cleaned = clean_prefix(turn)
            if not cleaned:
                is_valid = False
                break
                
            role = "user" if turn.startswith("customer:") else "assistant"
            messages.append({"role": role, "content": cleaned})
            
        if not is_valid:
            continue
            
        # Programmatic injection of widgets
        # 1. Personalization widget (first assistant turn, 30% chance if conversation starts with generic greeting)
        if len(messages) >= 3:
            first_user_content = messages[1]["content"].lower()
            if any(greet in first_user_content for greet in ["hello", "hi", "hey", "greetings"]) and random.random() < 0.3:
                widget_json = {
                    "type": "personalization",
                    "options": ["Solo", "Couple", "Family"]
                }
                messages[2]["content"] = f"Before I find the perfect destination for you, help me personalise your trip!\n\n[UI_WIDGET]\n{json.dumps(widget_json, indent=2)}\n[/UI_WIDGET]"
                
        # 2. Flight widget (on assistant turn that mentions flight number or near the end when offering flight)
        flight_injected = False
        for msg in messages[2:]:
            if msg["role"] == "assistant" and ("flight number" in msg["content"].lower() or "flight number-" in msg["content"].lower()):
                # Extract flight number
                flight_num_match = re.search(r'flight(?: number)?[- ]?(\d+)', msg["content"], re.IGNORECASE)
                flight_num = flight_num_match.group(1) if flight_num_match else "1000"
                if action.get("flight"):
                    flight_num = str(action["flight"][0])
                    
                origin = intent.get("departure_airport", "AUH")
                destination = intent.get("return_airport", "LHR")
                max_price = intent.get("max_price", 1000)
                price = f"AED {int(max_price * 0.8):,}" if max_price else "AED 950"
                
                widget_json = {
                    "type": "flight",
                    "origin": origin,
                    "destination": destination,
                    "price": price,
                    "flight_number": f"EY-{flight_num}",
                    "airline": "Etihad Airways"
                }
                msg["content"] = msg["content"] + f"\n\n[UI_WIDGET]\n{json.dumps(widget_json, indent=2)}\n[/UI_WIDGET]"
                flight_injected = True
                break
                
        # If no flight widget was injected but this was a successful booking, inject at the last assistant turn
        if not flight_injected and action.get("status") == "book" and len(messages) >= 4:
            origin = intent.get("departure_airport", "AUH")
            destination = intent.get("return_airport", "LHR")
            flight_num = str(action.get("flight", [1000])[0])
            widget_json = {
                "type": "flight",
                "origin": origin,
                "destination": destination,
                "price": "AED 850",
                "flight_number": f"EY-{flight_num}",
                "airline": "Etihad Airways"
            }
            # Inject on second to last turn if assistant, or last turn
            target_turn = messages[-1] if messages[-1]["role"] == "assistant" else messages[-2]
            if target_turn["role"] == "assistant":
                target_turn["content"] = target_turn["content"] + f"\n\n[UI_WIDGET]\n{json.dumps(widget_json, indent=2)}\n[/UI_WIDGET]"
                
        parsed_dialogues.append({"messages": messages})
        count += 1
        
    return parsed_dialogues

# Parser for SGD Flights (vidhikatkoria/SGD_Flights)
def parse_sgd_flights():
    print("Loading SGD Flights dataset...")
    ds = load_dataset("vidhikatkoria/SGD_Flights", split="train", streaming=True)
    
    parsed_dialogues = []
    
    for row in tqdm(ds, desc="Parsing SGD Flights"):
        context = row.get("context", "")
        response = row.get("response", "")
        
        if not context or not response:
            continue
            
        context_turns = [t.strip() for t in context.split("<SEP>") if t.strip()]
        turns = context_turns + [response]
        
        if len(turns) < 3:
            continue
            
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        is_valid = True
        
        for idx, turn in enumerate(turns):
            cleaned = clean_prefix(turn)
            if not cleaned:
                is_valid = False
                break
            # context alternates starting with User usually
            role = "user" if "user:" in turn.lower() or "customer:" in turn.lower() or idx % 2 == 0 else "assistant"
            messages.append({"role": role, "content": cleaned})
            
        if not is_valid:
            continue
            
        # Programmatic injection of flight widget
        flight_injected = False
        for msg in messages[2:]:
            if msg["role"] == "assistant" and any(kw in msg["content"].lower() for kw in ["flight", "ticket", "airline", "takes off", "returns"]):
                price = extract_price(msg["content"]) or "AED 650"
                # Search for airlines
                airline = "Etihad Airways"
                for air in ["American Airlines", "Southwest Airlines", "Delta", "United", "Frontier", "JetBlue"]:
                    if air.lower() in msg["content"].lower():
                        airline = air
                        break
                        
                widget_json = {
                    "type": "flight",
                    "origin": "AUH",
                    "destination": "LHR",
                    "price": price,
                    "flight_number": f"EY-{random.randint(100, 999)}",
                    "airline": airline
                }
                msg["content"] = msg["content"] + f"\n\n[UI_WIDGET]\n{json.dumps(widget_json, indent=2)}\n[/UI_WIDGET]"
                flight_injected = True
                break
                
        parsed_dialogues.append({"messages": messages})
        
    return parsed_dialogues

# Parser for MultiWOZ Hotel (vidhikatkoria/DA_MultiWOZ_hotel)
def parse_multiwoz_hotel():
    print("Loading MultiWOZ Hotel dataset...")
    ds = load_dataset("vidhikatkoria/DA_MultiWOZ_hotel", split="train", streaming=True)
    
    parsed_dialogues = []
    
    for row in tqdm(ds, desc="Parsing MultiWOZ Hotel"):
        context = row.get("context", "")
        response = row.get("response", "")
        
        if not context or not response:
            continue
            
        context_turns = [t.strip() for t in context.split("<SEP>") if t.strip()]
        turns = context_turns + [response]
        
        if len(turns) < 3:
            continue
            
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        is_valid = True
        
        for idx, turn in enumerate(turns):
            cleaned = clean_prefix(turn)
            if not cleaned:
                is_valid = False
                break
            # alternates starting with Agent or User depending on the row, let's map based on labels
            role = "user" if "user:" in turn.lower() or "customer:" in turn.lower() or idx % 2 == 0 else "assistant"
            messages.append({"role": role, "content": cleaned})
            
        if not is_valid:
            continue
            
        # Programmatic injection of hotel widget
        for msg in messages[2:]:
            if msg["role"] == "assistant" and any(kw in msg["content"].lower() for kw in ["hotel", "guest house", "guesthouse", "lodge", "stay", "room"]):
                hotel_name = extract_capitalized_phrase(msg["content"], ["Guest House", "Guesthouse", "Hotel", "Lodge", "Inn"]) or "Arbury Lodge Guesthouse"
                price = extract_price(msg["content"]) or "AED 350/night"
                
                # Extract amenities dynamically if possible
                amenities = []
                if "wifi" in msg["content"].lower() or "internet" in msg["content"].lower():
                    amenities.append("Free Wifi")
                if "park" in msg["content"].lower():
                    amenities.append("Free Parking")
                if not amenities:
                    amenities = ["Free Wifi", "Air Conditioning"]
                    
                widget_json = {
                    "type": "hotel",
                    "hotel_name": hotel_name,
                    "price": price,
                    "rating": "4.5",
                    "amenities": ", ".join(amenities)
                }
                msg["content"] = msg["content"] + f"\n\n[UI_WIDGET]\n{json.dumps(widget_json, indent=2)}\n[/UI_WIDGET]"
                break
                
        parsed_dialogues.append({"messages": messages})
        
    return parsed_dialogues

# Parser for MultiWOZ Attraction (vidhikatkoria/DA_MultiWOZ_attraction)
def parse_multiwoz_attraction():
    print("Loading MultiWOZ Attraction dataset...")
    ds = load_dataset("vidhikatkoria/DA_MultiWOZ_attraction", split="train", streaming=True)
    
    parsed_dialogues = []
    
    for row in tqdm(ds, desc="Parsing MultiWOZ Attraction"):
        context = row.get("context", "")
        response = row.get("response", "")
        
        if not context or not response:
            continue
            
        context_turns = [t.strip() for t in context.split("<SEP>") if t.strip()]
        turns = context_turns + [response]
        
        if len(turns) < 3:
            continue
            
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        is_valid = True
        
        for idx, turn in enumerate(turns):
            cleaned = clean_prefix(turn)
            if not cleaned:
                is_valid = False
                break
            role = "user" if "user:" in turn.lower() or "customer:" in turn.lower() or idx % 2 == 0 else "assistant"
            messages.append({"role": role, "content": cleaned})
            
        if not is_valid:
            continue
            
        # Programmatic injection of attraction widget
        for msg in messages[2:]:
            if msg["role"] == "assistant" and any(kw in msg["content"].lower() for kw in ["museum", "gallery", "church", "park", "attraction", "place to visit"]):
                name = extract_capitalized_phrase(msg["content"], ["Museum", "Gallery", "Church", "Park", "Schools", "College"]) or "Great Saint Mary's Church"
                
                widget_json = {
                    "type": "attraction",
                    "name": name,
                    "category": "Sightseeing",
                    "description": "A popular historic spot to explore local culture."
                }
                msg["content"] = msg["content"] + f"\n\n[UI_WIDGET]\n{json.dumps(widget_json, indent=2)}\n[/UI_WIDGET]"
                break
                
        parsed_dialogues.append({"messages": messages})
        
    return parsed_dialogues


# Execute extraction
flight_dialogues = parse_air_dialogue()
sgd_dialogues = parse_sgd_flights()

# Combine flight dialogues and slice to 7,000
total_flights = flight_dialogues + sgd_dialogues
random.shuffle(total_flights)
flight_slice = total_flights[:FLIGHT_TARGET]
print(f"Extracted and prepared {len(flight_slice)} flight dialogues.")

# Execute hotel and attraction extraction
hotel_dialogues = parse_multiwoz_hotel()
attraction_dialogues = parse_multiwoz_attraction()

# Combine hotel and attraction and slice to 3,000
total_hospitality = hotel_dialogues + attraction_dialogues
random.shuffle(total_hospitality)
hospitality_slice = total_hospitality[:HOTEL_ATTRACTION_TARGET]
print(f"Extracted and prepared {len(hospitality_slice)} hotel/attraction dialogues.")

# Merge and shuffle master dataset
master_slice = flight_slice + hospitality_slice
random.shuffle(master_slice)
print(f"Total merged master slice size: {len(master_slice)} dialogues.")

# Export to JSONL format
output_dir = "c:\\Users\\ajayv\\OneDrive\\Desktop\\Chatbot_project\\AI-TRAVEL-AGENT\\data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "master_dataset.jsonl")

print(f"Saving to {output_path}...")
with open(output_path, "w", encoding="utf-8") as f:
    for item in master_slice:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Dataset compilation completed successfully!")
