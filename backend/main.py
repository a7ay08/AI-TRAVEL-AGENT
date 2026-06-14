import os
import json
import re
import csv
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils.llm_handler import LLMHandler
from utils.weather_handler import get_live_weather
from pydantic import BaseModel

load_dotenv()

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_DIR           = os.path.join(os.path.dirname(__file__), "../data")
API_ORIGINS        = ["http://localhost:3000", "http://127.0.0.1:3000"]
SEARCH_API_URL     = "https://www.searchapi.io/api/v1/search"
DEFAULT_ORIGIN     = "AUH"
FLIGHT_DAYS_AHEAD  = 30
FLIGHT_TIMEOUT     = 15.0
LLM_MODEL          = "meta-llama-3.1-8b-instruct"
LLM_BASE_URL       = "http://localhost:1234/v1/"
LLM_API_KEY        = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
searchapi_key      = os.getenv("SEARCHAPI_KEY", "")

llm_handler = LLMHandler(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=LLM_MODEL)

app = FastAPI(title="AI Travel Agent Backend")
app.add_middleware(CORSMiddleware, allow_origins=API_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ─── CSV LOADER ──────────────────────────────────────────────────────────────
def load_csv(filename: str) -> List[Dict]:
    path = os.path.join(DATA_DIR, filename)
    rows = []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        print(f"[OK] Loaded {len(rows)} rows from {filename}")
    except Exception as e:
        print(f"[FAIL] Could not load {filename}: {e}")
    return rows

def load_json(filename: str) -> Any:
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[OK] Loaded {filename}")
        return data
    except Exception as e:
        print(f"[FAIL] Could not load {filename}: {e}")
        return {}

# ─── DATA LOADING (all at startup) ───────────────────────────────────────────
POIS_DATA    : List[Dict] = load_csv("pois_completed.csv")
HOTELS_DATA  : List[Dict] = load_csv("hotels_populated.csv")
ROUTES_DATA  : List[Dict] = load_csv("routes.csv")
PEXELS_MEDIA : Dict       = load_json("pexels_media_db.json")
CITY_PICS    : Dict       = load_json("city_pics.json")

# ─── IN-MEMORY LOOKUPS ───────────────────────────────────────────────────────

# IATA → unique category list (for tags on cards)
POIS_BY_IATA: Dict[str, List[str]] = {}
for _p in POIS_DATA:
    _iata = (_p.get("destination_iata") or "").upper().strip()
    _cat  = (_p.get("category") or "").strip()
    if _iata and _cat:
        POIS_BY_IATA.setdefault(_iata, [])
        if _cat not in POIS_BY_IATA[_iata]:
            POIS_BY_IATA[_iata].append(_cat)

# Unique POI categories (for vibe filter)
POI_CATEGORIES: List[str] = []
_seen_cats: set = set()
for _p in POIS_DATA:
    _c = (_p.get("category") or "").strip()
    if _c and _c not in _seen_cats:
        _seen_cats.add(_c)
        POI_CATEGORIES.append(_c)
POI_CATEGORIES.sort()

# city_name_lower → {iata, city, country, description}
DESTINATIONS_BY_CITY: Dict[str, Dict] = {}
# iata → {iata, city, country}
DESTINATIONS_BY_IATA: Dict[str, Dict] = {}
_seen_iata: set = set()
for _p in POIS_DATA:
    _iata    = (_p.get("destination_iata") or "").upper().strip()
    _city    = (_p.get("city") or "").strip()
    _country = (_p.get("country") or "").strip()
    _desc    = (_p.get("description") or "").strip()
    _lat     = _p.get("latitude")
    _lon     = _p.get("longitude")
    if _iata and _city:
        _key = _city.lower()
        if _key not in DESTINATIONS_BY_CITY:
            DESTINATIONS_BY_CITY[_key] = {"iata": _iata, "city": _city, "country": _country, "description": _desc, "latitude": _lat, "longitude": _lon}
        if _iata not in _seen_iata:
            DESTINATIONS_BY_IATA[_iata] = {"iata": _iata, "city": _city, "country": _country, "description": _desc, "latitude": _lat, "longitude": _lon}
            _seen_iata.add(_iata)

# (origin_iata, dest_iata) → route row
ROUTES_LOOKUP: Dict[Tuple[str, str], Dict] = {}
for _r in ROUTES_DATA:
    _o = (_r.get("origin_iata") or "").upper().strip()
    _d = (_r.get("destination_iata") or "").upper().strip()
    if _o and _d:
        # keep first match per pair (best data)
        if (_o, _d) not in ROUTES_LOOKUP:
            ROUTES_LOOKUP[(_o, _d)] = _r

print(f"[READY] {len(DESTINATIONS_BY_CITY)} cities | {len(ROUTES_LOOKUP)} routes | {len(POI_CATEGORIES)} categories")

# ─── PYDANTIC MODELS ─────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class SearchQuery(BaseModel):
    traveler_type: str = "Couple"
    occasion: str = "Relaxation"
    duration: str = "1 week"
    destination_query: str
    origin: str = DEFAULT_ORIGIN
    trip_type: str = "Round Trip"   # "Round Trip" | "One Way" | "Flexible"
    travel_date: Optional[str] = None
    return_date: Optional[str] = None
    chat_history: List[Message] = []

class ExploreQuery(BaseModel):
    iata_code: str
    city: str

class FlightQuery(BaseModel):
    origin: str
    destination_iata: str
    destination_city: str
    date: Optional[str] = None
    trip_type: str = "one_way"
    traveler_type: str = "Couple"
    cabin_class: str = "economy"

# ─── HELPERS ─────────────────────────────────────────────────────────────────

# Keywords that indicate the user wants POI / activity info about a place
# (not a destination suggestion, hotel, or flight)
CONVERSATIONAL_KEYWORDS = [
    "restaurant", "restaurants", "where to eat", "where to dine", "places to eat",
    "cafe", "café", "coffee", "bar ", "bars ", "pub ", "pubs",
    "dining", "cuisine", "street food", "local food", "food in",
    "attraction", "attractions", "things to do", "what to do", "what to see",
    "what can i do", "what can i visit", "what should i", "which places",
    "which attractions", "which restaurants", "which activities",
    "sightseeing", "museum", "gallery", "monument", "temple", "castle",
    "activities in", "places to visit in", "must see", "must visit",
    "family friendly", "family-friendly", "good for family", "for families",
    "for kids", "for children", "kid friendly", "kid-friendly",
    "nightlife", "nightlife in", "entertainment in", "shows in",
    "outdoor activities", "adventure activities", "hiking in", "beaches in",
    "shopping in", "markets in", "day trips", "day trip from",
    "popular spots", "hidden gems", "local experience",
]

def detect_intent(query: str) -> str:
    """Returns 'flight', 'hotel', 'conversational', or 'destination'."""
    q = query.lower()
    if any(kw in q for kw in ["flight", "fly ", "flying", "ticket", "airfare", "prices to", "cost to", "how much to", "book a flight"]):
        return "flight"
    if any(kw in q for kw in ["hotel", "stay", "accommodation", "hostel", "resort", "where to sleep", "place to stay", "lodging"]):
        return "hotel"
    if any(kw in q for kw in CONVERSATIONAL_KEYWORDS):
        return "conversational"
    return "destination"

def find_destinations(query: str, max_results: int = 3) -> List[Dict]:
    """Find matching destinations from pois_completed.csv by text matching."""
    q = query.lower()
    matches: List[Dict] = []
    seen: set = set()

    # Exact / substring city match first
    for city_key, dest in DESTINATIONS_BY_CITY.items():
        if city_key in q and dest["iata"] not in seen:
            matches.append(dest)
            seen.add(dest["iata"])

    # Country match
    if len(matches) < max_results:
        for city_key, dest in DESTINATIONS_BY_CITY.items():
            if dest["iata"] not in seen and dest["country"].lower() in q:
                matches.append(dest)
                seen.add(dest["iata"])
                if len(matches) >= max_results:
                    break

    # Vibe / keyword broadening — pick diverse destinations if still empty
    if not matches:
        for city_key, dest in DESTINATIONS_BY_CITY.items():
            if dest["iata"] not in seen:
                matches.append(dest)
                seen.add(dest["iata"])
                if len(matches) >= max_results:
                    break

    return matches[:max_results]

def find_destinations_with_context(query: str, chat_history: List[Dict[str, str]], max_results: int = 3) -> List[Dict]:
    """Find matching destinations using query first, falling back to scanning chat history context."""
    q = query.lower()
    matches: List[Dict] = []
    seen: set = set()

    # Exact / substring city match first
    for city_key, dest in DESTINATIONS_BY_CITY.items():
        if city_key in q and dest["iata"] not in seen:
            matches.append(dest)
            seen.add(dest["iata"])

    # Country match in query
    if len(matches) < max_results:
        for city_key, dest in DESTINATIONS_BY_CITY.items():
            if dest["iata"] not in seen and dest["country"].lower() in q:
                matches.append(dest)
                seen.add(dest["iata"])
                if len(matches) >= max_results:
                    break

    # Scan chat history from most recent to oldest if still empty
    if not matches:
        for msg in reversed(chat_history):
            content = msg.get("content", "").lower()
            for city_key, dest in DESTINATIONS_BY_CITY.items():
                if city_key in content and dest["iata"] not in seen:
                    matches.append(dest)
                    seen.add(dest["iata"])
                    if len(matches) >= max_results:
                        break
            if matches:
                break

    return matches[:max_results]


def get_route_data(origin: str, dest: str) -> Dict:
    """Look up static route data from routes.csv."""
    key = (origin.upper(), dest.upper())
    row = ROUTES_LOOKUP.get(key, {})
    if not row:
        # Try with AUH as origin (many routes go through AUH)
        row = ROUTES_LOOKUP.get(("AUH", dest.upper()), {})
    return row

def get_media(iata: str, city: str) -> Tuple[str, str]:
    """Returns (video_url, image_url) for a destination."""
    video_url, image_url = "", ""
    # Try pexels_media_db
    if iata in PEXELS_MEDIA:
        media = PEXELS_MEDIA[iata]
        vids  = media.get("videos", [])
        pics  = media.get("photos", [])
        video_url = vids[0] if vids else ""
        image_url = pics[0] if pics else ""
    # Fall back to city_pics
    if not image_url and city:
        city_lower = city.lower()
        for k, v in CITY_PICS.items():
            if k.lower() == city_lower or k.lower() == iata.lower():
                image_url = v if isinstance(v, str) else (v[0] if isinstance(v, list) and v else "")
                break
    return video_url, image_url

def resolve_travel_date(query: SearchQuery, extracted_date: Optional[str]) -> str:
    """Resolve the travel date based on trip_type and widget inputs."""
    if query.trip_type == "Flexible":
        return (datetime.now() + timedelta(days=15)).strftime("%Y-%m-%d")
    if query.travel_date:
        return query.travel_date
    if extracted_date:
        return extracted_date
    return (datetime.now() + timedelta(days=FLIGHT_DAYS_AHEAD)).strftime("%Y-%m-%d")

def build_destination_rec(dest: Dict, origin: str, flight_info: Dict, live_date: str, weather_info: Optional[Dict] = None) -> Dict:
    """Build a full destination recommendation object from CSV data."""
    iata    = dest["iata"]
    city    = dest["city"]
    route   = get_route_data(origin, iata)
    video_url, image_url = get_media(iata, city)

    # Static route data (fallback values)
    duration_h  = route.get("duration_hours", "")
    avg_price   = route.get("avg_price", "")
    aircraft    = route.get("aircraft_types", "")
    emissions   = route.get("emissions_co2", "")
    dep_time    = route.get("departure_time", "")
    arr_time    = route.get("arrival_time", "")
    n_stops     = route.get("number_of_stops", "0")
    layover     = route.get("layover_details", "Direct")
    freq        = route.get("number_of_flights_per_day", "")

    flight_type_static = "Non-stop" if str(n_stops) == "0" else f"{n_stops} stop(s)"

    weather_str = "Weather unavailable"
    weather_data = {}
    if weather_info:
        try:
            weather_str = f"{weather_info['temperature_2m']}°C - 🍃 {weather_info['wind_speed_10m']} km/h"
            weather_data = weather_info
        except Exception:
            pass

    return {
        "record_type": "poi",
        "destination": city,
        "city": city,
        "country": dest["country"],
        "iata_code": iata,
        "description": dest.get("description", "A remarkable destination."),
        "video_url": video_url,
        "image_url": image_url,
        "weather": weather_str,
        "weather_data": weather_data,
        "tags": POIS_BY_IATA.get(iata, [])[:4],
        "origin_iata": origin,
        # Flight data — live overrides static
        "flight_price":    flight_info.get("flight_price")    or (f"~AED {avg_price}" if avg_price else ""),
        "flight_price_raw": flight_info.get("flight_price_raw") or 0,
        "flight_duration":  flight_info.get("flight_duration") or (f"{duration_h}h" if duration_h else ""),
        "flight_type":      flight_info.get("flight_type")    or flight_type_static,
        "flight_date":      flight_info.get("flight_date")    or live_date,
        "departure_time":   flight_info.get("departure_time") or dep_time,
        "arrival_time":     flight_info.get("arrival_time")   or arr_time,
        "aircraft":         flight_info.get("aircraft")       or (aircraft.split("|")[0].strip() if aircraft else ""),
        "airline":          flight_info.get("airline")        or "Etihad Airways",
        "flight_number":    flight_info.get("flight_number")  or "",
        # Static extras
        "emissions_co2":    emissions,
        "flights_per_day":  freq,
        "layover_details":  layover,
    }

def find_hotels(iata: str, city: str, max_results: int = 6) -> List[Dict]:
    """Look up hotels from hotels_populated.csv for a city."""
    iata_up = iata.upper()
    city_lw = city.lower()
    results = []
    for h in HOTELS_DATA:
        match_iata = h.get("iata_code", "").upper() == iata_up
        match_city = h.get("city", "").lower() == city_lw
        if match_iata or match_city:
            name = h.get("hotel_name") or h.get("name")
            if name:
                results.append({
                    "hotel_name":    name,
                    "price":         h.get("price_per_night") or "",
                    "hotel_class":   h.get("hotel_class") or "",
                    "rating":        h.get("rating") or "",
                    "reviews":       h.get("reviews") or "0",
                    "amenities":     h.get("amenities") or "",
                    "image_url":     h.get("image_url") or "",
                    "iata":          iata_up,
                    "city":          city,
                })
                if len(results) >= max_results:
                    break
    return results

def get_pois_context(query: str, iata: str, city: str, max_items: int = 8) -> str:
    """Fetch relevant POIs for a city and format them as context text for the LLM."""
    iata_up = iata.upper()
    city_lw = city.lower()
    q_lower = query.lower()

    # Detect category filter from query
    cat_map = {
        "restaurant": "food", "food": "food", "eat": "food", "dining": "food", "café": "food", "cafe": "food",
        "museum": "culture", "art": "culture", "gallery": "culture", "history": "culture", "culture": "culture",
        "shop": "shopping", "shopping": "shopping", "market": "shopping",
        "outdoor": "outdoor", "park": "outdoor", "hike": "outdoor", "nature": "outdoor",
        "beach": "beach", "adventure": "adventure", "nightlife": "nightlife",
        "family": "family", "kids": "family", "children": "family",
    }
    filter_cat = None
    for kw, cat in cat_map.items():
        if kw in q_lower:
            filter_cat = cat
            break

    pois = []
    for p in POIS_DATA:
        matches_loc = (p.get("destination_iata", "").upper() == iata_up or
                       p.get("city", "").lower() == city_lw)
        if not matches_loc:
            continue
        # Check good_for_children if family query
        if filter_cat == "family":
            if p.get("good_for_children", "").lower() not in ("true", "1", "yes"):
                continue
        elif filter_cat:
            poi_cat = p.get("category", "").lower()
            poi_tags = p.get("tags", "").lower()
            if filter_cat not in poi_cat and filter_cat not in poi_tags:
                continue
        name = p.get("attraction_name") or p.get("name", "")
        desc = p.get("description", "")
        cat  = p.get("category", "")
        rating = p.get("rating", "")
        if name:
            pois.append(f"- {name} ({cat}){' ★'+rating if rating else ''}: {desc}")
        if len(pois) >= max_items:
            break

    if not pois:
        # No filter match — just take any top-rated POIs for this city
        for p in POIS_DATA:
            if (p.get("destination_iata", "").upper() == iata_up or
                    p.get("city", "").lower() == city_lw):
                name = p.get("attraction_name") or p.get("name", "")
                desc = p.get("description", "")
                cat  = p.get("category", "")
                rating = p.get("rating", "")
                if name:
                    pois.append(f"- {name} ({cat}){' ★'+rating if rating else ''}: {desc}")
            if len(pois) >= max_items:
                break

    return "\n".join(pois) if pois else ""


def build_flight_rec(dest: Dict, origin: str, flight_info: Dict, live_date: str, weather_info: Optional[Dict] = None) -> Dict:
    """Build a flight-only recommendation object."""
    base = build_destination_rec(dest, origin, flight_info, live_date, weather_info)
    base["record_type"] = "flight"
    return base

# ─── LIVE FLIGHT SEARCH ──────────────────────────────────────────────────────

async def get_live_flight_info(dest_iata: str, target_date: Optional[str] = None,
                               origin_iata: str = DEFAULT_ORIGIN) -> Dict:
    """Searches SearchAPI for Etihad (EY) flights, trying up to 4 consecutive dates."""
    empty = {"flight_price": "", "flight_duration": "", "flight_type": "",
             "flight_date": "", "departure_time": "", "arrival_time": "",
             "aircraft": "", "airline": "Etihad Airways", "flight_number": "",
             "flight_price_raw": 0}
    if not searchapi_key:
        return empty

    try:
        base_dt = datetime.strptime(target_date, "%Y-%m-%d") if target_date else \
                  datetime.now() + timedelta(days=FLIGHT_DAYS_AHEAD)
    except (ValueError, TypeError):
        base_dt = datetime.now() + timedelta(days=FLIGHT_DAYS_AHEAD)

    async def fetch_date(offset: int) -> Optional[Dict]:
        date_str = (base_dt + timedelta(days=offset)).strftime("%Y-%m-%d")
        params = {
            "engine": "google_flights",
            "departure_id": origin_iata,
            "arrival_id": dest_iata,
            "outbound_date": date_str,
            "flight_type": "one_way",
            "currency": "AED",
            "airlines": "EY",
            "api_key": searchapi_key,
        }
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(SEARCH_API_URL, params=params, timeout=FLIGHT_TIMEOUT)
            if r.status_code != 200:
                return None
            data = r.json()
            all_flights = data.get("best_flights", []) + data.get("other_flights", [])
            ey = [f for f in all_flights
                  for leg in f.get("flights", [])
                  if ("EY" in leg.get("airline_logo", "") or
                      leg.get("airline", "") == "Etihad Airways" or
                      leg.get("flight_number", "").startswith("EY"))]
            candidates = ey or all_flights
            if not candidates:
                return None
            best    = candidates[0]
            prices  = [f.get("price") for f in candidates if f.get("price")]
            dur_min = best.get("total_duration", 0)
            h, m    = divmod(dur_min, 60)
            layovers = best.get("layovers", [])
            leg0    = best.get("flights", [{}])[0]
            dep_t   = (leg0.get("departure_airport") or {}).get("time", "")
            arr_t   = (leg0.get("arrival_airport") or {}).get("time", "")
            return {
                "flight_price":     f"AED {min(prices):,}" if prices else "",
                "flight_price_raw": min(prices) if prices else 0,
                "flight_duration":  f"{h}h {m}m" if dur_min else "",
                "flight_type":      "Non-stop" if not layovers else f"{len(layovers)} stop(s)",
                "flight_date":      date_str,
                "departure_time":   dep_t.split(" ")[1] if " " in dep_t else dep_t,
                "arrival_time":     arr_t.split(" ")[1] if " " in arr_t else arr_t,
                "aircraft":         leg0.get("airplane", ""),
                "airline":          leg0.get("airline", "Etihad Airways"),
                "flight_number":    leg0.get("flight_number", ""),
                "_offset":          offset,
            }
        except Exception:
            return None

    try:
        results = await asyncio.gather(*[fetch_date(i) for i in range(4)])
        valid   = sorted([r for r in results if r], key=lambda x: x.get("_offset", 99))
        if valid:
            best = valid[0]
            best.pop("_offset", None)
            return best
    except Exception as e:
        print(f"Flight fetch error: {e}")

    return empty

# ─── MAIN SEARCH ENDPOINT ────────────────────────────────────────────────────

@app.post("/api/search")
async def search(query: SearchQuery):
    try:
        # --- Intent Resolution ---
        chat_hist = [{"role": m.role, "content": m.content} for m in query.chat_history]
        q_lw = query.destination_query.lower()
        if any(kw in q_lw for kw in ["flight", "fly ", "flying", "ticket", "airfare", "prices to", "cost to", "how much to", "book a flight"]):
            intent = "flight"
        elif any(kw in q_lw for kw in ["hotel", "stay", "accommodation", "hostel", "resort", "where to sleep", "place to stay", "lodging"]):
            intent = "hotel"
        else:
            try:
                intent = await llm_handler.classify_intent(query.destination_query, chat_hist)
            except Exception as e:
                print(f"LLM intent classification failed: {e}. Falling back to rule-based.")
                intent = detect_intent(query.destination_query)

        # --- Date resolution ---
        extracted_date = await llm_handler.extract_travel_date(query.destination_query, chat_hist)
        live_date = resolve_travel_date(query, extracted_date)

        # --- Find destinations ---
        dests = find_destinations_with_context(query.destination_query, chat_hist, max_results=3 if intent == "destination" else 1)
        has_real_city_match = len(dests) > 0
        if not dests:
            dests = list(DESTINATIONS_BY_IATA.values())[:3]

        # --- Live flight data (for top destination) ---
        top_dest   = dests[0]
        flight_info = await get_live_flight_info(top_dest["iata"], live_date, query.origin)
        flight_ctx  = (
            f"Etihad flight {query.origin}→{top_dest['iata']} on {flight_info.get('flight_date', live_date)}: "
            f"Economy {flight_info.get('flight_price','N/A')}, "
            f"{flight_info.get('flight_duration','N/A')}, {flight_info.get('flight_type','N/A')}, "
            f"Departs {flight_info.get('departure_time','')}, Arrives {flight_info.get('arrival_time','')}."
            if any(v for v in flight_info.values() if v) else "No live Etihad flight data available."
        )

        # --- Build context for LLM based on intent ---
        poi_context = ""
        hotel_context = ""

        if intent == "conversational":
            conv_dests = find_destinations_with_context(query.destination_query, chat_hist, max_results=1)
            if conv_dests:
                cd = conv_dests[0]
                poi_context = get_pois_context(query.destination_query, cd["iata"], cd["city"])
        elif intent == "hotel":
            hotel_city_dests_tmp = find_destinations_with_context(query.destination_query, chat_hist, max_results=1)
            if hotel_city_dests_tmp:
                hd_tmp = hotel_city_dests_tmp[0]
                hotels_tmp = find_hotels(hd_tmp["iata"], hd_tmp["city"], max_results=5)
                if hotels_tmp:
                    hotel_context = "\n".join([
                        f"- {h['hotel_name']} ({h['hotel_class']}, {h['price']}/night, ★{h['rating']}): {h['amenities'][:80]}"
                        for h in hotels_tmp
                    ])

        weather_ctx = ""
        if has_real_city_match:
            try:
                lat_str = dests[0].get("latitude")
                lon_str = dests[0].get("longitude")
                if lat_str and lon_str:
                    w = get_live_weather(float(lat_str), float(lon_str))
                    if w:
                        weather_ctx = f"Current weather in {dests[0]['city']}: {w['temperature_2m']}°C, Wind {w['wind_speed_10m']} km/h, Rain {w['rain']}mm, Humidity {w['relative_humidity_2m']}%."
            except Exception:
                pass

        llm_context = [{"city": d["city"], "country": d["country"], "iata_code": d["iata"]} for d in dests] if has_real_city_match else []
        ai_resp = await llm_handler.get_destination_recommendation(
            query=query.destination_query,
            chat_history=chat_hist,
            matches=llm_context,
            flight_context=flight_ctx,
            intent=intent,
            preferences={
                "traveler_type": query.traveler_type,
                "vibe": query.occasion,
                "duration": query.duration,
                "trip_type": query.trip_type,
            },
            poi_context=poi_context,
            hotel_context=hotel_context,
            weather_context=weather_ctx,
        )

        # --- Build recommendations array from CSV data (backend is the source of truth) ---
        recs = []

        if intent == "flight":
            # Only flight cards — unchanged
            for d in dests:
                fi = flight_info if d["iata"] == top_dest["iata"] else \
                     await get_live_flight_info(d["iata"], live_date, query.origin)
                w_info = None
                if d.get("latitude") and d.get("longitude"):
                    try:
                        w_info = await get_live_weather(float(d["latitude"]), float(d["longitude"]))
                    except Exception:
                        pass
                recs.append(build_flight_rec(d, query.origin, fi, live_date, w_info))

        elif intent == "destination":
            # Destination card + flight card per destination — unchanged
            for d in dests:
                fi = flight_info if d["iata"] == top_dest["iata"] else \
                     await get_live_flight_info(d["iata"], live_date, query.origin)
                w_info = None
                if d.get("latitude") and d.get("longitude"):
                    try:
                        w_info = await get_live_weather(float(d["latitude"]), float(d["longitude"]))
                    except Exception:
                        pass
                recs.append(build_destination_rec(d, query.origin, fi, live_date, w_info))
                recs.append(build_flight_rec(d, query.origin, fi, live_date, w_info))

        elif intent == "hotel":
            # Find the specific city mentioned in the query
            hotel_city_dests = find_destinations_with_context(query.destination_query, chat_hist, max_results=1)
            if not hotel_city_dests:
                hotel_city_dests = dests[:1]
            hd = hotel_city_dests[0]
            hotels = find_hotels(hd["iata"], hd["city"])
            if hotels:
                for h in hotels:
                    recs.append({
                        "record_type":   "hotel",
                        "destination":   h["hotel_name"],
                        "city":          hd["city"],
                        "country":       hd["country"],
                        "iata_code":     hd["iata"],
                        "image_url":     h["image_url"],
                        "rating":        h["rating"],
                        "price":         h["price"],
                        "hotel_class":   h["hotel_class"],
                        "amenities":     h["amenities"],
                        "reviews_count": h["reviews"],
                        "description":   h["amenities"] or "A luxurious property.",
                    })
            # Update LLM context with hotel list for the intro text
            hotel_names = ", ".join([h["hotel_name"] for h in hotels[:4]]) if hotels else "several properties"
            ai_resp["chat_response"] = (
                f"Here are top hotels in {hd['city']}:\n\n" +
                ai_resp.get("chat_response", hotel_names)
            )

        elif intent == "conversational":
            # Conversational query — find city, get POI context, no cards
            conv_dests = find_destinations_with_context(query.destination_query, chat_hist, max_results=1)
            city_name = conv_dests[0]["city"] if conv_dests else "the requested destination"
            iata_code  = conv_dests[0]["iata"] if conv_dests else ""
            # ai_resp already has the chat_response from the LLM (with POI context)
            recs = []  # No cards for conversational


        ai_resp["recommendations"] = recs
        return {"status": "success", "data": ai_resp}

    except Exception as exc:
        print(f"Search error: {exc}")
        import traceback; traceback.print_exc()
        return {"status": "error", "message": str(exc),
                "data": {"is_new_recommendation": False,
                         "chat_response": "I encountered a glitch. Could you repeat that?",
                         "recommendations": []}}

# ─── EXPLORE ENDPOINT ────────────────────────────────────────────────────────

@app.post("/api/explore")
async def explore(query: ExploreQuery):
    iata_up = query.iata_code.upper()
    city_lw = query.city.lower()

    pois = []
    for p in POIS_DATA:
        if (p.get("destination_iata", "").upper() == iata_up or
                p.get("city", "").lower() == city_lw):
            name = p.get("attraction_name") or p.get("name")
            desc = p.get("description")
            if name and desc:
                pois.append({"name": name, "description": desc,
                             "category": p.get("category", "other")})
    pois.sort(key=lambda x: x.get("category", "zzz"))

    hotels = []
    for h in HOTELS_DATA:
        if (h.get("iata_code", "").upper() == iata_up or
                h.get("city", "").lower() == city_lw):
            name = h.get("hotel_name") or h.get("name")
            if name:
                hotels.append({
                    "name":      name,
                    "price":     h.get("price_per_night") or "N/A",
                    "class":     h.get("hotel_class") or "N/A",
                    "rating":    h.get("rating") or "N/A",
                    "reviews":   h.get("reviews") or "0",
                    "amenities": h.get("amenities") or "N/A",
                    "image_url": h.get("image_url") or "",
                })
                if len(hotels) >= 5:
                    break

    try:
        sys_prompt = (f"You are an expert travel guide. Write exactly a 100-word vivid, immersive "
                      f"description of the city: {query.city} ({query.iata_code}).")
        resp = await llm_handler.client.chat.completions.create(
            model=llm_handler.model,
            messages=[{"role": "system", "content": sys_prompt}],
            temperature=0.7, max_tokens=200)
        location_description = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Description error: {e}")
        location_description = (f"Experience the incredible sights, sounds, and flavors of "
                                 f"{query.city}. A vibrant destination blending rich history with modern marvels.")

    return {"status": "success", "data": {"description": location_description, "pois": pois, "hotels": hotels}}

# ─── DIRECT FLIGHTS ENDPOINT ─────────────────────────────────────────────────

@app.post("/api/flights")
async def search_flights(query: FlightQuery):
    if not searchapi_key:
        return {"status": "error", "message": "No SearchAPI key configured."}
    eco = await get_live_flight_info(query.destination_iata, query.date, query.origin)
    # Business class
    biz_price = ""
    try:
        search_date = eco.get("flight_date") or query.date or \
                      (datetime.now() + timedelta(days=FLIGHT_DAYS_AHEAD)).strftime("%Y-%m-%d")
        params = {"engine": "google_flights", "departure_id": query.origin,
                  "arrival_id": query.destination_iata, "outbound_date": search_date,
                  "flight_type": "one_way", "currency": "AED", "travel_class": "2",
                  "airlines": "EY", "api_key": searchapi_key}
        async with httpx.AsyncClient() as client:
            r = await client.get(SEARCH_API_URL, params=params, timeout=FLIGHT_TIMEOUT)
        if r.status_code == 200:
            bf = r.json().get("best_flights", []) + r.json().get("other_flights", [])
            bps = [f.get("price") for f in bf if f.get("price")]
            if bps:
                biz_price = f"AED {min(bps):,}"
    except Exception as e:
        print(f"Biz class error: {e}")

    if not any(eco.values()):
        return {"status": "error", "message": "No Etihad flights found for the next 4 days on this route."}

    route = get_route_data(query.origin, query.destination_iata)
    return {"status": "success", "data": {
        "origin_iata":       query.origin,
        "destination_iata":  query.destination_iata,
        "destination_city":  query.destination_city,
        "economy_price":     eco.get("flight_price", ""),
        "business_price":    biz_price,
        "flight_duration":   eco.get("flight_duration", "") or route.get("duration_hours", ""),
        "flight_type":       eco.get("flight_type", "Non-stop"),
        "flight_date":       eco.get("flight_date", ""),
        "departure_time":    eco.get("departure_time", "") or route.get("departure_time", ""),
        "arrival_time":      eco.get("arrival_time", "") or route.get("arrival_time", ""),
        "aircraft":          eco.get("aircraft", "") or route.get("aircraft_types", ""),
        "airline":           eco.get("airline", "Etihad Airways"),
        "flight_number":     eco.get("flight_number", ""),
        "emissions_co2":     route.get("emissions_co2", ""),
        "trip_type":         query.trip_type,
    }}

# ─── META ENDPOINT (vibe categories) ─────────────────────────────────────────

@app.get("/api/meta")
async def get_meta():
    return {"status": "success", "categories": POI_CATEGORIES[:20]}

@app.get("/api/health")
async def health():
    return {"status": "ok", "destinations": len(DESTINATIONS_BY_IATA),
            "routes": len(ROUTES_LOOKUP), "hotels": len(HOTELS_DATA), "pois": len(POIS_DATA)}
