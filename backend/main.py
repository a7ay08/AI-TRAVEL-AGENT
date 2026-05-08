import os
import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils.llm_handler import LLMHandler
from pinecone import Pinecone
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool

load_dotenv()

# --- CONFIGURATION ---
API_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
PINECONE_INDEX_NAME = "airial-clone-index" 
SEARCH_API_URL = "https://www.searchapi.io/api/v1/search"
DEFAULT_ORIGIN_IATA = "AUH"
FLIGHT_LOOKUP_DAYS_AHEAD = 30
FLIGHT_TIMEOUT_SECONDS = 15.0
LLM_MODEL = "meta-llama-3.1-8b-instruct"
LLM_BASE_URL = "http://localhost:1234/v1/"
LLM_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

app = FastAPI(title="AI Travel Agent Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
searchapi_key = os.getenv("SEARCHAPI_KEY")

llm_handler = LLMHandler(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=LLM_MODEL)
pc = Pinecone(api_key=pinecone_api_key)
index = None
try:
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as exc:
    print(f"Error connecting to Pinecone: {exc}")

print("Loading local embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.")

# --- DATA MODELS ---
class Message(BaseModel):
    role: str
    content: str

class SearchQuery(BaseModel):
    traveler_type: str
    occasion: str
    duration: str
    destination_query: str
    origin: str = DEFAULT_ORIGIN_IATA 
    chat_history: List[Message] = []

# --- 1. LIVE FLIGHT CONTEXT ---

async def get_live_flight_info(dest_iata: str, target_date: Optional[str] = None, origin_iata: str = DEFAULT_ORIGIN_IATA) -> str:
    if not searchapi_key: return "Flight pricing data is currently syncing."
    
    search_date = target_date or (datetime.now() + timedelta(days=FLIGHT_LOOKUP_DAYS_AHEAD)).strftime("%Y-%m-%d")
    params = {
        "engine": "google_flights", "departure_id": origin_iata, "arrival_id": dest_iata,
        "outbound_date": search_date, "flight_type": "one_way", "currency": "USD", "api_key": searchapi_key
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(SEARCH_API_URL, params=params, timeout=FLIGHT_TIMEOUT_SECONDS)
            if response.status_code == 200:
                data = response.json()
                all_flights = data.get("best_flights", []) + data.get("other_flights", [])
                if all_flights:
                    prices = [f.get("price") for f in all_flights if f.get("price")]
                    airline = all_flights[0].get("flights", [{}])[0].get("airline", "carrier")
                    return f"Live flights from {origin_iata} for {search_date} start at ${min(prices)} via {airline}."
        return f"Flight info for {search_date} from {origin_iata} is currently limited."
    except Exception as e:
        print(f"Flight fetch error: {e}")
        return "Flight data temporarily unavailable."

# --- 2. CORE LOGIC ---
async def encode_text(text: str) -> List[float]:
    vector = await run_in_threadpool(embedding_model.encode, text)
    return list(vector.tolist() if hasattr(vector, "tolist") else vector)

async def query_pinecone(vector: List[float], top_k: int = 5) -> Dict[str, Any]:
    if index is None: raise RuntimeError("Index not initialized.")
    return await run_in_threadpool(lambda: index.query(vector=vector, top_k=top_k, include_metadata=True))

def normalize_match_metadata(match: Dict[str, Any]) -> Dict[str, str]:
    md = match.get("metadata", {}) or {}
    video_url = md.get("video_url") or (md.get("videos")[0] if isinstance(md.get("videos"), list) and md.get("videos") else "")
    image_url = md.get("image_url") or (md.get("photos")[0] if isinstance(md.get("photos"), list) and md.get("photos") else "")
    return {
        "destination": str(md.get("attraction_name") or md.get("location_name") or md.get("hotel_name") or "Destination"),
        "iata_code": str(md.get("destination_iata") or md.get("iata_code") or "DXB"),
        "description": str(md.get("description") or "A luxury escape."),
        "video_url": str(video_url) if video_url else "", 
        "image_url": str(image_url) if image_url else "",
        "weather": "28°C Sunny", 
        "tags": ["Luxury", "Escape"]
    }



@app.post("/api/search")
async def search_destination(query: SearchQuery):
    if index is None: raise HTTPException(status_code=500, detail="Pinecone error.")

    try:
        search_text = f"{query.destination_query}. {query.traveler_type} vibe."
        query_vector = await encode_text(search_text)
        results = await query_pinecone(query_vector, top_k=3) 
        matches = [normalize_match_metadata(m) for m in results.get("matches", [])] if results.get("matches") else []
        
        chat_hist_dicts = [{"role": m.role, "content": m.content} for m in query.chat_history]
        extracted_date = await llm_handler.extract_travel_date(query.destination_query, chat_hist_dicts)
        top_iata = matches[0]['iata_code'] if matches else "DXB"
        flight_context = await get_live_flight_info(top_iata, extracted_date, query.origin)

        ai_recommendation = await llm_handler.get_destination_recommendation(
            query=query.destination_query,
            chat_history=chat_hist_dicts,
            matches=matches,
            flight_context=flight_context
        )

        # --- LATE BINDING: Inject accurate URLs back into the response ---
        # The LLM gives us an iata_code or destination. We safely inject the actual media URLs
        # back into the payload before sending it to the frontend to guarantee zero hallucinations.
        if ai_recommendation.get("recommendations"):
            for rec in ai_recommendation["recommendations"]:
                # Match by IATA code or Destination Name
                match_item = next((m for m in matches if m["iata_code"] == rec.get("iata_code")), None)
                if not match_item:
                    match_item = next((m for m in matches if rec.get("destination", "") in m["destination"]), None)
                
                rec["video_url"] = match_item["video_url"] if match_item else ""
                rec["image_url"] = match_item["image_url"] if match_item else ""
        # ----------------------------------------------------------------

        return {"status": "success", "data": ai_recommendation}

    except Exception as exc:
        print(f"Backend Error: {exc}")
        return {
            "status": "error",
            "message": str(exc),
            "data": {
                "is_new_recommendation": False,
                "chat_response": "I encountered a minor glitch. Could you repeat that for me?",
                "recommendations": []
            }
        }