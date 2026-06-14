"""LLM Handler module.

Provides asynchronous interactions with the LLM (LLama 3.1 via LM Studio).
Includes intent classification, date extraction, and generation of JSON responses for the backend.
"""

import json
import re
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI

logger = logging.getLogger("ai_travel_agent")


class LLMHandler:
    """
    Handles all interactions with the LLM (Llama 3.1 via LM Studio).
    Responsible for date extraction and generating the chat_response text.
    NOTE: The backend (main.py) is now the source of truth for all structured
    recommendation data. The LLM only needs to produce the conversational text.
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        # Ensure base_url ends with /v1 for local/LM Studio endpoints if not specified
        if ("127.0.0.1" in base_url or "localhost" in base_url) and not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model  = model

    # Intent Classification
    async def classify_intent(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """Classify the user's intent using the LLM for natural, flexible conversation."""
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-3:]])
        prompt = f"""You are an intent classification system for a travel assistant.
Analyze the user's latest query and the recent conversation history. Classify the intent into exactly one of these categories:
- 'flight': The user wants flight details, flight costs, flight search, or ticket options.
- 'hotel': The user wants hotel recommendations, places to stay, lodging, or accommodation options.
- 'destination': The user is asking for new travel recommendations, suggesting destinations, or discovering where to go based on vibe/preferences.
- 'conversational': The user is asking general questions, seeking local recommendations (e.g., restaurants, food, attractions, things to do, culture), planning itineraries, asking about weather, or doing general small talk / chit-chat.

History:
{history_text}

Latest Query:
{query}

Classification (output ONLY one word: flight, hotel, destination, or conversational):"""
        # Retry up to 3 attempts
        for attempt in range(3):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                )
                result = response.choices[0].message.content.strip().lower()
                for category in ["flight", "hotel", "destination", "conversational"]:
                    if category in result:
                        return category
                return "conversational"
            except Exception as e:
                logger.error(f"LLM intent classification error on attempt {attempt+1}: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))
        return "conversational"


    # Temporal Intent Extraction
    async def extract_travel_date(self, destination_query: str,
                                   chat_history: List[Dict[str, str]]) -> Optional[str]:
        """Extract a specific travel date (YYYY-MM-DD) from the user's query."""
        current_date_str = datetime.now().strftime("%B %d, %Y")
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-3:]])
        prompt = f"""You are a precise date extraction tool. Today is {current_date_str}.
Analyze the user's latest query and history to find their intended travel date.
Format: YYYY-MM-DD. If no specific date is found, output: NONE.

History:
{history_text}

Latest Query:
{destination_query}

Result:"""
        # Retry up to 3 attempts
        for attempt in range(3):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=15,
                )
                result = response.choices[0].message.content.strip()
                date_match = re.search(r"\d{4}-\d{2}-\d{2}", result)
                return date_match.group(0) if date_match else None
            except Exception as e:
                logger.error(f"Date extraction error on attempt {attempt+1}: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))
        return None

    # Main Recommendation / Chat Response
    async def get_destination_recommendation(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
        matches: List[Dict],
        flight_context: str,
        intent: str = "destination",
        preferences: Optional[Dict] = None,
        poi_context: str = "",
        hotel_context: str = "",
        weather_context: str = "",
    ) -> Dict[str, Any]:
        """
        Generates the AI chat_response text.
        The backend constructs all structured recommendation data independently;
        the LLM only needs to return a valid JSON shell with chat_response.
        """
        prefs = preferences or {}
        cities_str = ", ".join([m.get("city", "") for m in matches]) or "various destinations"

        if intent == "flight":
            intent_instruction = (
                f"The user is asking about FLIGHTS. Describe the Etihad flight options using "
                f"FLIGHT CONTEXT. Mention the route, price, duration, and travel date. "
                f"Be concise and exciting. Destination(s): {cities_str}."
            )
            extra_ctx = f"FLIGHT CONTEXT: {flight_context}"

        elif intent == "hotel":
            # Pydantic Models
            intent_instruction = (
                f"The user is asking about HOTELS in {cities_str}. "
                f"Write a warm, helpful intro (2-3 sentences) about accommodations in {cities_str}. "
                f"Briefly highlight the variety of options available. "
                f"Do NOT list the hotels yourself — the UI will display them below your text."
            )
            extra_ctx = f"AVAILABLE HOTELS:\n{hotel_context}" if hotel_context else ""

        elif intent == "conversational":
            if not matches:
                intent_instruction = (
                    "The user is asking a general travel question, making small talk, or asking about general travel advice. "
                    "Provide a warm, premium, helpful, and luxury travel agent response. "
                    "Do NOT list specific places or suggest booking flights or hotels unless asked."
                )
                extra_ctx = "No specific destination matched. General travel conversation."
            else:
                intent_instruction = (
                    f"The user is asking a conversational travel question about {cities_str}. "
                    f"Use the PLACE DATA below to give a helpful, specific, and enthusiastic answer. "
                    f"Format your response clearly: name each place, its category, and a one-line description. "
                    f"Be concise but informative. Do NOT suggest booking flights or hotels."
                )
                extra_ctx = f"PLACE DATA for {cities_str}:\n{poi_context}" if poi_context else \
                            f"Answer based on general knowledge about {cities_str}."

        else:  # destination
            intent_instruction = (
                f"The user is asking about DESTINATIONS. Introduce these destination(s): {cities_str}. "
                f"One vivid sentence per destination. Highlight why it suits a "
                f"{prefs.get('traveler_type', 'traveler')} looking for a "
                f"{prefs.get('vibe', 'great')} {prefs.get('duration', 'trip')}."
            )
            extra_ctx = f"FLIGHT CONTEXT: {flight_context}"

        sys_prompt = (
            "You are the 'AI TRAVEL AGENT', an elite luxury travel assistant powered by Etihad Airways.\n"
            f"USER PREFERENCES: Traveler={prefs.get('traveler_type','?')}, "
            f"Vibe={prefs.get('vibe','?')}, Duration={prefs.get('duration','?')}, "
            f"Trip Type={prefs.get('trip_type','?')}.\n"
            f"{extra_ctx}\n\n"
            f"TASK: {intent_instruction}\n\n"
            "OUTPUT RULES:\n"
            "1. Output ONLY valid JSON with exactly these 3 keys: "
            "'is_new_recommendation', 'chat_response', 'recommendations'.\n"
            "2. Set 'is_new_recommendation': true.\n"
            "3. Set 'recommendations': [] — the backend fills this in.\n"
            "4. In 'chat_response', write your response then add EXACTLY 3 follow-up questions:\n\n"
            "You might also ask:\n"
            "- [Question 1]\n"
            "- [Question 2]\n"
            "- [Question 3]\n\n"
            "5. Output ONLY the JSON object. No text before or after it.\n\n"
            "{\n"
            "  \"is_new_recommendation\": true,\n"
            "  \"chat_response\": \"Your response...\\n\\nYou might also ask:\\n- Q1\\n- Q2\\n- Q3\",\n"
            "  \"recommendations\": []\n"
            "}"
        )


        msgs = [{"role": "system", "content": sys_prompt}]
        for m in chat_history[-6:]:
            msgs.append({"role": m["role"].replace("ai", "assistant"), "content": m["content"]})
        msgs.append({"role": "user", "content": query})

        try:
            response = await self.client.chat.completions.create(
                model=self.model, messages=msgs, temperature=0.3, max_tokens=600
            )
            raw = response.choices[0].message.content.strip()

            # Bulletproof JSON parser
            start = raw.find('{')
            end   = raw.rfind('}')
            if start == -1 or end == -1:
                return {"is_new_recommendation": True, "chat_response": raw, "recommendations": []}

            pre_text = raw[:start].strip()
            try:
                parsed = json.loads(raw[start:end + 1], strict=False)
                if pre_text:
                    parsed["chat_response"] = f"{pre_text}\n\n{parsed.get('chat_response', '')}".strip()
                return parsed
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}. Trying bulletproof regex and manual slicing fallbacks...")
                # 1. Regex fallback
                match = re.search(r'"chat_response"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
                if match:
                    chat_resp = match.group(1)
                    chat_resp = chat_resp.replace('\\n', '\n').replace('\\"', '"')
                    return {"is_new_recommendation": True, "chat_response": chat_resp, "recommendations": []}
                
                # 2. Manual slicing fallback (completely bulletproof against literal newlines and unescaped quotes)
                idx = raw.find('"chat_response"')
                if idx != -1:
                    colon_idx = raw.find(':', idx)
                    if colon_idx != -1:
                        start_quote = raw.find('"', colon_idx + 1)
                        if start_quote != -1:
                            # Scan forward to find closing quote before the recommendations key or closing brace
                            end_quote = -1
                            for i in range(start_quote + 1, len(raw)):
                                if raw[i] == '"':
                                    rest = raw[i+1:].strip()
                                    if rest.startswith(',') or rest.startswith('}') or rest == '':
                                        end_quote = i
                                        break
                            if end_quote != -1:
                                content = raw[start_quote + 1:end_quote].strip()
                                return {"is_new_recommendation": True, "chat_response": content, "recommendations": []}
                
                # 3. Ultimate fallback: strip brackets and output cleaned raw text
                clean_raw = raw.replace('{', '').replace('}', '').replace('"is_new_recommendation": true,', '').replace('"is_new_recommendation": false,', '').strip()
                return {"is_new_recommendation": True, "chat_response": clean_raw, "recommendations": []}

        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise e
