import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI

class LLMHandler:
    """
    Handles all interactions with the LLM (Llama 3.1).
    Responsible for date extraction (Temporal Intent) and generating recommendations.
    """
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    async def extract_travel_date(self, destination_query: str, chat_history: List[Dict[str, str]]) -> Optional[str]:
        """
        Temporal Intent Extraction:
        Uses a zero-temperature LLM call to extract specific travel dates (YYYY-MM-DD)
        from the user's prompt.
        """
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
            print(f"Error extracting travel date: {e}")
            return None

    async def get_destination_recommendation(self, query: str, chat_history: List[Dict[str, str]], matches: List[Dict[str, str]], flight_context: str) -> Dict[str, Any]:
        """
        Bulletproof Parser & Late Binding prep:
        Generates the core AI response with structured JSON, while hiding large media URLs
        from the prompt to prevent hallucination.
        """
        # Strip URLs to avoid hallucination and token waste
        llm_context = [{"destination": c["destination"], "iata_code": c["iata_code"], "description": c["description"], "tags": c["tags"]} for c in matches]
        
        sys_prompt = (
            "You are the 'AI TRAVEL AGENT', an elite luxury travel assistant.\n"
            f"DATABASE CONTEXT: {json.dumps(llm_context)}\n"
            f"FLIGHT CONTEXT: {flight_context}\n\n"
            "CONVERSATIONAL RULES:\n"
            "1. Check history for specific dates.\n"
            "2. IF NO DATES FOUND: set 'is_new_recommendation': true. Ask for dates in 'chat_response'. Return up to 3 recommendations.\n"
            "3. IF DATES FOUND: set 'is_new_recommendation': false. Provide a beautiful plaintext itinerary in 'chat_response'. Leave recommendations array empty.\n"
            "4. Output ONLY valid JSON brackets {{}}. It is okay to be chatty, but extra text must go in the 'chat_response' string.\n\n"
            "{\n"
            "  \"is_new_recommendation\": true,\n"
            "  \"chat_response\": \"conversational text\",\n"
            "  \"recommendations\": [\n"
            "    {\n"
            "      \"destination\": \"City, Country\",\n"
            "      \"iata_code\": \"XXX\",\n"
            "      \"weather\": \"vibe\",\n"
            "      \"description\": \"4-sentence pitch\",\n"
            "      \"tags\": [\"Tag1\"]\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        msgs = [{"role": "system", "content": sys_prompt}]
        for m in chat_history[-6:]:
            msgs.append({"role": m["role"].replace("ai", "assistant"), "content": m["content"]})
        msgs.append({"role": "user", "content": query})

        try:
            response = await self.client.chat.completions.create(model=self.model, messages=msgs, temperature=0.1)
            raw_output = response.choices[0].message.content.strip()

            # Bulletproof parser: find first { and last }
            start_idx = raw_output.find('{')
            end_idx = raw_output.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                return {"is_new_recommendation": False, "chat_response": raw_output, "recommendations": []}
            
            chatty_pre_text = raw_output[:start_idx].strip()
            
            try:
                ai_recommendation = json.loads(raw_output[start_idx:end_idx+1])
                if chatty_pre_text:
                    existing_response = ai_recommendation.get("chat_response", "")
                    ai_recommendation["chat_response"] = f"{chatty_pre_text}\n\n{existing_response}".strip()
                return ai_recommendation
            except json.JSONDecodeError as e:
                print(f"JSON Parsing Error: {e}")
                return {"is_new_recommendation": False, "chat_response": raw_output, "recommendations": []}
        except Exception as e:
            print(f"LLM Chat Error: {e}")
            raise e