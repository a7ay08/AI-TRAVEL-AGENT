#standalone utility script designed to verify that the LLM connection is functioning properly. It tests the LLMHandler's ability to classify chat intents and extract dates without having to run the entire FastAPI server.
import asyncio
import httpx
from openai import AsyncOpenAI

async def test():
    base_url = "http://127.0.0.1:1234/v1"
    model = "meta-llama-3.1-8b-instruct"
    
    # Test direct http request
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.0,
                    "max_tokens": 10
                },
                timeout=10.0
            )
            print("Direct HTTP Response Status:", r.status_code)
            print("Direct HTTP Response Body:", r.text)
        except Exception as e:
            print("Direct HTTP Error:", e)

    # Test OpenAI Client
    try:
        client = AsyncOpenAI(base_url=base_url, api_key="lm-studio")
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.0,
            max_tokens=10
        )
        print("OpenAI Client Response:", resp)
        print("OpenAI Choices:", getattr(resp, "choices", None))
    except Exception as e:
        print("OpenAI Client Error:", e)

if __name__ == "__main__":
    asyncio.run(test())
