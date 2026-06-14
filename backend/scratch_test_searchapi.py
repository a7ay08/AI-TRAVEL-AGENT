#t is a testing script used to isolate and validate integrations with the external SearchAPI endpoint. It verifies if API keys are correctly configured and fetches a raw sample of live flight data from the provider.
import asyncio
import os
from dotenv import load_dotenv

# Load env variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

from main import get_live_flight_info, searchapi_key, Settings

async def test_searchapi():
    print("Loaded Settings searchapi_key:", Settings().SEARCHAPI_KEY)
    print("Loaded module searchapi_key:", searchapi_key)
    
    # Try searching flight from AUH to LHR (London Heathrow)
    dest = "LHR"
    print(f"Calling get_live_flight_info for destination {dest}...")
    res = await get_live_flight_info(dest)
    print("Result:")
    import pprint
    pprint.pprint(res)

if __name__ == "__main__":
    asyncio.run(test_searchapi())
