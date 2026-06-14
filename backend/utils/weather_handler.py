"""Utility for fetching current weather data with simple caching.

Provides an asynchronous function `get_live_weather` that queries the
Open-Meteo API for current weather conditions given latitude and longitude.
Results are cached in‑memory for 10 minutes to reduce API calls.
"""

import httpx
import logging
import time

# In‑memory cache for weather responses (TTL 10 minutes)
_weather_cache: dict[tuple[float, float], tuple[float, dict]] = {}
_cache_ttl = 600  # seconds

logger = logging.getLogger(\"ai_travel_agent\")

async def get_live_weather(lat: float, lon: float) -> dict:
    # Check cache first
    key = (round(lat, 4), round(lon, 4))
    now = time.time()
    if key in _weather_cache:
        cached_ts, cached_data = _weather_cache[key]
        if now - cached_ts < _cache_ttl:
            logger.debug(f"Weather cache hit for {key}")
            return cached_data
        else:
            logger.debug(f"Weather cache expired for {key}")
            del _weather_cache[key]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["temperature_2m", "wind_speed_10m", "rain", "cloud_cover", "relative_humidity_2m", "snowfall"],
        "forecast_days": 1,
    }
    # Simple retry loop (3 attempts)
    for attempt in range(3):
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(url, params=params, timeout=5.0)
            if r.status_code == 200:
                data = r.json()
                current = data.get("current", {})
                result = {
                    "temperature_2m": round(current.get("temperature_2m", 0), 1) if current.get("temperature_2m") is not None else 0,
                    "wind_speed_10m": round(current.get("wind_speed_10m", 0), 1) if current.get("wind_speed_10m") is not None else 0,
                    "rain": round(current.get("rain", 0), 2) if current.get("rain") is not None else 0,
                    "cloud_cover": round(current.get("cloud_cover", 0), 1) if current.get("cloud_cover") is not None else 0,
                    "relative_humidity_2m": round(current.get("relative_humidity_2m", 0), 1) if current.get("relative_humidity_2m") is not None else 0,
                    "snowfall": round(current.get("snowfall", 0), 2) if current.get("snowfall") is not None else 0,
                }
                # Store in cache
                _weather_cache[key] = (now, result)
                return result
            else:
                logger.warning(f"Weather request failed with status {r.status_code}, attempt {attempt+1}")
        except Exception as e:
            logger.error(f"Weather API error on attempt {attempt+1}: {e}")
        # backoff
        await asyncio.sleep(0.5 * (attempt + 1))
    # final fallback
    return {}
