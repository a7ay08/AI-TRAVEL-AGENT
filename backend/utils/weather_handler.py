import httpx

async def get_live_weather(lat: float, lon: float) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["temperature_2m", "wind_speed_10m", "rain", "cloud_cover", "relative_humidity_2m", "snowfall"],
        "forecast_days": 1,
    }
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params, timeout=5.0)
        if r.status_code != 200:
            return {}
        data = r.json()
        current = data.get("current", {})
        
        return {
            "temperature_2m": round(current.get("temperature_2m", 0), 1) if current.get("temperature_2m") is not None else 0,
            "wind_speed_10m": round(current.get("wind_speed_10m", 0), 1) if current.get("wind_speed_10m") is not None else 0,
            "rain": round(current.get("rain", 0), 2) if current.get("rain") is not None else 0,
            "cloud_cover": round(current.get("cloud_cover", 0), 1) if current.get("cloud_cover") is not None else 0,
            "relative_humidity_2m": round(current.get("relative_humidity_2m", 0), 1) if current.get("relative_humidity_2m") is not None else 0,
            "snowfall": round(current.get("snowfall", 0), 2) if current.get("snowfall") is not None else 0
        }
    except Exception as e:
        print(f"Weather API error: {e}")
        return {}
