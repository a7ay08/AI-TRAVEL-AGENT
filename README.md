# AI Travel Agent

## Overview

**AI Travel Agent** is a FastAPI‑based backend with a minimal React/Next.js frontend that lets users ask a conversational AI for travel‑related information. The system integrates:

- **LLM Handler** – talks to a local or remote Large Language Model (LLM) to classify intent, generate responses and extract travel dates.
- **Search API** – queries live flight data via the SearchAPI provider.
- **Weather Handler** – fetches current weather for a destination.
- **Static data** – POIs, hotels, routes and media are loaded from CSV/JSON files at startup.

The backend stitches these services together and serves JSON endpoints consumed by the UI cards (flight info, weather, points of interest, etc.). The code has been cleaned up, redundant sections removed, and helpful high‑level comments added.

---

## Architecture

```
AI‑TRAVEL‑AGENT/
├─ backend/                 # FastAPI server
│   ├─ main.py             # Entry point, loads data, defines routes
│   ├─ config.py           # Pydantic settings (environment variables)
│   └─ utils/
│       ├─ llm_handler.py  # LLM interaction & intent classification
│       └─ weather_handler.py
├─ frontend/                # Next.js UI (src/app/...)
│   └─ ...                
├─ data/                    # CSV/JSON static assets
└─ README.md                # This document
```

- **`main.py`** – sets up the FastAPI app, loads CSV/JSON data into in‑memory look‑ups, and exposes endpoints for flight info, weather, and POI retrieval.
- **`config.py`** – uses Pydantic's `BaseSettings` to read environment variables (e.g., API keys, data directory).
- **`llm_handler.py`** – wraps LLM calls, implements retry logic, and parses intents.
- **`weather_handler.py`** – simple async wrapper around a weather API.
- **Frontend** – consumes the JSON endpoints and displays cards with dynamic UI effects.

---

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/a7ay08/AI-TRAVEL-AGENT.git
   cd AI-TRAVEL-AGENT
   ```
2. **Create a virtual environment** (Python 3.11+ recommended)
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate   # Windows PowerShell
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables**
   - Copy `.env.example` to `.env` (if provided) or create a new `.env` file.
   - Set the following keys:
     ```
     SEARCHAPI_KEY=your_searchapi_key
     LLM_API_KEY=your_llm_key
     LLM_BASE_URL=https://your-llm-endpoint
     ...
     ```
5. **Run the backend**
   ```bash
   python -m uvicorn main:app --reload
   ```
   The API will be reachable at `http://127.0.0.1:8000`.
6. **Run the frontend** (optional, requires Node.js)
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   The UI will be available at `http://localhost:3000`.

---

## Deployment Guide

- **Docker** – A Dockerfile can be added (not included here). Build with `docker build -t ai-travel-agent .` and run `docker run -p 8000:8000 ai-travel-agent`.
- **Cloud** – Deploy the FastAPI app to any platform that supports ASGI (e.g., Azure App Service, Render, Fly.io). Ensure the environment variables are set in the hosting environment.
- **Static Data** – Keep the `data/` folder in the container or mount it as a volume so CSV/JSON files are accessible.

---

## Usage Example

```bash
# Get live flight info for a destination (e.g., LHR)
curl -X GET "http://127.0.0.1:8000/flight-info?destination=LHR"
```

The response includes flight status, price, weather, and relevant POIs.

---

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome`).
3. Make your changes and ensure the code follows the existing style.
4. Open a Pull Request.

---

## License

NA
