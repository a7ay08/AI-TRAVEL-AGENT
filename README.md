🌍 AI TRAVEL AGENT
An elite, luxury AI travel assistant. This prototype leverages a Local-First LLM strategy and Vector RAG (Retrieval-Augmented Generation) to provide highly personalized destination discovery, real-time flight data, and immersive media previews.

📋 Table of Contents
Project Architecture

Core Features

Technology Stack

Getting Started

Project Structure

🏗 Project Architecture
The system operates on a Decoupled RAG Architecture:

Frontend: A Next.js 14 (App Router) interface styled with Tailwind CSS for a "Premium Dark Mode" aesthetic.

Backend: A FastAPI (Python) server handling orchestration, intent extraction, and data retrieval.

Brain: Llama-3.1-8B-Instruct running locally via LM Studio to ensure privacy and low latency.

Memory: Pinecone Vector DB storing ~9,000 flight routes, Points of Interest (POIs), and immersive media links.

✨ Features
1. In-Chat Personalization Widget
Unlike static forms, the bot intercepts your initial query and presents a sleek, interactive widget directly in the chat stream to capture:

Origin: Automatic and manual departure detection.

Traveler Type: Solo, Couple, Family, or Group.

Trip Vibe: Relaxation, Adventure, Honeymoon, or Culture.

Duration: Weekend, 1 Week, or 2+ Weeks.

2. Rich Multi-Recommendation Feed
Generates up to 3 side-scrolling "Destination Cards" based on semantic matching.

Immersive Media: Real-time vertical video previews (scraped from Pexels) or high-res city photography.

AI Pitches: 4-sentence luxury pitches generated dynamically by Llama 3.1.

Weather Intelligence: Integrated weather "vibes" for every suggested location.

3. Live Intent & Temporal Extraction
Temporal Brain: Converts conversational dates (e.g., "next Friday" or "June 15th") into standardized ISO formats using a zero-temperature LLM pass.

Real-Time Flights: Fetches live pricing and airline data via SearchAPI (Google Flights) once dates are confirmed.

4. Hybrid Chat Flow
Memory-Aware: Remembers your selected destination and preferences for follow-up questions.

Late Binding: Re-injects accurate database metadata (URLs/IATA codes) into LLM responses to prevent hallucinations.

🛠️ Technology Stack
Frontend: Next.js 14, TypeScript, Tailwind CSS, Framer Motion (Animations).

Backend: FastAPI (Asynchronous Python), Uvicorn.

Vector Database: Pinecone (Serverless).

LLM Engine: LM Studio (Local Server) hosting meta-llama-3.1-8b-instruct.

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (Running locally).

APIs: SearchAPI.io (Google Flights & Hotels).

🚀 Getting Started
Prerequisites
Node.js 18+ & NPM

Python 3.10+

LM Studio installed with Llama-3.1-8B-Instruct loaded.

Installation
Clone & Setup Backend

Bash
cd backend
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
Configure Environment
Create a .env file in the backend/ directory:

Code snippet
PINECONE_API_KEY=your_key
SEARCHAPI_KEY=your_key
LM_STUDIO_API_KEY=lm-studio
Setup Frontend

Bash
cd ../frontend
npm install
Run the App

Terminal 1: Open LM Studio and start the Local Server on port 1234.

Terminal 2 (Backend): uvicorn main:app --reload

Terminal 3 (Frontend): npm run dev

📁 Project Structure
Plaintext
AI-TRAVEL-AGENT/
├── frontend/                  # Next.js Application
│   ├── src/app/page.tsx       # Chat UI & Widget Logic
│   └── tailwind.config.ts     # Luxury Gold/Navy Theme
├── backend/                   # FastAPI Application
│   ├── main.py                # RAG Logic & Intent Extraction
│   └── .env                   # Secrets
├── data/                      # Local datasets (CSV/JSON)
└── scripts/                   # Ingestion & Scraping scripts
📜 License
Internal Prototype - Proprietary logic for Airial-Clone project.
