import os
import uuid
import pdfplumber
import chromadb
from fastapi import FastAPI, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from dotenv import load_dotenv
import logging
from pathlib import Path
import io
import re
import requests
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
CHROMA_DB_PATH = "/mnt/data/chroma_db"  # Use persistent storage for HF Spaces
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
CHUNK_SIZE = 500

# Create necessary directories
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

app = FastAPI(title="DOC AI", description="AI-powered document analysis tool")

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks while preserving sentence boundaries"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# Initialize services
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    document_collection = chroma_client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Services initialized successfully")
except Exception as e:
    logger.error(f"Service Init Error: {e}")
    raise

class DictionaryService:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    # All dictionary methods remain unchanged...
    def fetch_free_dictionary_api(self, word: str) -> List[str]:
        try:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
            logger.info(f"Fetching definition from Free Dictionary API for: {word}")
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                definitions = []
                if isinstance(data, list) and len(data) > 0:
                    for meaning in data[0].get('meanings', []):
                        part_of_speech = meaning.get('partOfSpeech', '')
                        for definition in meaning.get('definitions', []):
                            def_text = definition.get('definition', '').strip()
                            if def_text:
                                if part_of_speech:
                                    definitions.append(f"({part_of_speech}) {def_text}")
                                else:
                                    definitions.append(def_text)
                return definitions[:8]
            elif response.status_code == 404:
                return []
            else:
                return []
        except Exception as e:
            logger.warning(f"Free Dictionary API failed: {e}")
            return []

    def fetch_urbandictionary(self, word: str) -> List[str]:
        try:
            url = f"https://api.urbandictionary.com/v0/define?term={word}"
            logger.info(f"Fetching definition from Urban Dictionary for: {word}")
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                definitions = []
                for item in data.get('list', [])[:3]:
                    def_text = item.get('definition', '').strip()
                    def_text = re.sub(r'\[.*?\]', '', def_text)
                    if def_text and len(def_text) > 10:
                        definitions.append(def_text)
                return definitions
            else:
                return []
        except Exception as e:
            logger.warning(f"Urban Dictionary failed: {e}")
            return []

    def fetch_merriam_webster(self, word: str) -> List[str]:
        try:
            url = f"https://www.merriam-webster.com/dictionary/{word}"
            logger.info(f"Fetching definition from Merriam-Webster for: {word}")
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                html = response.text
                definitions = []
                patterns = [
                    r'<span class="dt-text">([^<]+)</span>',
                    r'<p class="definition-inner-item"[^>]*>([^<]+)</p>',
                    r'<div class="vg">([^<]+)</div>'
                ]
                for pattern in patterns:
                    matches = re.findall(pattern, html)
                    for match in matches:
                        def_text = match.strip()
                        if def_text and len(def_text) > 10 and word.lower() in def_text.lower():
                            definitions.append(def_text)
                return definitions[:5]
            else:
                return []
        except Exception as e:
            logger.warning(f"Merriam-Webster failed: {e}")
            return []

    def get_definitions(self, word: str) -> List[str]:
        sources = [
            self.fetch_free_dictionary_api(word),
            self.fetch_urbandictionary(word),
            self.fetch_merriam_webster(word)
        ]
        all_definitions = []
        seen_definitions = set()
        for definitions in sources:
            for definition in definitions:
                clean_def = definition.strip()
                if (len(clean_def) > 10 and clean_def not in seen_definitions and not clean_def.startswith('http')):
                    seen_definitions.add(clean_def)
                    all_definitions.append(clean_def)
        unique_definitions = []
        for definition in all_definitions:
            is_duplicate = False
            for existing in unique_definitions:
                similarity = len(set(definition.lower().split()) & set(existing.lower().split()))
                if similarity > 3:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_definitions.append(definition)
        return unique_definitions[:10]

dict_service = DictionaryService()

def create_local_summary(text: str, max_sentences: int = 8) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if not sentences:
        return "Unable to generate summary from the provided text."
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        score = len(sentence)
        if i < len(sentences) * 0.3:
            score *= 1.2
        key_terms = ['summary', 'conclusion', 'important', 'key', 'main', 'primary', 'result']
        if any(term in sentence.lower() for term in key_terms):
            score *= 1.5
        scored_sentences.append((sentence, score))
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s[0] for s in scored_sentences[:max_sentences]]
    final_sentences = []
    for sentence in sentences:
        if sentence in top_sentences:
            final_sentences.append(sentence)
    summary = " ".join(final_sentences)
    return summary

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# All remaining routes (/upload, /ask, /dictionary, /summarize, /files, /health, /test-dictionary)
# remain exactly as in your original file

# ----

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))  # Use HF Space port
    uvicorn.run("backend:app", host="0.0.0.0", port=port)
