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
CHROMA_DB_PATH = "./chroma_db"
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
    
    def fetch_free_dictionary_api(self, word: str) -> List[str]:
        """Fetch definitions from Free Dictionary API (free, no key required)"""
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
                
                logger.info(f"Found {len(definitions)} definitions for {word}")
                return definitions[:8]  # Return first 8 definitions
                
            elif response.status_code == 404:
                logger.warning(f"Word '{word}' not found in Free Dictionary API")
                return []
            else:
                logger.warning(f"Free Dictionary API returned status: {response.status_code}")
                return []
                
        except Exception as e:
            logger.warning(f"Free Dictionary API failed: {e}")
            return []
    
    def fetch_urbandictionary(self, word: str) -> List[str]:
        """Fetch definitions from Urban Dictionary (good for modern/slang terms)"""
        try:
            url = f"https://api.urbandictionary.com/v0/define?term={word}"
            logger.info(f"Fetching definition from Urban Dictionary for: {word}")
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                definitions = []
                
                for item in data.get('list', [])[:3]:  # Get first 3 definitions
                    def_text = item.get('definition', '').strip()
                    # Clean up Urban Dictionary formatting
                    def_text = re.sub(r'\[.*?\]', '', def_text)  # Remove [word] links
                    if def_text and len(def_text) > 10:
                        definitions.append(def_text)
                
                logger.info(f"Found {len(definitions)} definitions from Urban Dictionary")
                return definitions
            else:
                return []
                
        except Exception as e:
            logger.warning(f"Urban Dictionary failed: {e}")
            return []
    
    def fetch_merriam_webster(self, word: str) -> List[str]:
        """Fetch definitions from Merriam-Webster (web scraping fallback)"""
        try:
            url = f"https://www.merriam-webster.com/dictionary/{word}"
            logger.info(f"Fetching definition from Merriam-Webster for: {word}")
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                # Simple regex to extract definitions
                html = response.text
                definitions = []
                
                # Look for definition patterns
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
                
                logger.info(f"Found {len(definitions)} definitions from Merriam-Webster")
                return definitions[:5]
            else:
                return []
                
        except Exception as e:
            logger.warning(f"Merriam-Webster failed: {e}")
            return []
    
    def get_definitions(self, word: str) -> List[str]:
        """Get definitions from multiple reliable sources"""
        logger.info(f"Looking up definitions for: {word}")
        
        # Try multiple sources
        sources = [
            self.fetch_free_dictionary_api(word),  # Primary source
            self.fetch_urbandictionary(word),      # For modern terms
            self.fetch_merriam_webster(word)       # Fallback
        ]
        
        all_definitions = []
        seen_definitions = set()
        
        for definitions in sources:
            for definition in definitions:
                # Clean and deduplicate
                clean_def = definition.strip()
                if (len(clean_def) > 10 and 
                    clean_def not in seen_definitions and 
                    not clean_def.startswith('http')):
                    seen_definitions.add(clean_def)
                    all_definitions.append(clean_def)
        
        # Remove near-duplicates
        unique_definitions = []
        for definition in all_definitions:
            is_duplicate = False
            for existing in unique_definitions:
                similarity = len(set(definition.lower().split()) & set(existing.lower().split()))
                if similarity > 3:  # If more than 3 words overlap
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_definitions.append(definition)
        
        logger.info(f"Total unique definitions found: {len(unique_definitions)}")
        return unique_definitions[:10]  # Return up to 10 definitions

# Initialize dictionary service
dict_service = DictionaryService()

def create_local_summary(text: str, max_sentences: int = 8) -> str:
    """Create a simple summary by extracting key sentences"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return "Unable to generate summary from the provided text."
    
    # Score sentences (simple heuristic)
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

@app.post("/upload")
async def upload_file(file: UploadFile):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are accepted.")
    
    try:
        file_content = await file.read()
        
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        
        if not text.strip():
            raise HTTPException(400, "PDF contains no readable text")
        
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(400, "Could not extract meaningful text from PDF")
        
        embeddings = embedder.encode(chunks)
        file_id = str(uuid.uuid4())

        document_collection.add(
            ids=[f"{file_id}_{i}" for i in range(len(chunks))],
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=[{"file_id": file_id, "filename": file.filename} for _ in chunks]
        )

        return {
            "file_id": file_id,
            "filename": file.filename,
            "status": "success",
            "chunks": len(chunks)
        }
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, f"Failed to process PDF: {str(e)}")

@app.post("/ask")
async def ask_question(question: str = Form(...), file_id: Optional[str] = Form(None)):
    try:
        if not question.strip():
            raise HTTPException(400, "Question cannot be empty")
            
        question_embedding = embedder.encode(question).tolist()
        
        query_params = {"query_embeddings": [question_embedding], "n_results": 3}
        if file_id:
            query_params["where"] = {"file_id": file_id}
            
        results = document_collection.query(
            **query_params,
            include=["documents", "distances", "metadatas"]
        )

        answers = []
        if results['documents']:
            for doc, dist, meta in zip(results['documents'][0], results['distances'][0], results['metadatas'][0]):
                answers.append({
                    "text": doc,
                    "similarity": float(1 - dist),
                    "source": "document",
                    "filename": meta.get("filename", "Unknown")
                })

        # For definition questions, use dictionary service
        if not answers and ("definition" in question.lower() or "define" in question.lower() or "what is" in question.lower()):
            # Extract word from question
            words = re.findall(r"['\"](.*?)['\"]", question)
            if not words:
                # Try to extract word from "what is X" pattern
                match = re.search(r'(?:what is|define|definition of)\s+([a-zA-Z]+)', question.lower())
                if match:
                    words = [match.group(1)]
            
            if words:
                word = words[0].lower()
                definitions = dict_service.get_definitions(word)
                if definitions:
                    answers.append({
                        "text": "\n".join([f"{i+1}. {defn}" for i, defn in enumerate(definitions)]),
                        "similarity": 1.0,
                        "source": "dictionary",
                        "filename": f"Definition of {word}"
                    })

        if not answers:
            answers.append({
                "text": "No relevant information found in the documents. Try rephrasing your question or check the dictionary feature for definitions.",
                "similarity": 0.0,
                "source": "system",
                "filename": "Help"
            })

        return {"answers": answers}
    except Exception as e:
        logger.error(f"Question error: {e}")
        raise HTTPException(500, f"Failed to get answer: {str(e)}")

@app.get("/dictionary/{word}")
async def dictionary_lookup(word: str):
    try:
        if not word.strip():
            raise HTTPException(400, "Word cannot be empty")
        
        logger.info(f"Dictionary lookup for: {word}")
        
        # Get definitions from reliable sources
        definitions = dict_service.get_definitions(word)
        
        if definitions:
            return {
                "word": word,
                "definitions": definitions,
                "source": "Free Dictionary API + Merriam-Webster",
                "sources_used": len(definitions)
            }
        else:
            # Provide helpful fallback for common words
            common_words = {
                "document": [
                    "A written or printed record that provides information or serves as an official record",
                    "A computer file containing text, images, or other data",
                    "To record or report something in detail, typically in writing"
                ],
                "computer": [
                    "An electronic device for storing and processing data, typically in binary form",
                    "A machine that can be instructed to carry out sequences of arithmetic or logical operations automatically"
                ],
                "python": [
                    "A high-level programming language known for its readability and versatility, used for web development, data analysis, and artificial intelligence",
                    "A large constricting snake found in tropical regions of Africa, Asia, and Australia"
                ],
                "ai": [
                    "Artificial Intelligence: The simulation of human intelligence processes by machines, especially computer systems",
                    "The capability of a machine to imitate intelligent human behavior and perform tasks that typically require human intelligence"
                ],
                "cat": [
                    "A small domesticated carnivorous mammal with soft fur, a short snout, and retractable claws",
                    "Any member of the family Felidae, including lions, tigers, and leopards"
                ],
                "dog": [
                    "A domesticated carnivorous mammal that typically has a long snout, an acute sense of smell, and a barking, howling, or whining voice",
                    "A member of the canine family, often kept as a pet or used for hunting, guarding, or assisting"
                ]
            }
            
            if word.lower() in common_words:
                return {
                    "word": word,
                    "definitions": common_words[word.lower()],
                    "source": "Common Definitions",
                    "sources_used": 1
                }
            else:
                return {
                    "word": word,
                    "definitions": [
                        f"No definitions found for '{word}'.",
                        "This might be because:",
                        "• The word is very uncommon or specialized",
                        "• There might be a temporary network issue",
                        "• The word might be misspelled",
                        "Please try checking the spelling or try a different word."
                    ],
                    "source": "No definitions available",
                    "sources_used": 0
                }
            
    except Exception as e:
        logger.error(f"Dictionary error: {e}")
        return {
            "word": word,
            "definitions": [f"Error retrieving definitions: {str(e)}"],
            "source": "Error",
            "sources_used": 0
        }

@app.get("/summarize/{file_id}")
async def summarize_document(file_id: str):
    try:
        results = document_collection.get(
            where={"file_id": file_id},
            include=["documents", "metadatas"]
        )
        
        if not results['documents']:
            raise HTTPException(404, "Document not found")
        
        full_text = " ".join(results['documents'])
        filename = results['metadatas'][0]['filename'] if results['metadatas'] else "Unknown"
        
        summary = create_local_summary(full_text)
        
        return {
            "file_id": file_id,
            "filename": filename,
            "summary": summary.strip(),
            "source": "Local Processing"
        }
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(500, f"Failed to summarize document: {str(e)}")

@app.get("/files")
async def list_files():
    try:
        results = document_collection.get()
        files = {}
        for metadata in results['metadatas']:
            file_id = metadata['file_id']
            if file_id not in files:
                files[file_id] = metadata['filename']
        
        return {"files": [{"id": k, "name": v} for k, v in files.items()]}
    except Exception as e:
        logger.error(f"File list error: {e}")
        return {"files": []}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "dictionary_service": "active",
        "embedding_model": EMBEDDING_MODEL,
        "message": "Dictionary service is running with Free Dictionary API"
    }

@app.get("/test-dictionary/{word}")
async def test_dictionary(word: str):
    """Test the dictionary service"""
    definitions = dict_service.get_definitions(word)
    return {
        "word": word,
        "definitions_found": len(definitions),
        "definitions": definitions,
        "status": "success" if definitions else "no_definitions"
    }
if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port)

