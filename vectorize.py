import os
import uuid
import json
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import datetime as dt
from urllib.parse import urlparse
from dotenv import load_dotenv

# Pinecone client imports
from pinecone import Pinecone, ServerlessSpec

import google.generativeai as genai
from google.generativeai import types

# Load environment variables
load_dotenv()
CONCURRENT_LIMIT = 30

# Configure Gemini API key and model
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite")

# Pinecone setup
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Extract cloud and region from pinecone_env for error message clarity
try:
    cloud, region = pinecone_env.split("-", 1)
except AttributeError:
    print("Error: PINECONE_ENV environment variable is not set correctly. It should be in the format 'cloud-region' (e.g., 'aws-us-east-1').")
    exit(1)
except ValueError:
    print("Error: PINECONE_ENV environment variable is not in the expected 'cloud-region' format.")
    exit(1)

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

# --- FIX APPLIED HERE ---
# Create a set of existing index names for correct lookup
existing_index_names = {idx['name'] for idx in pc.list_indexes()}

# Check if the index exists. If not, print an error and exit.
# The code does NOT create the index; it assumes it already exists.
if INDEX_NAME not in existing_index_names: # Corrected check
    print(f"Error: Pinecone index '{INDEX_NAME}' not found. Please ensure it is created in your Pinecone dashboard with cloud='{cloud}' and region='{region}'.")
    exit(1)
else:
    print(f"Connecting to existing Pinecone index: {INDEX_NAME}.")

# Connect to the index
pinecone_index = pc.Index(INDEX_NAME)


@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 7500) -> List[str]:
    """Split text into chunks roughly of chunk_size characters, preserving paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]

        # Try to cut at paragraph break
        last_break = chunk.rfind('\n\n')
        if last_break > chunk_size * 0.3:
            end = start + last_break
        else:
            # Otherwise cut at sentence end
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, end)

    return chunks

def create_or_get_cache_for_document(doc_text: str, model: str = "gemini-2.0-flash-lite", ttl: int = 5):
    """Create an explicit Gemini cache for the full document text."""
    try:
        cache = genai.caching.CachedContent.create(
            model=model,
            display_name="Full document cache",
            system_instruction="You are an assistant that uses the full document context.",
            contents=[doc_text],
            ttl=dt.timedelta(minutes=ttl),
        )
        print(f"Created cache with name: {cache.name}")
        return cache.name, None
    except Exception as e:
        print(f"Error creating cache: {e}")
        return None, doc_text

async def get_context(full_document: str, chunk: str, cache_name: str) -> Dict[str, str]:
    if cache_name:
        prompt = (
            "Here is the chunk we want to situate within the whole document:\n"
            "<chunk>\n"
            f"{chunk}\n"
            "</chunk>\n\n"
            "Please give a short succinct context to situate this chunk within the overall document "
            "for the purposes of improving search retrieval of the chunk. "
            "Also suggest a suitable title for the chunk.\n\n"
            "Respond ONLY with JSON:\n"
            '{ "title": "...", "context": "..." }'
        )
        try:
            gemini_model = genai.GenerativeModel.from_cached_content(cached_content=cache_name)
            response = gemini_model.generate_content(contents=prompt)
            text = response.text.strip()
            json_part = text.split('```json')[-1].split('```')[0] if '```json' in text else text
            return json.loads(json_part)
        except Exception as e:
            print(f"Gemini context error with cache: {e}")
    # Fallback — embed full doc in prompt
    fallback_prompt = (
        "<document>\n"
        f"{full_document}\n"
        "</document>\n\n"
        "Here is the chunk we want to situate within the whole document:\n"
        "<chunk>\n"
        f"{chunk}\n"
        "</chunk>\n\n"
        "Please give a short succinct context to situate this chunk within the overall document "
        "for the purposes of improving search retrieval of the chunk. "
        "Also suggest a suitable title for the chunk.\n\n"
        "Respond ONLY with JSON:\n"
        '{ "title": "...", "context": "..." }'
    )
    try:
        gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite")
        response = gemini_model.generate_content(fallback_prompt)
        text = response.text.strip()
        json_part = text.split('```json')[-1].split('```')[0] if '```json' in text else text
        return json.loads(json_part)
    except Exception as e:
        print(f"Gemini context fallback error: {e}")
        return {"title": "Untitled", "context": "Could not generate context."}


async def get_embedding(text: str) -> List[float]:
    """Use Gemini embedding API."""
    try:
        embedding_resp = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return embedding_resp["embedding"]
    except Exception as e:
        print(f"Embedding error: {e}")
        return [0.0] * 768

async def process_chunk(chunk: str, chunk_number: int, url: str, full_doc: str, cache_name: str) -> ProcessedChunk:
    extracted = await get_context(full_doc, chunk, cache_name)
    # prepend context to chunk content
    chunk_content_with_context = f"{extracted.get('context', '')} --- {chunk.strip()}"
    embedding = await get_embedding(chunk_content_with_context)
    metadata = {
        "source": "Delaware Business Website",
        "chunk_size": len(chunk_content_with_context),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": url,
        "original_content": chunk.strip()
    }
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted.get("title", "Untitled"),
        summary=extracted.get("context", ""),
        content=chunk_content_with_context,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk_pinecone(chunk: ProcessedChunk):
    try:
        row_id = uuid.uuid4()

        # Insert into Pinecone
        # Store all relevant information in Pinecone's metadata
        pinecone_index.upsert(
            vectors=[
                {
                    "id": str(row_id),
                    "values": chunk.embedding,
                    "metadata": {
                        "url": chunk.url,
                        "chunk_number": chunk.chunk_number,
                        "title": chunk.title,
                        "summary": chunk.summary,
                        "content": chunk.content,
                        **chunk.metadata
                    }
                }
            ]
        )
        print(f"Inserted chunk {chunk.chunk_number} with ID {row_id} into Pinecone.")
    except Exception as e:
        print(f"Pinecone insertion error: {e}")

async def process_and_store_file(path: str, url: str):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Processing {path} from URL {url}")
    # Create explicit cache for full document - DISABLED DUE TO FREE TIER LIMITS
    cache_name = None
    full_doc = content
    # If you uncomment this, ensure your Gemini plan supports caching
    # cache_name, full_doc = create_or_get_cache_for_document(content, model="gemini-2.0-flash-lite", ttl=5)


    chunks = chunk_text(content, chunk_size=7500)
    tasks = [process_chunk(chunk, i, url, full_doc, cache_name) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks)

    insert_tasks = [insert_chunk_pinecone(chunk) for chunk in results]
    await asyncio.gather(*insert_tasks)

async def main():
    output_dir = "output"
    md_files = [f for f in os.listdir(output_dir) if f.endswith(".md")]

    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)

    async def wrapped_process(file):
        async with semaphore:
            readme_path = os.path.join(output_dir, file)
            # Make sure this URL is consistent with what you're scraping
            url = "https://business.delaware.gov/osd/"
            try:
                await process_and_store_file(readme_path, url)
            except Exception as e:
                print(f"[ERROR] Failed to process {readme_path}: {e}")

    tasks = [wrapped_process(f) for f in md_files]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())