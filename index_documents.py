"""Document vectorization module for indexing PDF and DOCX files.

Extracts text from documents, splits into chunks using configurable
strategies, generates embeddings via Google Gemini API, and stores
results in PostgreSQL with pgvector.
"""

import argparse
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

import psycopg2
from google import genai
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from PyPDF2 import PdfReader

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
POSTGRES_URL: str = os.getenv("POSTGRES_URL", "")

EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_DIMENSION: int = 3072

CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100

GEMINI_BATCH_SIZE: int = 100
GEMINI_RATE_LIMIT_PAUSE: float = 1.0


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text_from_pdf(file_path: str) -> str:
    """Extract plain text from a PDF file.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        Concatenated text from all pages.
    """
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_text_from_docx(file_path: str) -> str:
    """Extract plain text from a DOCX file.

    Args:
        file_path: Absolute or relative path to the DOCX file.

    Returns:
        Concatenated text from all paragraphs.
    """
    from docx import Document

    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def extract_text(file_path: str) -> str:
    """Extract text from a PDF or DOCX file based on its extension.

    Args:
        file_path: Path to the document.

    Returns:
        Clean extracted text.

    Raises:
        ValueError: If the file extension is not .pdf or .docx.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    if ext == ".docx":
        return extract_text_from_docx(file_path)
    raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .docx")


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------


def chunk_fixed_size(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into fixed-size chunks with overlap, preserving whole words.

    The cut point snaps back to the nearest space so that words are never
    split in the middle.

    Args:
        text: The source text to split.
        size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space_pos = text.rfind(" ", start, end)
            if space_pos > start:
                end = space_pos
        chunks.append(text[start:end].strip())
        start = end - overlap if end < len(text) else len(text)
    return [c for c in chunks if c]


def chunk_by_sentences(text: str) -> list[str]:
    """Split text into chunks where each chunk is one sentence.

    Uses a simple regex-based sentence boundary detector.

    Args:
        text: The source text to split.

    Returns:
        List of sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_paragraphs(text: str) -> list[str]:
    """Split text into chunks where each chunk is one paragraph.

    Paragraphs are separated by one or more blank lines.

    Args:
        text: The source text to split.

    Returns:
        List of paragraphs.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


CHUNKING_STRATEGIES: dict[str, Callable[..., list[str]]] = {
    "fixed": chunk_fixed_size,
    "sentence": chunk_by_sentences,
    "paragraph": chunk_by_paragraphs,
}


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------


def generate_embeddings(chunks: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text chunks using Google Gemini API.

    Processes chunks in batches to respect API rate limits.

    Args:
        chunks: List of text strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    client = genai.Client(api_key=GEMINI_API_KEY)

    embeddings: list[list[float]] = []
    for i in range(0, len(chunks), GEMINI_BATCH_SIZE):
        batch = chunks[i : i + GEMINI_BATCH_SIZE]
        result = client.models.embed_content(model=EMBEDDING_MODEL, contents=batch)
        for emb in result.embeddings:
            embeddings.append(list(emb.values))
        if i + GEMINI_BATCH_SIZE < len(chunks):
            time.sleep(GEMINI_RATE_LIMIT_PAUSE)

    return embeddings


# ---------------------------------------------------------------------------
# PostgreSQL storage
# ---------------------------------------------------------------------------


def get_connection() -> psycopg2.extensions.connection:
    """Create a PostgreSQL connection using the POSTGRES_URL env variable.

    Returns:
        A psycopg2 connection object.
    """
    return psycopg2.connect(POSTGRES_URL)


def init_db(conn: psycopg2.extensions.connection) -> None:
    """Initialize the pgvector extension, register it, and create the table.

    Must be called before any vector operations on the connection.

    Args:
        conn: An open psycopg2 connection.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                embedding VECTOR({EMBEDDING_DIMENSION}) NOT NULL,
                filename VARCHAR(255) NOT NULL,
                split_strategy VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
    conn.commit()
    register_vector(conn)


def store_chunks(
    conn: psycopg2.extensions.connection,
    chunks: list[str],
    embeddings: list[list[float]],
    filename: str,
    strategy: str,
) -> int:
    """Insert chunks and their embeddings into the database.

    Args:
        conn: An open psycopg2 connection.
        chunks: List of text chunks.
        embeddings: Corresponding embedding vectors.
        filename: Original document filename.
        strategy: The chunking strategy that was used.

    Returns:
        Number of rows inserted.
    """
    with conn.cursor() as cur:
        for chunk_text, embedding in zip(chunks, embeddings):
            cur.execute(
                """
                INSERT INTO document_chunks (chunk_text, embedding, filename, split_strategy)
                VALUES (%s, %s, %s, %s)
                """,
                (chunk_text, embedding, filename, strategy),
            )
    conn.commit()
    return len(chunks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def validate_config() -> None:
    """Validate that required environment variables are set.

    Raises:
        SystemExit: If any required variable is missing.
    """
    if not GEMINI_API_KEY:
        sys.exit("Error: GEMINI_API_KEY is not set. Add it to your .env file.")
    if not POSTGRES_URL:
        sys.exit("Error: POSTGRES_URL is not set. Add it to your .env file.")


def main() -> None:
    """CLI entry point for indexing documents."""
    parser = argparse.ArgumentParser(
        description="Index a PDF or DOCX document into PostgreSQL with vector embeddings."
    )
    parser.add_argument("--file", required=True, help="Path to the PDF or DOCX file.")
    parser.add_argument(
        "--strategy",
        choices=list(CHUNKING_STRATEGIES.keys()),
        default="fixed",
        help="Text chunking strategy (default: fixed).",
    )
    args = parser.parse_args()

    if not Path(args.file).is_file():
        sys.exit(f"Error: File not found: {args.file}")

    validate_config()

    filename = Path(args.file).name

    print(f"Extracting text from '{filename}'...")
    text = extract_text(args.file)
    if not text.strip():
        sys.exit("Error: No text could be extracted from the document.")
    print(f"  Extracted {len(text)} characters.")

    print(f"Chunking text using '{args.strategy}' strategy...")
    chunks = CHUNKING_STRATEGIES[args.strategy](text)
    if not chunks:
        sys.exit("Error: Chunking produced zero chunks.")
    print(f"  Created {len(chunks)} chunks.")

    print("Generating embeddings via Gemini API...")
    embeddings = generate_embeddings(chunks)
    print(f"  Generated {len(embeddings)} embeddings ({EMBEDDING_DIMENSION}d each).")

    print("Storing in PostgreSQL...")
    conn = get_connection()
    try:
        init_db(conn)
        rows = store_chunks(conn, chunks, embeddings, filename, args.strategy)
        print(f"  Inserted {rows} rows into 'document_chunks'.")
    finally:
        conn.close()

    print("Done.")


if __name__ == "__main__":
    main()
