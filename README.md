# Document Vectorization Module

A Python script that extracts text from PDF/DOCX documents, splits it into chunks using configurable strategies, generates vector embeddings via Google Gemini API, and stores everything in PostgreSQL with pgvector.

## How It Works

```
+----------------+      +----------------+      +------------------+      +----------------+
|                |      |                |      |                  |      |                |
|   PDF / DOCX   +----->+  Text Chunks   +----->+  Gemini API      +----->+  PostgreSQL    |
|   Document     |      |  (split text)  |      |  (embeddings)    |      |  (pgvector)    |
|                |      |                |      |                  |      |                |
+----------------+      +----------------+      +------------------+      +----------------+

  1. Extract text      2. Split into         3. Generate vector      4. Store chunks
     from file            chunks                embeddings             + vectors in DB
```

## Features

- **PDF & DOCX support**: extract clean text from both formats
- **3 chunking strategies**: fixed-size with overlap, sentence-based, paragraph-based
- **Google Gemini embeddings**: uses `gemini-embedding-001` (3072 dimensions)
- **PostgreSQL + pgvector**: stores chunks and vectors for semantic search

## Prerequisites

- Python 3.10+
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension installed
- A Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

## Installation

```bash
# Clone the repository
git clone https://github.com/ShayFeldboy1010/doc-vector.git
cd doc-vector

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY=your-gemini-api-key-here
POSTGRES_URL=postgresql://user:password@localhost:5432/your_database
```

> **Security:** Never commit your `.env` file. It is excluded via `.gitignore`.

## Usage

```bash
python index_documents.py --file <path-to-document> --strategy <strategy>
```

### Arguments

| Argument     | Required | Description                                      |
|-------------|----------|--------------------------------------------------|
| `--file`    | Yes      | Path to a `.pdf` or `.docx` file                 |
| `--strategy`| No       | Chunking strategy: `fixed`, `sentence`, or `paragraph` (default: `fixed`) |

### Examples

```bash
# Index a PDF using fixed-size chunks (default)
python index_documents.py --file report.pdf

# Index a DOCX using sentence-based splitting
python index_documents.py --file article.docx --strategy sentence

# Index a PDF using paragraph-based splitting
python index_documents.py --file notes.pdf --strategy paragraph
```

### Sample Output

```
Extracting text from 'report.pdf'...
  Extracted 15234 characters.
Chunking text using 'fixed' strategy...
  Created 38 chunks.
Generating embeddings via Gemini API...
  Generated 38 embeddings (3072d each).
Storing in PostgreSQL...
  Inserted 38 rows into 'document_chunks'.
Done.
```

## Database Schema

The script automatically creates the required table on first run:

```sql
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(3072) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    split_strategy VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Chunking Strategies

| Strategy    | Description                                              |
|------------|----------------------------------------------------------|
| `fixed`     | Splits text into 500-character chunks with 100-char overlap |
| `sentence`  | Splits on sentence boundaries (`.`, `!`, `?`)            |
| `paragraph` | Splits on blank lines between paragraphs                 |

## Project Structure

```
.
├── index_documents.py   # Main script (text extraction, chunking, embedding, storage)
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables
├── .gitignore           # Excludes .env and cache files
└── README.md            # This file
```
