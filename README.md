
# RAG Proof of Concept

A Retrieval-Augmented Generation (RAG) implementation using Ollama, ChromaDB, and Python. This project demonstrates how to build a very simple RAG system that can answer questions based on a custom knowledge base.

## 🎯 Overview

This proof of concept showcases:
- **Vector Database**: ChromaDB for persistent storage of embeddings
- **LLM Integration**: Ollama for both embeddings and text generation
- **RAG Pipeline**: Document chunking, embedding, retrieval, and generation

## 📋 Prerequisites

- Python 3.8+
- Docker & Docker Compose

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone git@github.com:raphayol/poc-rag.git
cd rag-poc
```

### 2. Set Up Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` to customize your configuration:

**Available Configuration Options:**

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_EMBEDDING_MODEL` | Model for generating embeddings | `nomic-embed-text` |
| `OLLAMA_GENERATION_MODEL` | Model for text generation | `deepseek-r1:1.5b` |

Browse available models at: https://ollama.com/search

### 4. Start Ollama Server

```bash
docker compose up -d
```

### 5. Download Required Models

Access the Ollama container:

```bash
docker exec -it ollama_server bash
```

Inside the container, pull the required models (must match your `.env` configuration):

```bash
# Embedding model (required)
ollama pull nomic-embed-text

# Generation model (required)
ollama pull deepseek-r1:1.5b

# Optional: View model details
ollama show nomic-embed-text
ollama show deepseek-r1:1.5b
```

Exit the container:

```bash
exit
```

### 6. Change the Knowledge Base

Replace the content in `data.txt` with your own knowledge base.

**Important Notes:**
In this simple poc data are only loaded at startup, changes to `data.txt` require deleting `chroma_db/` to reload

### 7. Run the RAG System

**1. With command-line arguments:**
```bash
python3 src/rag.py "What is my name?"
# or without quotes (spaces separate words)
python3 src/rag.py What is the meaning of life
```

**2. Interactive mode (prompts for question):**
```bash
python3 src/rag.py
# Will prompt: Enter your question:
```
