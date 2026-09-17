# Zepto Support Assistant

A GenAI-powered customer support assistant built using Retrieval-Augmented Generation (RAG). The application retrieves relevant information from support documents and uses an LLM to generate clear and contextual answers.

## Features

- RAG-based customer support
- Document chunking and text embeddings
- ChromaDB vector database
- Sentence Transformers (`all-MiniLM-L6-v2`)
- LangGraph for intent-based routing
- KIE GPT-5.2 for response generation
- Pydantic for structured responses
- Flask backend
- HTML, CSS and JavaScript frontend
- Dockerized application

## Workflow

User Question → LangGraph → Intent Classification

- Policy Question → ChromaDB Retrieval → Relevant Context → LLM → Answer
- General Question → LLM → Answer

## Technologies Used

- Python
- Flask
- LangGraph
- ChromaDB
- Sentence Transformers
- KIE GPT
- Pydantic
- HTML/CSS/JavaScript
- Docker

## Project Structure

```text
support_assistant/
│
├── docs/
├── static/
├── templates/
├── app.py
├── rag.py
├── graph.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md