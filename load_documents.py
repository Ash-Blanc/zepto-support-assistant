import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from pydantic import BaseModel
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DOCS_PATH = os.getenv("DOCS_PATH", "docs")
COLLECTION_NAME = "support_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_chunks(docs_path: str) -> list[str]:
    """Read all .txt files and split into overlapping chunks."""
    files = [f for f in os.listdir(docs_path) if f.endswith(".txt")]
    logger.info("Found %d documents", len(files))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks: list[str] = []
    for file in files:
        path = os.path.join(docs_path, file)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = splitter.split_text(content)
        logger.info("%s → %d chunks", file, len(chunks))
        all_chunks.extend(chunks)

    logger.info("Total chunks: %d", len(all_chunks))
    return all_chunks


def embed_and_store(chunks: list[str]) -> None:
    """Encode chunks and persist to ChromaDB."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    logger.info("Embeddings shape: %s", embeddings.shape)

    client = chromadb.PersistentClient(path="vector_store")
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = [str(i) for i in range(len(chunks))]
    metadatas = [{"source": "support_docs"} for _ in chunks]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    logger.info("Stored %d documents in ChromaDB", collection.count())


class SupportResponse(BaseModel):
    question: str
    answer: str
    source: str


if __name__ == "__main__":
    # This block only runs when you execute the script directly:
    #   python load_documents.py
    # Importing the module elsewhere won't trigger this one-time setup code.
    chunks = load_chunks(DOCS_PATH)
    embed_and_store(chunks)

    # One-off demo query (kept for reference)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.kie.ai/gpt-5-2/v1",
        )
        question = "What is the delivery time?"
        resp = client.chat.completions.create(
            model="gpt-5-2",
            messages=[{"role": "user", "content": question}],
        )
        result = SupportResponse(
            question=question,
            answer=resp.choices[0].message.content,
            source="ChromaDB",
        )
        logger.info("Demo response: %s", result)
    else:
        logger.warning("OPENAI_API_KEY not set — skipping demo query")

