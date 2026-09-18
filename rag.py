
import os
import time
import logging

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Load embedding model once at module level
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="vector_store")
collection = client.get_collection(name="support_docs")

# LLM client — configurable via env vars with sensible defaults
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.warning("OPENAI_API_KEY not set — LLM calls will fail")

llm = OpenAI(
    api_key=api_key,
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.kie.ai/gpt-5-2/v1"),
)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5-2")


def _build_prompt(context: str, question: str) -> str:
    """Build the system prompt for policy-based answers."""
    return f"""ROLE:
You are a helpful Zepto customer support assistant.

CONTEXT:
{context}

TASK:
Answer the user's question using ONLY the context.

RULES:
- Do not invent information.
- If the answer is not in the context, say that the provided policy information does not contain the answer.
- Keep the answer short and clear.

EXAMPLE:

Question:
What should I do if my order contains a damaged item?

Answer:
You should report the damaged item according to the damaged item policy.

USER QUESTION:
{question}
"""


def answer_question(question: str) -> str:
    """Retrieve relevant chunks from ChromaDB and generate an answer."""
    total_start = time.time()

    # Embed the question
    start = time.time()
    question_embedding = model.encode([question], show_progress_bar=False)
    logger.info("Embedding time: %.2fs", time.time() - start)

    # Search similar chunks
    start = time.time()
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=3,
    )
    logger.info("ChromaDB time: %.2fs", time.time() - start)

    context = "\n\n".join(results["documents"][0])

    # Generate answer via LLM
    prompt = _build_prompt(context, question)
    start = time.time()
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    logger.info("LLM time: %.2fs", time.time() - start)
    logger.info("Total time: %.2fs", time.time() - total_start)

    return response.choices[0].message.content


def direct_answer(question: str) -> str:
    """Answer general questions that don't need RAG retrieval."""
    start = time.time()
    prompt = f"""You are a helpful customer support assistant.

Answer this general question politely and briefly.
Do not invent Zepto policy information.

Question:
{question}
"""
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    logger.info("LLM time: %.2fs", time.time() - start)

    return response.choices[0].message.content

