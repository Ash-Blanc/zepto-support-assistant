
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Load embedding model only once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="vector_store")

collection = client.get_collection(
    name="support_docs"
)

# KIE AI client
api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(
    api_key=api_key,
    base_url="https://api.kie.ai/gpt-5-2/v1"
)


def answer_question(question):

    total_start = time.time()

    # Convert question into embedding
    start = time.time()

    question_embedding = model.encode(
        [question],
        show_progress_bar=False
    )

    print("Embedding time:", round(time.time() - start, 2), "seconds")

    # Search similar chunks
    start = time.time()

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=3
    )

    print("ChromaDB time:", round(time.time() - start, 2), "seconds")

    # Create context
    context = "\n\n".join(results["documents"][0])

    # Shorter prompt
    prompt = f"""
ROLE:
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

    # Generate answer
    start = time.time()

    response = llm.chat.completions.create(
        model="gpt-5-2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("LLM time:", round(time.time() - start, 2), "seconds")
    print("Total time:", round(time.time() - total_start, 2), "seconds")

    return response.choices[0].message.content


def direct_answer(question):

    start = time.time()

    prompt = f"""
You are a helpful customer support assistant.

Answer this general question politely and briefly.

Do not invent Zepto policy information.

Question:
{question}
"""

    response = llm.chat.completions.create(
        model="gpt-5-2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("LLM time:", round(time.time() - start, 2), "seconds")

    return response.choices[0].message.content

