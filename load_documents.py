from dotenv import load_dotenv
from openai import OpenAI
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from pydantic import BaseModel


# Load environment variables from .env
load_dotenv()

docs_path = "docs"

# open and read documents from the docs folder and split them into chunks of 500 characters with an overlap of 50 characters

files = os.listdir(docs_path)

txt_files = []

for file in files:
    if file.endswith(".txt"):
        txt_files.append(file)

print("Number of documents:", len(txt_files))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

all_chunks = []

for file in txt_files:
    file_path = os.path.join(docs_path, file)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = text_splitter.split_text(content)

    print("\n", file)
    print("Number of chunks:", len(chunks))

    for chunk in chunks:
        all_chunks.append(chunk)

        print(chunk)
        print("----------------")

print("\nTotal chunks:", len(all_chunks))


# Create embeddings for all chunks using the SentenceTransformer model

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(all_chunks)

print("Number of embeddings:", len(embeddings))
print("Vector size:", len(embeddings[0]))


# Store the embeddings in ChromaDB

client = chromadb.PersistentClient(path="vector_store")

collection = client.get_or_create_collection(
    name="support_docs"
)

ids = []
documents = []
metadatas = []

for i, chunk in enumerate(all_chunks):
    ids.append(str(i))
    documents.append(chunk)
    metadatas.append({
        "source": "support_docs"
    })

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("Data stored in ChromaDB")
print("Total documents in collection:", collection.count())


# Query the collection for relevant chunks based on a user question

question = "What is the delivery time?"

question_embedding = model.encode([question])

results = collection.query(
    query_embeddings=question_embedding.tolist(),
    n_results=3
)

print("\nTop 3 relevant chunks:\n")

for i, chunk in enumerate(results["documents"][0]):
    print("Result", i + 1)
    print(chunk)
    print("----------------")


# chunks into single context string with two new lines between each chunk

context = "\n\n".join(results["documents"][0])

print("\nContext:\n")
print(context)


# prompt for the LLM to answer the user question based on the context

prompt = f"""
ROLE:
You are a helpful customer support assistant for Zepto.

CONTEXT:
{context}

TASK:
Answer the user's question using only the information provided in the context.

FORMAT:
Give a clear and simple answer.

LENGTH:
Keep the answer short, around 2-4 sentences.

CONSTRAINT:
Do not invent information that is not present in the context.

FEW-SHOT EXAMPLE:

Question:
What should I do if my order contains a damaged item?

Answer:
You should report the damaged item to customer support according to the damaged item policy.

USER QUESTION:
{question}
"""

print("\nPrompt:\n")
print(prompt)


# Connect to KIE AI's OpenAI-compatible API

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("\nERROR: OPENAI_API_KEY was not found in .env")
    exit()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.kie.ai/gpt-5-2/v1"
)


# Send the prompt to the LLM

response = client.chat.completions.create(
    model="gpt-5-2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# Get the final answer from the LLM

answer = response.choices[0].message.content

print("\nFinal Answer:\n")
print(answer)


class SupportResponse(BaseModel):
    question: str
    answer: str
    source: str


structured_response = SupportResponse(
    question=question,
    answer=answer,
    source="ChromaDB"
)

print("\nStructured Response:\n")
print(structured_response)

