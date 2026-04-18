"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ---- Request schema ----
class QueryRequest(BaseModel):
    question: str


# ---- Your existing function ----
def run_rag(question: str):
    docs = retriever.invoke(question)
    filtered_docs = filter_docs(docs, question)

    context = "\n\n".join([doc.page_content for doc in filtered_docs])

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response


# ---- API endpoint ----
@app.post("/rag")
def query_rag(req: QueryRequest):
    answer = run_rag(req.question)
    return {
        "question": req.question,
        "answer": answer
    }
"""


from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gc
from fastapi import HTTPException
import pandas as pd

app = FastAPI()

# -----------------------------
# GLOBALS (important)
# -----------------------------
embedding_model = None
vectorstore = None
retriever = None
chain = None
df = pd.read_csv("data.csv")


# -----------------------------
# Request schema
# -----------------------------
class QueryRequest(BaseModel):
    question: str


# -----------------------------
# Embedding class
# -----------------------------
class MyEmbedding(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


# -----------------------------
# Filter
# -----------------------------
def filter_docs(docs, question):
    now = datetime.now(timezone.utc)
    q = question.lower()

    filtered = docs

    if "pop" in q:
        filtered = [d for d in filtered if "pop" in d.page_content.lower()]

#    if "recent" in q:
#        filtered = [
#            d for d in filtered
#            if d.metadata["date"] >= now - timedelta(days=7)
#        ]

    return filtered


# -----------------------------
# Build vectorstore
# -----------------------------
def build_vectorstore():
    global vectorstore, retriever, df

    documents_all = []

    for _, row in df.iterrows():
        content = f"""
        Titre: {row['title.fr']}
        Description: {row['description.fr']}
        Ville: {row['location.city']}
        Lieu: {row['location.name']}
        Date: {row['lastTiming.begin']}
        """
        documents_all.append(Document(
            page_content=content,
            metadata={
                "date": row["lastTiming.begin"],
                "city": row["location.city"]
            }
        ))

    vectorstore = FAISS.from_documents(
        documents_all,
        MyEmbedding(embedding_model)
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# -----------------------------
# RAG pipeline
# -----------------------------
def run_rag(question: str):
    docs = retriever.invoke(question)
    filtered_docs = filter_docs(docs, question)

    context = "\n\n".join([doc.page_content for doc in filtered_docs])

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response


# -----------------------------
# STARTUP (important)
# -----------------------------
@app.on_event("startup")
def startup():
    global embedding_model, chain, df

    print("Loading models...")

    #embedding_model = SentenceTransformer('all-mpnet-base-v2')
    embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    pipe = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.1",
        max_new_tokens=100,
        temperature=0.7,
        device_map="auto",
        eos_token_id=2,
        pad_token_id=2,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    prompt = PromptTemplate.from_template("""
    Tu es un assistant qui recommande des événements.

    Contexte:
    {context}

    Question:
    {question}

    Réponds en français avec une liste claire
    Limite la réponse à 5 événements maximum.
    """)

    chain = prompt | llm | StrOutputParser()

    # Charger dataset
    df = pd.read_csv("data.csv")

    # Build vector DB
    build_vectorstore()

    print("API ready")


# -----------------------------
# ENDPOINT /ask
# -----------------------------

@app.post("/ask")
def ask(req: QueryRequest):
    try:
        answer = run_rag(req.question)
        return {
            "question": req.question,
            "answer": answer
        }
    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))




#@app.post("/ask")
#def ask(req: QueryRequest):
    #try:
        #answer = run_rag(req.question)
        #return {
            #"question": req.question,
            #"answer": answer
        #}
    #except Exception as e:
        #print(e)
        #print("ERROR:", e)
        #raise HTTPException(status_code=500, detail=str(e))
        #print(":)")
    #startup()
    #gc.collect()

# -----------------------------
# ENDPOINT /rebuild
# -----------------------------
@app.post("/rebuild")
def rebuild():
    build_vectorstore()
    return {
        "status": "ok",
        "message": "Vector store rebuilt successfully"
    }