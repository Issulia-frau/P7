import json
import pandas as pd
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

# -----------------------------
# Embedding
# -----------------------------
class MyEmbedding(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data.csv")

embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')


# -----------------------------
# Build vectorstore
# -----------------------------
def build_vectorstore():
    documents = []

    for _, row in df.iterrows():
        content = f"""
        Titre: {row['title.fr']}
        Description: {row['description.fr']}
        Ville: {row['location.city']}
        Lieu: {row['location.name']}
        Date: {row['lastTiming.begin']}
        """

        documents.append(Document(
            page_content=content,
            metadata={
                "id": row["uid"],   # 🔥 IMPORTANT
                "city": row["location.city"],
                "date": row["lastTiming.begin"]
            }
        ))

    vectorstore = FAISS.from_documents(
        documents,
        MyEmbedding(embedding_model)
    )

    return vectorstore.as_retriever(search_kwargs={"k": 5})


retriever = build_vectorstore()


# -----------------------------
# RAG (retrieval only ici)
# -----------------------------
def retrieve(question):
    docs = retriever.invoke(question)
    return docs


# -----------------------------
# Recall@k
# -----------------------------
def recall_at_k(retrieved_docs, ground_truth_ids):
    retrieved_ids = [doc.metadata["id"] for doc in retrieved_docs]
    hits = len(set(retrieved_ids) & set(ground_truth_ids))
    return hits / len(ground_truth_ids)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate():
    with open("eval.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []

    for sample in dataset:
        question = sample["question"]
        gt_ids = sample["ground_truth_ids"]

        docs = retrieve(question)
        recall = recall_at_k(docs, gt_ids)

        results.append({
            "question": question,
            "recall@k": recall,
            "retrieved_ids": [d.metadata["id"] for d in docs]
        })

    df_results = pd.DataFrame(results)

    print("\n===== RESULTS =====")
    print(df_results)

    print("\nAverage Recall@k:", df_results["recall@k"].mean())


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    evaluate()