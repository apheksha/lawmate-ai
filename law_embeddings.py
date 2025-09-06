import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class LawDatabase:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.qa_laws = []
        self.ref_laws = []
        self.qa_embeddings = None
        self.ref_embeddings = None
        self.load_qa_laws()
        self.load_ref_laws()
        self.create_embeddings()

    def load_qa_laws(self):
        files = ["data/crpc_qa.json", "data/constitution_qa.json"]
        for f in files:
            path = Path(f)
            if path.exists():
                data = json.load(open(path, "r", encoding="utf-8"))
                for entry in data:
                    if "question" in entry and "answer" in entry:
                        self.qa_laws.append(entry)

    def load_ref_laws(self):
        path = Path("data/law_refs.csv")
        if path.exists():
            df = pd.read_csv(path)
            # Store as dicts with title and URL
            self.ref_laws = df.to_dict(orient="records")

    def create_embeddings(self):
        # QA embeddings
        if self.qa_laws:
            questions = [q["question"] for q in self.qa_laws]
            self.qa_embeddings = self.model.encode(questions, convert_to_tensor=True)

        # Ref laws embeddings
        if self.ref_laws:
            texts = [r["title"] + " " + str(r.get("source","")) for r in self.ref_laws]
            self.ref_embeddings = self.model.encode(texts, convert_to_tensor=True)

    def query_laws(self, query, top_k=3):
        emb = self.model.encode([query], convert_to_tensor=True)
        results = []

        # QA similarity
        if self.qa_embeddings is not None:
            sims = cosine_similarity(emb.cpu(), self.qa_embeddings.cpu())[0]
            top_idx = sims.argsort()[-top_k:][::-1]
            for idx in top_idx:
                results.append({"type": "qa", "question": self.qa_laws[idx]["question"],
                                "answer": self.qa_laws[idx]["answer"],
                                "score": float(sims[idx])})

        # Ref similarity
        if self.ref_embeddings is not None:
            sims = cosine_similarity(emb.cpu(), self.ref_embeddings.cpu())[0]
            top_idx = sims.argsort()[-top_k:][::-1]
            for idx in top_idx:
                r = self.ref_laws[idx]
                results.append({"type": "ref", "title": r["title"], "url": r["url"], "score": float(sims[idx])})

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k*2]
