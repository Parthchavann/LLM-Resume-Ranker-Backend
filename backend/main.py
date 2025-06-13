from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import openai
import os

# ---------------------------
# 📌 Load Models
# ---------------------------

# Embedding model (MiniLM: 384-dim)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# LLM model (requires huggingface-cli login)
llm = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B",
    device_map="auto"
)

# OpenAI GPT-4 API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------
# 📌 FAISS Setup
# ---------------------------

dimension = 384  # Correct for MiniLM
faiss_index = faiss.IndexFlatL2(dimension)
resume_texts = []

# ---------------------------
# 📌 FastAPI Setup
# ---------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 📌 Request Models
# ---------------------------

class RankRequest(BaseModel):
    resumes: List[str]
    job_description: str

class FeedbackRequest(BaseModel):
    resume: str

# ---------------------------
# 📌 API Endpoints
# ---------------------------

@app.post("/rank")
def rank_resumes(req: RankRequest):
    try:
        # Embed job description and resumes
        job_emb = embedder.encode([req.job_description])
        resume_embs = embedder.encode(req.resumes)

        # FAISS Indexing
        faiss_index.reset()
        faiss_index.add(np.array(resume_embs).astype('float32'))
        resume_texts.clear()
        resume_texts.extend(req.resumes)

        # Nearest Neighbor Search
        D, I = faiss_index.search(np.array(job_emb).astype('float32'), len(req.resumes))
        ranked = [{"resume": req.resumes[i], "score": float(1 / (d + 1e-5))} for d, i in zip(D[0], I[0])]

        # Optional: LLM Reranking
        for r in ranked[:3]:
            prompt = f"Given the job description: {req.job_description}\n\nDoes this resume match? {r['resume']}\n\nScore 1-10 and explain:"
            try:
                llm_out = llm(prompt, max_new_tokens=100)[0]['generated_text']
                r['llm_reasoning'] = llm_out
            except Exception as e:
                r['llm_reasoning'] = f"LLM error: {str(e)}"

        return {"ranked": ranked}
    except Exception as e:
        return {"error": f"Ranking failed: {str(e)}"}

@app.post("/feedback")
def resume_feedback(req: FeedbackRequest):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful career coach."},
                {"role": "user", "content": f"Give actionable feedback on this resume:\n{req.resume}"}
            ]
        )
        return {"feedback": response['choices'][0]['message']['content']}
    except Exception as e:
        return {"feedback": f"OpenAI API error: {str(e)}"}

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    content = await file.read()
    return {"text": content.decode("utf-8")}
