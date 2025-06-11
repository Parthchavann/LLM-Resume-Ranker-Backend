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

# Load embedding model
embedder = SentenceTransformer("hkunlp/instructor-xl")  # Alternative: 'all-MiniLM-L6-v2'

# ✅ Use Mistral (open-access)
llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")

# ❌ DO NOT UNCOMMENT UNLESS YOU GET LLaMA ACCESS
# llm = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B", device_map="auto", use_auth_token=True)

# FAISS setup
dimension = 768  # Instructor-XL output vector size
faiss_index = faiss.IndexFlatL2(dimension)
resume_texts = []

# OpenAI GPT-4 key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Init FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class RankRequest(BaseModel):
    resumes: List[str]
    job_description: str

class FeedbackRequest(BaseModel):
    resume: str

# POST /rank → returns ranked resumes + LLM reasoning
@app.post("/rank")
def rank_resumes(req: RankRequest):
    job_emb = embedder.encode([req.job_description])
    resume_embs = embedder.encode(req.resumes)

    # Reset and add new embeddings
    faiss_index.reset()
    faiss_index.add(np.array(resume_embs).astype('float32'))
    resume_texts.clear()
    resume_texts.extend(req.resumes)

    # Semantic similarity search
    D, I = faiss_index.search(np.array(job_emb).astype('float32'), len(req.resumes))
    ranked = [{"resume": req.resumes[i], "score": float(1 / (d + 1e-5))} for d, i in zip(D[0], I[0])]

    # Mistral LLM reasoning for top 3
    for r in ranked[:3]:
        prompt = f"Given the job description: {req.job_description}\n\nDoes this resume match? {r['resume']}\n\nScore 1-10 and explain:"
        llm_out = llm(prompt, max_new_tokens=100)[0]['generated_text']
        r['llm_reasoning'] = llm_out

    return {"ranked": ranked}

# POST /feedback → GPT-4 resume improvement
@app.post("/feedback")
def resume_feedback(req: FeedbackRequest):
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful career coach."},
            {"role": "user", "content": f"Give actionable feedback on this resume:\n{req.resume}"}
        ]
    )
    return {"feedback": response['choices'][0]['message']['content']}

# POST /upload_resume → for file-to-text conversion
@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    content = await file.read()
    return {"text": content.decode("utf-8")}
