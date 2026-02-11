"""
MEMORA - RAG Memory System
Vector DB: Chroma
Embedding: bge-base-en-v1.5
LLM: Qwen2.5-3B-Instruct (local)

Goal:
- Store memory entries with metadata
- Retrieve relevant memories
- Generate safe, non-hallucinated responses
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
import time
import json
import torch

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# CONFIG
# =========================

@dataclass
class RAGConfig:
    persist_dir: str = "./chroma_memory_db"
    collection_name: str = "memories"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    top_k: int = 5
    min_similarity: float = 0.55
    max_context_chars: int = 2000
    fallback_response: str = "I am here with you. Please take your time."
    system_prompt: str = (
        "You are a memory assistant for Alzheimer's patients.\n"
        "Use ONLY the provided memory context.\n"
        "If the answer is not in the context, reply exactly with the fallback response.\n"
        "Do NOT add new facts, dates, times, or opinions.\n"
        "Answer in ONE short sentence.\n"
        "Do NOT ask questions."
    )


# =========================
# MEMORY SCHEMA
# =========================

@dataclass
class MemoryItem:
    user_id: str
    text: str
    relationship: Optional[str] = None
    event_type: Optional[str] = None
    timestamp: Optional[str] = None
    importance: Optional[str] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None


# =========================
# RAG SYSTEM
# =========================

class MemoryRAG:
    def __init__(self, config: RAGConfig = RAGConfig()):
        self.config = config

        # Embedding model
        self.embedder = SentenceTransformer(self.config.embedding_model, device=self.config.device)

        # Chroma DB
        self.client = chromadb.PersistentClient(path=self.config.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # LLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model,
            trust_remote_code=True,
            device_map="auto"
        )

    # =========================
    # MEMORY STORAGE
    # =========================
    def add_memories(self, memories: List[MemoryItem]) -> List[str]:
        texts = [m.text for m in memories]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)

        # Stable IDs to prevent duplicates
        ids = [f"{m.user_id}::{m.text}" for m in memories]
        metadatas = [self._memory_to_metadata(m) for m in memories]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        return ids

    def _memory_to_metadata(self, m: MemoryItem) -> Dict[str, Any]:
        data = asdict(m)

        # Ensure timestamp
        if data["timestamp"] is None:
            data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Chroma does not allow lists in metadata
        if data["tags"] is None:
            data["tags"] = ""
        elif isinstance(data["tags"], list):
            data["tags"] = ", ".join(data["tags"])

        # Remove None values (Chroma disallows None)
        clean = {k: v for k, v in data.items() if v is not None}

        return clean

    # =========================
    # RETRIEVAL
    # =========================
    def retrieve(self, query: str, user_id: Optional[str] = None) -> List[Dict]:
        query_emb = self.embedder.encode([query], normalize_embeddings=True)[0]

        where = {"user_id": user_id} if user_id else None

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=self.config.top_k,
            where=where
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = 1 - dist
            if similarity >= self.config.min_similarity:
                retrieved.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity": similarity
                })

        return retrieved

    # =========================
    # CONTEXT BUILDER
    # =========================
    def build_context(self, retrieved: List[Dict]) -> str:
        if not retrieved:
            return ""

        blocks = []
        total_len = 0
        for item in retrieved:
            meta = item["metadata"]
            block = (
                f"[Memory]\n"
                f"Text: {item['text']}\n"
                f"Relationship: {meta.get('relationship')}\n"
                f"Event: {meta.get('event_type')}\n"
                f"Importance: {meta.get('importance')}\n"
            )
            if total_len + len(block) > self.config.max_context_chars:
                break
            blocks.append(block)
            total_len += len(block)

        return "\n".join(blocks)

    # =========================
    # RESPONSE GENERATION
    # =========================
    def generate_response(self, query: str, user_id: Optional[str] = None) -> Dict:
        retrieved = self.retrieve(query, user_id=user_id)
        context = self.build_context(retrieved)

        if not context:
            return {
                "response": self.config.fallback_response,
                "retrieved": retrieved,
                "used_fallback": True
            }

        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": f"MEMORY CONTEXT:\n{context}\n\nQUESTION: {query}\n\nReturn ONE short sentence only."}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        )

        # Always convert to device and pass as kwargs
        inputs = inputs.to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        response = response.split("\n")[-1].strip()

        if not response:
            response = self.config.fallback_response

        return {
            "response": response,
            "retrieved": retrieved,
            "used_fallback": False
        }


# =========================
# Example Usage
# =========================
if __name__ == "__main__":
    rag = MemoryRAG()

    # Add memory
    rag.add_memories([
        MemoryItem(
            user_id="patient_001",
            text="Your daughter Ananya visited you last Sunday.",
            relationship="daughter",
            event_type="family_visit",
            importance="high"
        )
    ])

    # Ask question
    result = rag.generate_response("Did my daughter visit me recently?", user_id="patient_001")
    print(json.dumps(result, indent=2))