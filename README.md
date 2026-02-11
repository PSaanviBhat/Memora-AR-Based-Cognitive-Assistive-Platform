# Memora-AR-Based-Cognitive-Assistive-Platform
Memora: AR-Based Cognitive Assistive Platform for Early Stage Alzheimer's Patients Integrating Facial Recognition, Age Progression Tolerance, Lifelong Learning, and Voice-Based NLP Prompts.
# MEMORA – Multimodal Memory Assistant for Alzheimer’s Patients

A full end‑to‑end pipeline that combines **biometric identity (ArcFace + ECAPA‑TDNN)**, **memory recall (RAG)**, and **voice interaction (STT/TTS)** into a safe, low‑latency assistant designed for Alzheimer’s care.

---

##  System Overview

MEMORA performs:

1) **Identity Recognition**
   - Face ID (ArcFace)
   - Voice ID (ECAPA‑TDNN)
   - Fusion → stable `user_id`

2) **Memory Recall (RAG)**
   - Vector DB search (Chroma)
   - Metadata‑aware memory retrieval
   - Safe response generation (Qwen LLM)

3) **Voice Interaction**
   - STT (Whisper)
   - TTS (Coqui TTS)
   - Opus audio streaming

---

##  End‑to‑End Pipeline

```
Sensors → (Face/Voice ID) → Identity Fusion → User ID
            ↓
        Transcript (STT)
            ↓
   Memory RAG (vector DB + LLM)
            ↓
        Response (TTS + Opus)
```

---

##  Safety Guarantees

- **Always responds** (fallback reply if memory or STT fails)
- **No hallucinated facts** in memory retrieval (context‑only)
- **Calm and stable TTS output** for Alzheimer’s safety
- **Stable identity mapping** across sessions

---


##  Contact / Collaboration

This repo is intended for **clinical‑grade**, **patient‑safe** deployment.  
All changes should maintain **predictability**, **calm responses**, and **low latency**.

---
