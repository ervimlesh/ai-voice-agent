"""
Medical RAG Service: Provides doctor suggestions based on patient symptoms using embeddings and FAISS.
Uses sentence-transformers for embedding and FAISS for fast similarity search over medical knowledge base.
"""

import logging
import threading
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)

# Medical knowledge base: (symptom description, suggested doctor question)
MEDICAL_KNOWLEDGE_BASE = [
    # Fever (3 pairs)
    (
        "patient has fever high temperature",
        "How high is your temperature, and for how many days have you had the fever?",
    ),
    (
        "fever with chills sweating night sweats",
        "Are you experiencing chills or night sweats along with the fever?",
    ),
    (
        "fever after travel or exposure",
        "Have you recently traveled or been exposed to anyone who was ill?",
    ),
    # Cough (3 pairs)
    (
        "cough dry persistent throat irritation",
        "Is the cough dry or are you producing mucus, and how long has it lasted?",
    ),
    (
        "cough with blood hemoptysis",
        "Have you coughed up any blood or noticed blood in your mucus?",
    ),
    (
        "cough shortness of breath wheezing",
        "Do you experience shortness of breath or wheezing along with the cough?",
    ),
    # Headache (3 pairs)
    (
        "headache head pain migraine",
        "On a scale of 1 to 10, how severe is the headache, and where exactly is the pain located?",
    ),
    (
        "headache with nausea light sensitivity",
        "Is the headache accompanied by nausea, vomiting, or sensitivity to light and sound?",
    ),
    (
        "sudden severe headache thunderclap",
        "Did the headache come on suddenly and severely, or did it build up gradually?",
    ),
    # Pain general (3 pairs)
    (
        "pain ache body soreness",
        "Can you describe the character of the pain — is it sharp, dull, throbbing, or burning?",
    ),
    (
        "pain worse with movement or rest",
        "Does the pain worsen with movement or activity, or is it constant even at rest?",
    ),
    (
        "radiating pain shooting pain",
        "Does the pain stay in one place or does it radiate or spread to other areas?",
    ),
    # Nausea and vomiting (3 pairs)
    (
        "nausea vomiting upset stomach",
        "How frequently are you vomiting, and is there any blood or bile in the vomit?",
    ),
    (
        "nausea after eating food intolerance",
        "Does the nausea occur after eating specific foods or at particular times of day?",
    ),
    (
        "nausea dizziness vertigo",
        "Are you also experiencing dizziness or a spinning sensation along with the nausea?",
    ),
    # Fatigue and weakness (3 pairs)
    (
        "fatigue tiredness weakness low energy",
        "How long have you been feeling fatigued, and does rest improve your energy levels?",
    ),
    (
        "fatigue with weight loss appetite loss",
        "Have you noticed any unintended weight loss or decreased appetite along with the fatigue?",
    ),
    (
        "fatigue with shortness of breath exertion",
        "Do you feel breathless or experience chest discomfort when physically active?",
    ),
    # Breathing difficulty (3 pairs)
    (
        "shortness of breath breathing difficulty dyspnea",
        "At what level of activity does the shortness of breath begin — at rest, walking, or climbing stairs?",
    ),
    (
        "breathing difficulty at night lying down",
        "Do you experience more difficulty breathing when lying down or at night?",
    ),
    (
        "breathing difficulty with chest tightness",
        "Do you feel tightness or pressure in your chest along with the breathing difficulty?",
    ),
    # Heart conditions (3 pairs)
    (
        "chest pain chest tightness heart",
        "Does the chest pain radiate to your arm, jaw, or back, and is it associated with sweating?",
    ),
    (
        "palpitations irregular heartbeat",
        "How often do the palpitations occur, and do they come with dizziness or fainting?",
    ),
    (
        "heart condition family history cardiac",
        "Do you have a family history of heart disease or have you had a previous cardiac event?",
    ),
    # Diabetes (3 pairs)
    (
        "diabetes blood sugar glucose",
        "When was your last blood sugar check, and what was the reading?",
    ),
    (
        "increased thirst frequent urination diabetes symptoms",
        "Are you experiencing excessive thirst, frequent urination, or unusual hunger?",
    ),
    (
        "diabetes foot numbness tingling",
        "Have you noticed any numbness, tingling, or wounds on your feet that are slow to heal?",
    ),
    # Blood pressure (3 pairs)
    (
        "high blood pressure hypertension",
        "What is your usual blood pressure reading, and are you currently taking any medication for it?",
    ),
    (
        "low blood pressure dizziness fainting",
        "Do you feel dizzy or faint when you stand up quickly?",
    ),
    (
        "blood pressure headache vision changes",
        "Have you experienced sudden severe headaches or changes in your vision recently?",
    ),
]


class MedicalRAGService:
    """
    Medical RAG service providing context-aware doctor suggestions.
    Uses sentence-transformers + FAISS for efficient similarity search.
    """

    def __init__(self):
        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.Index] = None
        self._questions: List[str] = []
        self._lock = threading.Lock()
        self._initialized = False

    def _initialize(self) -> None:
        """Lazy initialization: load model and build FAISS index on first use."""
        with self._lock:
            if self._initialized:
                return

            if faiss is None:
                logger.warning("FAISS not installed, RAG service will not provide suggestions. Install with: pip install faiss-cpu")
                self._initialized = True
                return

            logger.info("Initializing Medical RAG Service...")

            # Load embedding model (22MB, 384-dim embeddings)
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence-transformers model: all-MiniLM-L6-v2")

            # Extract symptom descriptions and corresponding questions
            symptoms = [pair[0] for pair in MEDICAL_KNOWLEDGE_BASE]
            self._questions = [pair[1] for pair in MEDICAL_KNOWLEDGE_BASE]

            # Encode all symptoms with L2 normalization (for cosine similarity)
            logger.info(f"Encoding {len(symptoms)} medical knowledge base entries...")
            embeddings = self._model.encode(
                symptoms, convert_to_numpy=True, normalize_embeddings=True
            )

            # Create FAISS index for cosine similarity (IndexFlatIP on normalized vectors)
            dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
            self._index = faiss.IndexFlatIP(dimension)
            self._index.add(embeddings.astype(np.float32))

            logger.info(f"Built FAISS index with {self._index.ntotal} vectors (dim={dimension})")
            self._initialized = True

    def get_suggested_questions(self, patient_text: str, top_k: int = 3) -> List[str]:
        """
        Query the medical knowledge base for relevant doctor questions.

        Args:
            patient_text: Patient's spoken statement or transcript
            top_k: Number of suggestions to return (default 3)

        Returns:
            List of suggested follow-up questions for the doctor
        """
        if not self._initialized:
            self._initialize()

        # If FAISS is not available, return empty suggestions
        if faiss is None or self._index is None:
            return []

        try:
            # Encode patient query
            query_embedding = self._model.encode(
                [patient_text], convert_to_numpy=True, normalize_embeddings=True
            )

            # Search FAISS index (returns distances and indices)
            distances, indices = self._index.search(
                query_embedding.astype(np.float32), top_k
            )

            # Extract questions with similarity > threshold
            suggestions = []
            min_similarity_threshold = 0.2  # Cosine similarity threshold (0-1 scale)

            for i, idx in enumerate(indices[0]):
                if idx >= 0 and distances[0][i] > min_similarity_threshold:
                    question = self._questions[idx]
                    suggestions.append(question)
                    logger.debug(f"RAG suggestion {i+1}: similarity={distances[0][i]:.3f} -> {question[:60]}...")

            logger.info(f"RAG returned {len(suggestions)} suggestions for patient text: '{patient_text[:80]}...'")
            return suggestions

        except Exception as e:
            logger.error(f"Error querying medical RAG: {e}")
            return []

    async def get_suggested_questions_async(self, patient_text: str, top_k: int = 3) -> List[str]:
        """
        Async wrapper for getting suggested questions (runs in thread pool).
        Use this in async contexts like WebSocket handlers.
        """
        import asyncio

        return await asyncio.to_thread(self.get_suggested_questions, patient_text, top_k)
