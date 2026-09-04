"""
RakshakAI Retrieval Agent
Queries the ChromaDB statutory vector store for matching legal provisions under Indian law.
Performs 1 query reformulation retry if initial match distance/relevance is weak.
Connects with Hallucination Guard if legal grounding cannot be retrieved.
"""

import logging
from retrieval.vector_store import vector_store
from agents.hallucination_guard import trigger_hallucination_fallback

logger = logging.getLogger("RakshakAI.RetrievalAgent")

class RetrievalAgent:
    def __init__(self):
        self.store = vector_store
        self.distance_threshold = 1.35  # Threshold for Chroma default embedding L2 distance

    def reformulate_query(self, clause_text: str) -> str:
        """
        Creates a legally-focused keyword query from candidate clause text.
        """
        text = clause_text.lower()
        keywords = []
        if any(w in text for w in ["contact", "phonebook", "call log", "gallery", "photo", "permission"]):
            keywords.append("RBI Digital Lending Guidelines prohibited contact list gallery phone access")
        if any(w in text for w in ["consent", "privacy", "personal data", "erase", "share", "third-party", "notice"]):
            keywords.append("DPDP Act 2023 consent notice purpose limitation erasure")
        if any(w in text for w in ["terminate", "suspend", "deactivate", "block", "fire", "notice", "driver", "partner"]):
            keywords.append("Gig worker platform arbitrary deactivation notice fair hearing")
        if any(w in text for w in ["recovery", "agent", "call", "hour", "night", "threat", "harass"]):
            keywords.append("RBI Fair practices debt recovery permitted hours harassment")
        if any(w in text for w in ["interest", "apr", "cost", "fee", "kfs", "fact statement", "charge"]):
            keywords.append("RBI Key Fact Statement KFS APR total cost of credit disclosure")
        if any(w in text for w in ["liability", "penalty", "indemnity", "dispute", "court", "arbitration"]):
            keywords.append("Indian Contract Act 1872 unconscionable terms arbitrary penalty limitation of liability")
        if any(w in text for w in ["union", "strike", "association", "collective", "gather"]):
            keywords.append("Constitution of India freedom of association gig worker collective representation")
            
        if keywords:
            return " ".join(keywords)
        return f"Indian legal regulatory compliance DPDP Act RBI Guidelines {clause_text}"

    def retrieve(self, clause_text: str) -> tuple[list[dict], bool]:
        """
        Retrieves top statutory provisions for candidate clause.
        Returns (list_of_provisions, is_valid).
        If off-topic or empty, performs 1 reformulated retry.
        """
        if not clause_text or not clause_text.strip():
            return [], False

        # Attempt 1: Direct query
        results = self.store.query_statutes(clause_text, top_k=3)
        
        # Check relevance
        best_dist = results[0]["distance"] if results else 999.0
        if not results or best_dist > self.distance_threshold:
            logger.warning(f"[Retrieval] Initial match weak (dist={best_dist:.3f}). Attempting reformulated retry...")
            
            # Attempt 2: Reformulated retry
            reformulated = self.reformulate_query(clause_text)
            results = self.store.query_statutes(reformulated, top_k=3)
            best_dist = results[0]["distance"] if results else 999.0

        if not results or best_dist > 1.60:
            logger.error("[Retrieval] Failed to retrieve grounded legal context after retry.")
            return [], False

        logger.info(f"[Retrieval] Successfully retrieved {len(results)} statutory provisions (Best dist: {best_dist:.3f})")
        return results, True

retrieval_agent = RetrievalAgent()
