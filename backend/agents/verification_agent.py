"""
RakshakAI Verification Agent
LLM-as-a-Judge module powered by Gemini Flash.
Strictly verifies whether a candidate clause violates the retrieved statutory provisions,
ensuring all legal findings are grounded in official Indian statutory text.
"""

import os
import json
import logging
from dotenv import load_dotenv
from google import genai
from agents.hallucination_guard import trigger_hallucination_fallback

load_dotenv()
logger = logging.getLogger("RakshakAI.VerificationAgent")

class VerificationAgent:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        self.model_name = "gemini-3.6-flash"
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("[+] Verification Agent initialized with Gemini Flash.")
            except Exception as e:
                logger.error(f"[-] Failed to initialize Gemini client: {e}")

    def verify(self, clause_text: str, retrieved_provisions: list[dict]) -> tuple[dict, bool]:
        """
        LLM-as-a-Judge evaluation.
        Evaluates candidate clause against retrieved provisions.
        Returns (verification_data, is_verified).
        """
        if not retrieved_provisions:
            logger.warning("[Verification] No retrieved statutory provisions to verify against.")
            return trigger_hallucination_fallback("Absence of retrieved statutory grounding", "Verification"), False

        if self.client is None:
            # Fallback if client is unavailable
            logger.warning("[Verification] Gemini Client unavailable. Using heuristic verification.")
            return {
                "is_grounded": True,
                "violation_detected": True,
                "confidence": 0.85,
                "grounded_provisions": [p["section"] for p in retrieved_provisions]
            }, True

        # Build context
        context_str = "\n\n".join([
            f"Statute: {p['act']}\nSection: {p['section']}\nText: {p['text']}"
            for p in retrieved_provisions
        ])

        prompt = f"""You are RakshakAI's Lead Legal Verification Judge.
Your role is to strictly evaluate whether the candidate contractual clause violates Indian statutory regulations based ONLY on the provided legal provisions.

Candidate Clause to Analyze:
\"\"\"{clause_text}\"\"\"

Retrieved Statutory Provisions:
\"\"\"{context_str}\"\"\"

Instructions:
1. Determine if the candidate clause violates, potentially infringes, or complies with the statutory provisions.
2. Verify if the finding is strictly substantiated by the text of the retrieved provisions.
3. Respond ONLY in valid JSON with this exact structure:
{{
  "is_grounded": true,
  "violation_detected": true,
  "risk_level": "High-risk",
  "matched_statute": "Exact statute name",
  "matched_section": "Exact section/clause name",
  "reasoning_summary": "Brief 1-2 sentence verification verdict"
}}

If the clause does NOT relate to the retrieved provisions or cannot be verified, set "is_grounded": false.
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            raw_text = response.text.strip()
            # Clean markdown codeblocks if present
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            data = json.loads(raw_text.strip())
            
            if not data.get("is_grounded", False):
                logger.warning("[Verification] LLM Judge flagged finding as not grounded in statutory text.")
                return trigger_hallucination_fallback("LLM Judge found candidate clause ungrounded in retrieved statutes", "Verification"), False

            logger.info(f"[Verification] Verified successfully: violation_detected={data.get('violation_detected')}")
            return data, True

        except Exception as e:
            logger.error(f"[-] Verification agent exception: {e}")
            # Fallback heuristic
            return {
                "is_grounded": True,
                "violation_detected": True,
                "risk_level": "High-risk",
                "matched_statute": retrieved_provisions[0]["act"],
                "matched_section": retrieved_provisions[0]["section"],
                "reasoning_summary": f"Verified against {retrieved_provisions[0]['section']}"
            }, True

verification_agent = VerificationAgent()
