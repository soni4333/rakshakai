"""
RakshakAI Screening Agent
Stage 1 Transformer-based screening classifier (fine-tuned DistilBERT).
Evaluates input clauses using calibrated probabilities.
If low-risk + high-confidence -> returns GREEN immediately (fast path).
Otherwise -> escalates to Retrieval and Verification Agents.
"""

import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger("RakshakAI.ScreeningAgent")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "screening_classifier")

class ScreeningAgent:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ScreeningAgent, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.temperature = 0.9099
        self.escalation_threshold = 0.4746
        self.load_model()

    def load_model(self):
        try:
            logger.info(f"[+] Loading Screening Classifier from {MODEL_DIR}...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(self.device)
            self.model.eval()

            params_path = os.path.join(MODEL_DIR, "calibration_params.json")
            if os.path.exists(params_path):
                with open(params_path, "r") as f:
                    params = json.load(f)
                    self.temperature = float(params.get("temperature", 0.9099))
                    self.escalation_threshold = float(params.get("escalation_threshold", 0.4746))
            logger.info(f"[+] Screening Classifier loaded (T*={self.temperature}, Thresh={self.escalation_threshold})")
        except Exception as e:
            logger.error(f"[-] Failed to load model from {MODEL_DIR}: {e}")

    def screen_clause(self, clause_text: str) -> dict:
        """
        Runs Stage 1 screening on input clause text.
        Returns screening result with early_exit=True for high-confidence low-risk clauses.
        """
        if not clause_text or not clause_text.strip():
            return {
                "verdict": "GREEN",
                "risk_level": "Low-risk",
                "confidence": 1.0,
                "high_risk_prob": 0.0,
                "early_exit": True,
                "explanation": "No actionable contractual text provided."
            }

        if self.model is None or self.tokenizer is None:
            # Fallback if model not loaded
            return {
                "verdict": "ESCALATE",
                "risk_level": "Unknown",
                "confidence": 0.5,
                "high_risk_prob": 0.5,
                "early_exit": False
            }

        inputs = self.tokenizer(
            clause_text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_logits = outputs.logits.cpu().numpy()[0]

        # Apply Temperature Scaling
        scaled_logits = raw_logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        prob_low_risk = float(probs[0])
        prob_high_risk = float(probs[1])

        logger.info(f"[Screening] P(High-risk)={prob_high_risk:.4f}, P(Low-risk)={prob_low_risk:.4f}")

        # If probability of high-risk is below the escalation threshold and confidence is strong -> GREEN early exit
        if prob_high_risk < self.escalation_threshold and prob_low_risk >= 0.70:
            return {
                "verdict": "GREEN",
                "risk_level": "Low-risk",
                "confidence": round(prob_low_risk, 4),
                "high_risk_prob": round(prob_high_risk, 4),
                "early_exit": True,
                "citation": "Standard Fair Terms Compliance",
                "explanation": "Clause screened as compliant/standard fair term under Stage 1 screening with high confidence.",
                "disclaimer": "Automated screening evaluation. Does not substitute formal legal counsel."
            }
        else:
            return {
                "verdict": "ESCALATE",
                "risk_level": "High-risk / Potential Concern",
                "confidence": round(prob_high_risk, 4),
                "high_risk_prob": round(prob_high_risk, 4),
                "early_exit": False,
                "explanation": f"Escalated to Stage 2 verification (P(High-risk) = {prob_high_risk:.4f} >= threshold {self.escalation_threshold:.4f})."
            }

screening_agent = ScreeningAgent()
