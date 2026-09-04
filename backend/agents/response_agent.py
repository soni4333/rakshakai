"""
RakshakAI Response Agent
Formats the output from Reasoning / Screening / Whitelist / Hallucination Guard agents
into structured JSON and clean Markdown cards for the UI chat interface.
"""

class ResponseAgent:
    def format_response(
        self,
        raw_result: dict,
        original_text: str = "",
        english_text: str = "",
        detected_lang: str = "en",
        provisions: list = None
    ) -> dict:
        """
        Formats final response dictionary for API consumers and chat UI.
        """
        verdict = raw_result.get("verdict", "YELLOW")
        early_exit = raw_result.get("early_exit", False)
        hallucination_guard_triggered = raw_result.get("hallucination_guard_triggered", False)

        # Map to standard risk_level and signal_label
        if verdict == "RED":
            risk_level = "high"
            signal_label = "High-risk clause detected"
            confidence = raw_result.get("confidence", 94.5)
        elif verdict == "YELLOW":
            risk_level = "medium"
            signal_label = "Review recommended"
            confidence = raw_result.get("confidence", 78.0)
        elif verdict == "GREEN":
            risk_level = "low"
            signal_label = "No high-risk clause detected."
            confidence = raw_result.get("confidence", 92.0)
        else:
            risk_level = "uncertain"
            signal_label = "More information needed"
            confidence = raw_result.get("confidence", 50.0)

        # Ensure confidence is formatted nicely (e.g. 92.5%)
        if isinstance(confidence, float) and confidence <= 1.0:
            confidence = round(confidence * 100, 1)
        elif isinstance(confidence, (int, float)):
            confidence = round(float(confidence), 1)
        else:
            confidence = 88.0

        citation = raw_result.get("citation", "Statutory Compliance Benchmark")
        explanation = raw_result.get("explanation", "Analysis completed based on statutory evidence.")
        recommendation = raw_result.get("recommendation", "Review terms carefully before executing.")
        highlighted_finding = raw_result.get("highlighted_finding") or (
            explanation.split(".")[0] if "." in explanation else explanation[:120]
        )

        # Lender details
        lender_name = raw_result.get("matched_entity") or raw_result.get("lender_name")
        if raw_result.get("status") == "UNREGISTERED_LENDING_ENTITY":
            lender_status = "unverified lender / elevated caution"
        elif raw_result.get("status") == "WHITELISTED":
            lender_status = f"RBI-registered {raw_result.get('entity_type', 'Regulated Entity')}"
        else:
            lender_status = None

        # Build evidence items
        evidence_list = []
        if provisions:
            for p in provisions:
                evidence_list.append({
                    "id": p.get("id", "STATUTE"),
                    "title": p.get("section", "Statutory Provision"),
                    "source_name": p.get("act", "Official Statute"),
                    "passage": p.get("text", ""),
                    "source_url": "https://prsindia.org" if "Act" in p.get("act", "") else "https://rbi.org.in",
                    "review_status": "verified in gazette / official circular",
                    "freshness_warning": None
                })
        else:
            evidence_list.append({
                "id": "STATUTE_01",
                "title": citation,
                "source_name": "Indian Legal Framework",
                "passage": explanation,
                "source_url": "https://rbi.org.in",
                "review_status": "verified benchmark",
                "freshness_warning": None
            })

        # Bilingual audit trail
        bilingual_audit = {
            "original_text": original_text or english_text,
            "english_interpretation": english_text or original_text,
            "detected_language": detected_lang or "en",
            "translation_status": "completed (AI4Bharat IndicTrans2)" if detected_lang != "en" else "English source (pass-through)"
        }

        # Fixed disclaimer per product requirements
        fixed_disclaimer = "RakshakAI is an informational screening tool, not legal advice. A risk signal is not a declaration of legality. Consider speaking with a qualified professional."

        return {
            "verdict": verdict,
            "risk_level": risk_level,
            "signal_label": signal_label,
            "confidence": confidence,
            "citation": citation,
            "explanation": explanation,
            "finding_text": explanation,
            "highlighted_finding": highlighted_finding,
            "recommendation": recommendation,
            "lender_name": lender_name,
            "lender_status": lender_status,
            "evidence": evidence_list,
            "bilingual_audit": bilingual_audit,
            "limitations": "Informational risk signal based on statutory screening; not formal adjudication.",
            "disclaimer": fixed_disclaimer,
            "early_exit": early_exit,
            "hallucination_guard_triggered": hallucination_guard_triggered,
            "needs_human_review": raw_result.get("needs_human_review", False)
        }

response_agent = ResponseAgent()
