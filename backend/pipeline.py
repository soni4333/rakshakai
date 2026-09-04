"""
RakshakAI End-to-End Analysis Pipeline
Orchestrates sequential multi-agent execution with early exits and safety guardrails:
1. Translation Agent (Input)
2. Whitelist Agent (for lending entity/app queries)
3. Screening Agent (DistilBERT Stage 1 model; fast GREEN exit)
4. Retrieval Agent (ChromaDB vector store + 1 retry)
5. Verification Agent (Gemini Flash LLM-as-a-Judge)
6. Reasoning Agent (Gemini Flash verdict synthesis)
7. Response Agent (Chat & API formatting)
8. Translation Agent (Output)
"""

import logging
from agents.translation_agent import translation_agent
from agents.whitelist_agent import check_whitelist
from agents.screening_agent import screening_agent
from agents.retrieval_agent import retrieval_agent
from agents.verification_agent import verification_agent
from agents.reasoning_agent import reasoning_agent
from agents.response_agent import response_agent
from agents.hallucination_guard import trigger_hallucination_fallback

logger = logging.getLogger("RakshakAI.Pipeline")

def run_pipeline(
    text: str,
    entity_name: str = None,
    source_lang: str = "en",
    target_lang: str = "en"
) -> dict:
    """
    Executes the full RakshakAI multi-agent pipeline on a candidate clause or entity.
    """
    logger.info(f"[Pipeline] Starting analysis (source_lang={source_lang}, target_lang={target_lang})")

    # Step 1: Translation Agent (Input)
    english_text, detected_lang = translation_agent.translate_input(text, source_lang)
    lang_to_return = target_lang if target_lang and target_lang != "auto" else detected_lang

    # Step 2: Whitelist Agent (Ad/Lending entity path)
    if entity_name and str(entity_name).strip():
        logger.info(f"[Pipeline] Checking entity whitelist for: '{entity_name}'")
        whitelist_result = check_whitelist(entity_name)
        if whitelist_result.get("early_exit", False):
            logger.info("[Pipeline] Whitelist early exit triggered (Unregistered Entity -> RED).")
            formatted = response_agent.format_response(
                whitelist_result,
                original_text=text,
                english_text=english_text,
                detected_lang=detected_lang
            )
            return translation_agent.translate_output(formatted, lang_to_return)

    # Step 3: Screening Agent (DistilBERT Phase 2 model)
    logger.info("[Pipeline] Executing Stage 1 Screening Agent...")
    screening_result = screening_agent.screen_clause(english_text)

    # Fast-Path: High confidence Low-risk -> GREEN immediately
    if screening_result.get("early_exit", False):
        logger.info("[Pipeline] Screening Agent Fast Path Triggered (High-confidence Low-risk -> GREEN).")
        formatted = response_agent.format_response(
            screening_result,
            original_text=text,
            english_text=english_text,
            detected_lang=detected_lang
        )
        return translation_agent.translate_output(formatted, lang_to_return)

    # Step 4: Retrieval Agent (ChromaDB + Reformulation Retry)
    logger.info("[Pipeline] Executing Retrieval Agent...")
    provisions, is_valid = retrieval_agent.retrieve(english_text)
    if not is_valid or not provisions:
        logger.warning("[Pipeline] Legal grounding retrieval failed. Triggering Hallucination Guard.")
        fallback = trigger_hallucination_fallback("No statutory grounding retrieved", "Retrieval")
        formatted = response_agent.format_response(
            fallback,
            original_text=text,
            english_text=english_text,
            detected_lang=detected_lang
        )
        return translation_agent.translate_output(formatted, lang_to_return)

    # Step 5: Verification Agent (Gemini Flash LLM-as-a-Judge)
    logger.info("[Pipeline] Executing Verification Agent (LLM-as-a-Judge)...")
    verification_data, is_verified = verification_agent.verify(english_text, provisions)
    if not is_verified:
        logger.warning("[Pipeline] Verification failed. Triggering Hallucination Guard.")
        fallback = trigger_hallucination_fallback("Clause failed legal verification against statutory text", "Verification")
        formatted = response_agent.format_response(
            fallback,
            original_text=text,
            english_text=english_text,
            detected_lang=detected_lang
        )
        return translation_agent.translate_output(formatted, lang_to_return)

    # Step 6: Reasoning Agent (Gemini Flash Verdict Synthesis)
    logger.info("[Pipeline] Executing Reasoning Agent...")
    reasoning_data = reasoning_agent.reason(english_text, provisions, verification_data)

    # Step 7: Response Agent (Format for Chat)
    logger.info("[Pipeline] Formatting final response...")
    formatted_response = response_agent.format_response(
        reasoning_data,
        original_text=text,
        english_text=english_text,
        detected_lang=detected_lang,
        provisions=provisions
    )

    # Step 8: Translation Agent (Output)
    final_output = translation_agent.translate_output(formatted_response, lang_to_return)
    logger.info(f"[Pipeline] Pipeline complete. Final Verdict: {final_output.get('verdict')}")
    return final_output
