"""
RakshakAI Pipeline Test Script
Evaluates 3-4 sample clauses across gig work, digital lending, compliant terms, and entity whitelisting.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add backend directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
sys.path.insert(0, BACKEND_DIR)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv(os.path.join(BASE_DIR, ".env"))


from pipeline import run_pipeline

TEST_CASES = [
    {
        "id": "TEST_CASE_1",
        "title": "Gig Contract Arbitrary Deactivation Clause",
        "category": "Gig Economy / Worker Protection",
        "text": "Platform reserves the right to suspend, block, or terminate this Agreement and deactivate the driver-partner at any point of time without assigning any reason whatsoever or providing prior notice.",
        "entity_name": None
    },
    {
        "id": "TEST_CASE_2",
        "title": "Predatory Digital Lending Contact Harvesting Clause",
        "category": "Loan App / Fintech",
        "text": "Borrower grants the app continuous permission to access full phonebook contacts, call logs, and photo gallery to contact friends, family, and employers during debt recovery.",
        "entity_name": None
    },
    {
        "id": "TEST_CASE_3",
        "title": "Compliant Key Fact Statement (KFS) Clause",
        "category": "Loan App / Fintech",
        "text": "Prior to loan agreement execution, lender provides a standardized Key Fact Statement disclosing the all-inclusive Annual Percentage Rate (APR), fee breakdown, repayment schedule, and designated Nodal Grievance Redressal Officer contact details.",
        "entity_name": "KreditBee"
    },
    {
        "id": "TEST_CASE_4",
        "title": "Unregistered / Illegal Loan App Entity Check",
        "category": "Fintech Whitelist Verification",
        "text": "Instant 5-minute loan disbursal with minimal KYC requirements.",
        "entity_name": "QuickCash Chinese Paisa App 2026"
    }
]

def main():
    print("=" * 80)
    print("🚀 RAKSHAKAI MULTI-AGENT PIPELINE TEST RUN")
    print("=" * 80)

    results = []

    for case in TEST_CASES:
        print(f"\n👉 Running [{case['id']}] {case['title']} ({case['category']})")
        print(f"   Input Text: \"{case['text']}\"")
        if case["entity_name"]:
            print(f"   Entity: \"{case['entity_name']}\"")

        res = run_pipeline(
            text=case["text"],
            entity_name=case["entity_name"]
        )

        results.append({
            "id": case["id"],
            "title": case["title"],
            "result": res
        })

        print("-" * 50)
        print(f"   🛡️ VERDICT:     [{res.get('verdict')}]")
        print(f"   📊 RISK LEVEL:  {res.get('risk_level')}")
        print(f"   📜 CITATION:    {res.get('citation')}")
        print(f"   ⚖️ EXPLANATION: {res.get('explanation')}")
        print(f"   💡 ACTION:      {res.get('recommendation')}")
        print(f"   ⚡ EARLY EXIT:  {res.get('early_exit')}")
        print("-" * 50)

    print("\n" + "=" * 80)
    print("✅ ALL TEST PIPELINE RUNS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
