"""
RakshakAI Whitelist Agent
Verifies lending entities and digital lending apps (DLAs) against the official RBI Registered Entities
and Whitelist snapshot using rapidfuzz fuzzy matching.
If an unregistered lending app/entity is detected -> returns RED verdict immediately.
"""

from rapidfuzz import process, fuzz
import logging

logger = logging.getLogger("RakshakAI.WhitelistAgent")

# Snapshot of RBI Registered Banks, NBFCs, and Authorized Digital Lending Partners (DLAs)
RBI_WHITELIST_SNAPSHOT = [
    {"name": "State Bank of India", "type": "Scheduled Commercial Bank", "aliases": ["SBI", "YONO SBI"]},
    {"name": "HDFC Bank Ltd", "type": "Scheduled Commercial Bank", "aliases": ["HDFC", "PayZapp"]},
    {"name": "ICICI Bank Ltd", "type": "Scheduled Commercial Bank", "aliases": ["ICICI", "iMobile"]},
    {"name": "Axis Bank Ltd", "type": "Scheduled Commercial Bank", "aliases": ["Axis Bank", "Freecharge"]},
    {"name": "Kotak Mahindra Bank", "type": "Scheduled Commercial Bank", "aliases": ["Kotak 811", "Kotak"]},
    {"name": "Bajaj Finance Limited", "type": "NBFC-UL", "aliases": ["Bajaj Finserv", "Bajaj Markets"]},
    {"name": "Tata Capital Financial Services Ltd", "type": "NBFC-ICC", "aliases": ["Tata Capital", "Tata Neu"]},
    {"name": "KrazyBee Services Pvt Ltd", "type": "NBFC-ND-SI", "aliases": ["KreditBee", "KrazyBee"]},
    {"name": "Navi Finserv Limited", "type": "NBFC-ND-SI", "aliases": ["Navi", "Navi Loans"]},
    {"name": "DMI Finance Pvt Ltd", "type": "NBFC-ND-SI", "aliases": ["DMI Finance", "Samsung Finance+"]},
    {"name": "Aditya Birla Finance Ltd", "type": "NBFC-UL", "aliases": ["Aditya Birla Capital", "ABFL"]},
    {"name": "Poonawalla Fincorp Ltd", "type": "NBFC-ND-SI", "aliases": ["Poonawalla", "Magma Fincorp"]},
    {"name": "Muthoot Finance Ltd", "type": "NBFC-ND-SI", "aliases": ["Muthoot", "Muthoot Gold Loan"]},
    {"name": "Credit Saison India (Kisetsu Saison)", "type": "NBFC-ND-SI", "aliases": ["Credit Saison", "Saison"]},
    {"name": "PayU Finance India Pvt Ltd", "type": "NBFC-ND-SI", "aliases": ["PayU", "LazyPay"]},
    {"name": "InCred Financial Services Ltd", "type": "NBFC-ND-SI", "aliases": ["InCred", "InCred Finance"]},
    {"name": "L&T Finance Ltd", "type": "NBFC-ND-SI", "aliases": ["L&T Finance", "PLANET app"]},
    {"name": "Piramal Capital & Housing Finance Ltd", "type": "NBFC-HFC", "aliases": ["Piramal Finance"]},
    {"name": "EarlySalary Services Pvt Ltd (Fibe)", "type": "NBFC-ND-SI", "aliases": ["Fibe", "EarlySalary"]},
    {"name": "MAS Financial Services Ltd", "type": "NBFC-ND-SI", "aliases": ["MAS Financial"]},
    {"name": "Northern Arc Capital Ltd", "type": "NBFC-ND-SI", "aliases": ["Northern Arc"]},
    {"name": "Whizdm Finance Pvt Ltd (Money View)", "type": "NBFC-ND", "aliases": ["Money View", "MoneyView"]},
    {"name": "Vivriti Capital Ltd", "type": "NBFC-ND-SI", "aliases": ["Vivriti Capital"]}
]

# Build flat lookup list of names and aliases
LOOKUP_ENTITIES = []
ENTITY_MAP = {}

for item in RBI_WHITELIST_SNAPSHOT:
    LOOKUP_ENTITIES.append(item["name"])
    ENTITY_MAP[item["name"]] = item
    for alias in item.get("aliases", []):
        LOOKUP_ENTITIES.append(alias)
        ENTITY_MAP[alias] = item

def check_whitelist(entity_query: str, threshold: int = 80) -> dict:
    """
    Checks if a given app or entity name is in the RBI DLA whitelist.
    Returns result dictionary. If no match is found for a lending entity query -> returns RED immediately.
    """
    if not entity_query or not entity_query.strip():
        return {
            "is_whitelisted": False,
            "checked": False,
            "matched_entity": None,
            "score": 0,
            "status": "NO_ENTITY_PROVIDED"
        }

    clean_query = entity_query.strip()
    match = process.extractOne(
        clean_query,
        LOOKUP_ENTITIES,
        scorer=fuzz.token_sort_ratio
    )

    if match and match[1] >= threshold:
        matched_key = match[0]
        entity_info = ENTITY_MAP.get(matched_key, {"name": matched_key, "type": "Regulated Entity"})
        logger.info(f"[Whitelist] Match found: '{clean_query}' -> '{matched_key}' (score={match[1]})")
        return {
            "is_whitelisted": True,
            "checked": True,
            "matched_entity": entity_info["name"],
            "entity_type": entity_info["type"],
            "score": match[1],
            "verdict": "GREEN",
            "status": "WHITELISTED"
        }
    else:
        logger.warning(f"[Whitelist] No match found for lending app: '{clean_query}'")
        return {
            "is_whitelisted": False,
            "checked": True,
            "matched_entity": None,
            "score": match[1] if match else 0,
            "verdict": "RED",
            "status": "UNREGISTERED_LENDING_ENTITY",
            "citation": "RBI Digital Lending Guidelines 2022 / Digital Lending Directions 2025",
            "explanation": (
                f"The lending application/entity '{clean_query}' was not found in the official RBI Registered "
                "Banks/NBFCs and Digital Lending Apps (DLA) whitelist. Operating digital lending without licensed "
                "regulatory backing is prohibited under RBI guidelines."
            ),
            "disclaimer": "Verify entity credentials directly on rbi.org.in before entering any financial transaction.",
            "early_exit": True
        }
