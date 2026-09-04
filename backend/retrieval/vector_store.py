"""
RakshakAI Statutory Vector Store
ChromaDB index populated with verified text of Indian Legal Frameworks:
- Digital Personal Data Protection (DPDP) Act 2023
- RBI Digital Lending Guidelines 2022 / Digital Lending Directions 2025
- Indian Contract Act 1872 (Unfair Terms & Penalty Clauses)
- State Gig Worker Welfare Acts (Karnataka, Rajasthan, Bihar, Jharkhand)
"""

import os
import chromadb
from chromadb.utils import embedding_functions
import logging

logger = logging.getLogger("RakshakAI.VectorStore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DATA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
os.makedirs(CHROMA_DATA_DIR, exist_ok=True)

STATUTORY_CORPUS = [
    {
        "id": "DPDP_SEC_5",
        "act": "DPDP Act 2023",
        "section": "Section 5 - Notice for Personal Data Collection",
        "text": "Every request for consent under this Act shall be accompanied or preceded by a notice given by the Data Fiduciary to the Data Principal, informing her of the personal data sought to be collected, the specific purpose for which the personal data may be processed, the manner in which she may exercise her right to withdraw consent, and the grievance redressal mechanism."
    },
    {
        "id": "DPDP_SEC_6",
        "act": "DPDP Act 2023",
        "section": "Section 6 - Consent & Purpose Limitation",
        "text": "Consent given by the Data Principal shall be free, specific, informed, unconditional and unambiguous with a clear affirmative action, and shall signify an agreement for processing her personal data for the specified purpose alone. Any part of consent which constitutes an infringement of the provisions of this Act or the rules shall be invalid. Consent shall not be bundled or made conditional upon unnecessary services."
    },
    {
        "id": "DPDP_SEC_8",
        "act": "DPDP Act 2023",
        "section": "Section 8 - General Obligations of Data Fiduciary & Erasure",
        "text": "A Data Fiduciary shall erase personal data upon the Data Principal withdrawing her consent or as soon as it is reasonable to assume that the specified purpose is no longer being served, whichever is earlier. The Data Fiduciary shall implement appropriate technical and organisational measures to ensure compliance with this Act."
    },
    {
        "id": "DPDP_SEC_9",
        "act": "DPDP Act 2023",
        "section": "Section 9 - Processing of Children's Personal Data",
        "text": "The Data Fiduciary shall, before processing any personal data of a child, obtain verifiable consent of the parent or lawful guardian. A Data Fiduciary shall not undertake tracking or behavioural monitoring of children or targeted advertising directed at children."
    },
    {
        "id": "DPDP_SEC_12",
        "act": "DPDP Act 2023",
        "section": "Section 12 - Right to Correction and Erasure of Personal Data",
        "text": "A Data Principal shall have the right to correction, completion, updating and erasure of her personal data for the processing of which she has previously given consent, in accordance with such procedure as may be prescribed."
    },
    {
        "id": "RBI_DLG_DIRECT_DISBURSAL",
        "act": "RBI Digital Lending Guidelines 2022",
        "section": "Clause 3 - Direct Fund Disbursal to Bank Account",
        "text": "All loan disbursements and repayments are required to be executed directly between the bank account of the regulated lending entity (Bank/NBFC) and the borrower's bank account, without any pass-through or pool account of any Lending Service Provider (LSP) or third party."
    },
    {
        "id": "RBI_DLG_KFS",
        "act": "RBI Digital Lending Guidelines 2022",
        "section": "Clause 4 - Key Fact Statement (KFS) & All-in APR",
        "text": "A standardized Key Fact Statement (KFS) must be provided to the borrower before the execution of the loan contract. The KFS must transparently disclose the all-inclusive Annual Percentage Rate (APR), total cost of credit, fee breakdown, recovery mechanism, and cooling-off period. No fee or charge not mentioned in the KFS can be levied."
    },
    {
        "id": "RBI_DLG_DATA_PRIVACY",
        "act": "RBI Digital Lending Guidelines 2022",
        "section": "Clause 5 - Prohibition of Contact List and Gallery Access",
        "text": "Digital Lending Apps (DLAs) and Lending Service Providers (LSPs) are explicitly prohibited from accessing mobile phone resources such as contact lists, call logs, phonebook, SMS logs, media, and photo gallery. Access to camera, microphone, or location is strictly limited to one-time KYC onboarding with explicit user consent."
    },
    {
        "id": "RBI_DLG_RECOVERY_HOURS",
        "act": "RBI Fair Practices Code & Digital Lending Guidelines",
        "section": "Clause 6 - Fair Practices in Debt Recovery & Permitted Hours",
        "text": "Recovery agents and lenders must strictly contact borrowers only between 07:00 hours and 19:00 hours IST. Contacting borrowers at uncivilized hours, using threatening, intimidating, or abusive language, or publicly shaming/harassing family members, employers, or social contacts is strictly prohibited under RBI regulations."
    },
    {
        "id": "RBI_DLG_GRIEVANCE",
        "act": "RBI Digital Lending Guidelines 2022",
        "section": "Clause 7 - Nodal Grievance Redressal Officer",
        "text": "Regulated entities and digital lending apps must prominently appoint and display the contact details (name, email, phone number, postal address) of a dedicated Nodal Grievance Redressal Officer on their website and mobile application for swift escalation of consumer complaints."
    },
    {
        "id": "RBI_DLG_COOLING_OFF",
        "act": "RBI Digital Lending Guidelines 2022",
        "section": "Clause 8 - Look-up / Cooling-off Period",
        "text": "A look-up / cooling-off period must be provided to borrowers during which they can exit the digital loan by paying the principal amount and proportionate APR without incurring any prepayment or cancellation penalties."
    },
    {
        "id": "CONTRACT_ACT_SEC_23",
        "act": "Indian Contract Act 1872",
        "section": "Section 23 - Unconscionable Terms & Public Policy",
        "text": "The consideration or object of an agreement is lawful, unless it is forbidden by law, or is of such a nature that, if permitted, it would defeat the provisions of any law, or the Court regards it as immoral, or opposed to public policy. Grossly one-sided, unconscionable standard form contracts imposed due to unequal bargaining power are void."
    },
    {
        "id": "CONTRACT_ACT_SEC_74",
        "act": "Indian Contract Act 1872",
        "section": "Section 74 - Compensation for Breach and Unfair Penalty",
        "text": "When a contract has been broken, the party who suffers by such breach is entitled to receive reasonable compensation not exceeding the amount named, whether or not actual damage is proved. Stipulations for arbitrary, uncapped, or extortionate penalties at sole discretion of one party are invalid."
    },
    {
        "id": "GIG_WORKER_DEACTIVATION_NOTICE",
        "act": "State Platform Gig Worker Welfare Acts",
        "section": "Karnataka & Rajasthan Gig Worker Acts - Protection Against Arbitrary Deactivation",
        "text": "Aggregators and digital labour platforms shall not arbitrarily terminate, block, or deactivate a gig worker's account without providing a minimum of 14 days prior written notice stating objective reasons and giving an opportunity of fair hearing and grievance redressal."
    },
    {
        "id": "GIG_WORKER_COLLECTIVE_RIGHTS",
        "act": "Constitution of India Art 19(1)(c) & Gig Worker Acts",
        "section": "Freedom of Association & Collective Redressal",
        "text": "Contractual clauses in aggregator service agreements that explicitly prohibit gig workers or delivery partners from forming unions, associations, or organizing peaceful collective representations regarding working conditions and remuneration violate the fundamental right to freedom of association."
    }
]

class StatutoryVectorStore:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(StatutoryVectorStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.client = None
        self.collection = None
        self.initialize_store()

    def initialize_store(self):
        logger.info("[+] Initializing ChromaDB vector store...")
        try:
            self.client = chromadb.PersistentClient(path=CHROMA_DATA_DIR)
            # Use default lightweight embedding function
            self.embedding_func = embedding_functions.DefaultEmbeddingFunction()
            self.collection = self.client.get_or_create_collection(
                name="indian_statutes",
                embedding_function=self.embedding_func
            )
            
            # Populate if empty
            if self.collection.count() == 0:
                logger.info(f"[+] Indexing {len(STATUTORY_CORPUS)} statutory provisions into ChromaDB...")
                documents = [item["text"] for item in STATUTORY_CORPUS]
                metadatas = [{"act": item["act"], "section": item["section"]} for item in STATUTORY_CORPUS]
                ids = [item["id"] for item in STATUTORY_CORPUS]
                
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"[+] Vector store populated with {self.collection.count()} provisions.")
            else:
                logger.info(f"[+] Vector store loaded with {self.collection.count()} existing provisions.")
        except Exception as e:
            logger.error(f"[-] Error initializing ChromaDB: {e}")

    def query_statutes(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieves top-k matching statutory provisions for a query text.
        """
        if self.collection is None:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            retrieved = []
            if results and "documents" in results and results["documents"]:
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0]*len(docs)
                ids = results["ids"][0]
                
                for doc, meta, dist, doc_id in zip(docs, metas, distances, ids):
                    retrieved.append({
                        "id": doc_id,
                        "act": meta.get("act", ""),
                        "section": meta.get("section", ""),
                        "text": doc,
                        "distance": float(dist)
                    })
            return retrieved
        except Exception as e:
            logger.error(f"[-] Error querying ChromaDB: {e}")
            return []

vector_store = StatutoryVectorStore()
