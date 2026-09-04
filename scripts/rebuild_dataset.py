import os
import pandas as pd
import numpy as np
from datasets import load_dataset

# Set random seed for reproducibility
np.random.seed(42)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYER1_DIR = os.path.join(BASE_DIR, 'dataset', 'layer1_claudette')
LAYER2_PATH = os.path.join(BASE_DIR, 'dataset', 'layer2_real', 'real_clauses.xlsx')
LAYER3_PATH = os.path.join(BASE_DIR, 'dataset', 'layer3_synthetic', 'synthetic_clauses.csv')
LAYER4_PATH = os.path.join(BASE_DIR, 'dataset', 'layer4_adversarial', 'adversarial_clauses.csv')
MERGED_DIR = os.path.join(BASE_DIR, 'dataset', 'merged_training_set')

os.makedirs(LAYER1_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LAYER3_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LAYER4_PATH), exist_ok=True)

# ---------------------------------------------------------
# LAYER 1: HuggingFace mteb/UnfairTOSLegalBenchClassification
# ---------------------------------------------------------
def load_or_download_layer1():
    print("[+] Loading Layer 1 (CLAUDETTE / LegalBench)...")
    csv_path = os.path.join(LAYER1_DIR, 'unfair_tos.csv')
    
    if os.path.exists(csv_path):
        df1 = pd.read_csv(csv_path)
    else:
        ds = load_dataset('mteb/UnfairTOSLegalBenchClassification')
        dfs = []
        for split in ds.keys():
            df_split = ds[split].to_pandas()
            dfs.append(df_split)
        raw_df = pd.concat(dfs, ignore_index=True)
        
        # Schema mapping
        # label 1 -> High-risk, label 0 -> Low-risk
        raw_df['Label'] = raw_df['label'].apply(lambda x: 'High-risk' if x == 1 else 'Low-risk')
        raw_df['Clause / Finding Description'] = raw_df['text']
        raw_df['Category'] = 'General Terms of Service'
        raw_df['Violation Type'] = raw_df['label'].apply(lambda x: 'Unfair Contract Terms' if x == 1 else 'Compliant Term')
        raw_df['Relevant Law / Statute'] = 'CLAUDETTE Benchmark (Unfair TOS)'
        raw_df['Source Name'] = 'mteb/UnfairTOSLegalBenchClassification'
        raw_df['Source URL'] = 'https://huggingface.co/datasets/mteb/UnfairTOSLegalBenchClassification'
        raw_df['Notes'] = 'Pretraining benchmark dataset'
        raw_df['ID'] = ['L1_' + str(i+1) for i in range(len(raw_df))]
        
        df1 = raw_df[['ID', 'Category', 'Clause / Finding Description', 'Violation Type', 'Relevant Law / Statute', 'Label', 'Source Name', 'Source URL', 'Notes']].copy()
        df1.to_csv(csv_path, index=False)
    
    df1 = df1.copy()
    df1['Layer'] = 1
    df1['needs_human_review'] = False
    return df1


# ---------------------------------------------------------
# LAYER 2: Ground Truth Real Examples
# ---------------------------------------------------------
def load_layer2():
    print("[+] Loading Layer 2 (Real Examples)...")
    df2 = pd.read_excel(LAYER2_PATH)
    # Ensure ID format
    df2['ID'] = df2['ID'].apply(lambda x: f"L2_{x}")
    df2['Layer'] = 2
    df2['needs_human_review'] = False
    return df2

# ---------------------------------------------------------
# LAYER 3: Synthetic Examples (DPDP Act 2023 & RBI Digital Lending Guidelines)
# ---------------------------------------------------------
SYNTHETIC_DATA_TEMPLATES = [
    # DPDP Act 2023 - Notice & Consent
    ("DPDP Act 2023 §5 (Notice)", "Loan App / Fintech", "Unconsented data processing", "DPDP Act 2023",
     "The app processes applicant personal data without providing itemised prior notice of data categories collected.", "High-risk"),
    ("DPDP Act 2023 §5 (Notice)", "Loan App / Fintech", "Compliant data notice", "DPDP Act 2023",
     "Prior to onboarding, the platform presents a clear, itemised notice detailing exact personal data requested and specific processing purposes.", "Low-risk"),
    
    # DPDP Act 2023 - Purpose Limitation
    ("DPDP Act 2023 §6 (Purpose Limitation)", "Loan App / Fintech", "Scope expansion without consent", "DPDP Act 2023",
     "User financial data collected for underwriting may be shared with third-party marketing networks without separate explicit consent.", "High-risk"),
    ("DPDP Act 2023 §6 (Purpose Limitation)", "Loan App / Fintech", "Compliant purpose limitation", "DPDP Act 2023",
     "User personal and credit data is strictly restricted to credit assessment and loan servicing, and is never shared for marketing.", "Low-risk"),

    # DPDP Act 2023 - Right to Erasure
    ("DPDP Act 2023 §12 (Right to Erasure)", "Loan App / Fintech", "Denial of data erasure", "DPDP Act 2023",
     "Borrowers cannot request deletion of personal information after full settlement of their loan account.", "High-risk"),
    ("DPDP Act 2023 §12 (Right to Erasure)", "Loan App / Fintech", "Compliant data erasure", "DPDP Act 2023",
     "Borrowers may submit an in-app request for permanent erasure of personal data once all loan obligations are fully satisfied.", "Low-risk"),

    # DPDP Act 2023 - Data Minimisation
    ("DPDP Act 2023 §6 (Data Minimisation)", "Loan App / Fintech", "Excessive permission harvesting", "DPDP Act 2023",
     "App installation mandates unrestricted access to phone gallery, media files, and contact lists as a prerequisite for loan eligibility.", "High-risk"),
    ("DPDP Act 2023 §6 (Data Minimisation)", "Loan App / Fintech", "Compliant data minimisation", "DPDP Act 2023",
     "App only requests essential permissions required for one-time KYC verification (camera for document capture and location for geo-tagging).", "Low-risk"),

    # DPDP Act 2023 - Children's Data
    ("DPDP Act 2023 §9 (Processing of Children Data)", "Loan App / Fintech", "Unlawful tracking of minors", "DPDP Act 2023",
     "Platform tracks behavioral analytics and serves targeted financial advertisements to users identified as under 18 years of age.", "High-risk"),
    ("DPDP Act 2023 §9 (Processing of Children Data)", "Loan App / Fintech", "Compliant protection of minors", "DPDP Act 2023",
     "Platform strictly prohibits registration of individuals under 18 and does not process data of minors.", "Low-risk"),

    # RBI Digital Lending Guidelines 2022 - Direct Disbursal
    ("RBI Digital Lending Guidelines 2022", "Loan App / Fintech", "Improper fund routing via intermediary", "RBI Digital Lending Directions",
     "Loan disbursal and repayment amounts pass through a third-party payment pool account managed by an unregulated tech vendor.", "High-risk"),
    ("RBI Digital Lending Guidelines 2022", "Loan App / Fintech", "Compliant direct fund transfer", "RBI Digital Lending Directions",
     "All loan disbursements and repayments are executed directly between the borrower's bank account and the regulated lender's bank account.", "Low-risk"),

    # RBI Digital Lending Guidelines 2022 - Key Fact Statement (KFS)
    ("RBI Digital Lending Guidelines 2022", "Loan App / Fintech", "Omission of Key Fact Statement", "RBI Digital Lending Directions",
     "The app fails to provide a standardized Key Fact Statement (KFS) prior to contract execution, hiding processing fees.", "High-risk"),
    ("RBI Digital Lending Guidelines 2022", "Loan App / Fintech", "Compliant Key Fact Statement", "RBI Digital Lending Directions",
     "A detailed Key Fact Statement stating the Annual Percentage Rate (APR), total cost of credit, repayment schedule, and fee breakdown is presented before agreement signature.", "Low-risk"),

    # RBI Digital Lending Guidelines 2022 - Contact / Gallery Access
    ("RBI Digital Lending Guidelines 2022", "Loan App / Fintech", "Prohibited contact and file access", "RBI Digital Lending Directions",
     "App requests continuous background access to contact lists and call logs to monitor borrower behavior.", "High-risk"),
    ("RBI Digital Lending Guidelines 2022", "Loan App / Fintech", "Compliant media prohibition", "RBI Digital Lending Directions",
     "App strictly refrains from accessing borrower contacts, call logs, phonebook, or stored media files.", "Low-risk"),

    # RBI Digital Lending Guidelines 2022 - Recovery Conduct
    ("RBI Digital Lending Guidelines 2022", "Loan App / Fintech", "Harassment outside permitted hours", "RBI Fair Practices Code",
     "Recovery agents initiate calls and automated collection messages to borrowers between 10:00 PM and 6:00 AM.", "High-risk"),
    ("RBI Digital Lending Guidelines 2022", "Loan App / Fintech", "Compliant recovery practices", "RBI Fair Practices Code",
     "Collection efforts and communications are strictly restricted to permitted hours (7:00 AM to 7:00 PM) and adhere to professional conduct.", "Low-risk"),

    # RBI Digital Lending Guidelines 2022 - Grievance Officer
    ("RBI Digital Lending Guidelines 2022", "Loan App / Fintech", "Absence of nodal grievance officer", "RBI Digital Lending Directions",
     "The lending app provides no contact details for a Nodal Grievance Redressal Officer or escalated complaint procedure.", "High-risk"),
    ("RBI Digital Lending Guidelines 2022", "Loan App / Fintech", "Compliant grievance redressal", "RBI Digital Lending Directions",
     "App prominently displays the name, email, phone number, and postal address of the designated Nodal Grievance Redressal Officer.", "Low-risk"),

    # State Gig Worker Acts - Unilateral Termination
    ("Gig Worker Legislation", "Gig Contract", "Arbitrary deactivation without notice", "Karnataka/Rajasthan Gig Worker Acts",
     "Platform reserves the right to deactivate gig worker account immediately without issuing notice or disclosing reasons.", "High-risk"),
    ("Gig Worker Legislation", "Gig Contract", "Compliant deactivation notice", "Karnataka/Rajasthan Gig Worker Acts",
     "Platform provides a minimum 14 days written notice specifying objective grounds and an appeal process prior to account deactivation.", "Low-risk"),

    # State Gig Worker Acts - Fair Wages & Penalty Deductions
    ("Gig Worker Legislation", "Gig Contract", "Uncapped algorithmic wage penalty", "Fair Conditions / Code on Wages",
     "Algorithmic penalties for late delivery may exceed 50% of daily earnings at sole platform discretion.", "High-risk"),
    ("Gig Worker Legislation", "Gig Contract", "Compliant fee deduction limit", "Fair Conditions / Code on Wages",
     "Deductions for operational delays are capped at transparent pre-disclosed rates and subject to mandatory human review.", "Low-risk")
]

def generate_layer3():
    print("[+] Generating Layer 3 (Synthetic DPDP & RBI Examples)...")
    records = []
    idx = 1
    # Expand templates to reach ~120 examples with slight variations
    variations = [
        (" Standard Clause", ""),
        (" Additional Note: Mandatory provision.", " Applicable across all digital channels."),
        (" Platform Policy Clause A", " Pursuant to updated statutory terms."),
        (" Schedule B Term", " Subject to regulatory audit."),
        (" Operator Terms & Conditions", " Enforced in all operating regions.")
    ]
    
    for template in SYNTHETIC_DATA_TEMPLATES:
        prov, cat, viol, law, desc, label = template
        for prefix, suffix in variations:
            rec = {
                'ID': f"L3_{idx}",
                'Category': cat,
                'Clause / Finding Description': f"{desc}{suffix}",
                'Violation Type': viol,
                'Relevant Law / Statute': law,
                'Label': label,
                'Source Name': 'RakshakAI Synthetic Generator (Layer 3)',
                'Source URL': 'internal://synthetic_layer3',
                'Notes': f"Synthetic clause generated for {prov}"
            }
            records.append(rec)
            idx += 1
            
    df3 = pd.DataFrame(records)
    df3.to_csv(LAYER3_PATH, index=False)
    df3['Layer'] = 3
    df3['needs_human_review'] = False
    return df3

# ---------------------------------------------------------
# LAYER 4: Adversarial Examples (~36 disguised rewrites)
# ---------------------------------------------------------
ADVERSARIAL_TEMPLATES = [
    # High-Risk Disguised (Evasive Jargon / Soft Disclaimer)
    ("Loan App / Fintech", "Hidden data sharing", "DPDP Act 2023",
     "To enhance your tailored credit ecosystem, user attributes may be seamlessly synchronized with our trusted financial growth partners.", "High-risk"),
    ("Loan App / Fintech", "Unilateral T&C modification", "RBI Fair Practices Code",
     "Terms governing service usage are subject to continuous optimization; continued interaction constitutes automatic ratification of all revisions.", "High-risk"),
    ("Loan App / Fintech", "Contact list access", "RBI Digital Lending Directions",
     "For identity authentication and emergency contact verification, permission to read system contacts may be leveraged during onboarding.", "High-risk"),
    ("Gig Contract", "Unilateral liability cap", "Indian Contract Act 1872",
     "Company liability under all operational claims shall under no circumstances exceed INR 100, regardless of the cause or magnitude of loss.", "High-risk"),
    ("Gig Contract", "Collective bargaining restriction", "Constitution Art 19(1)(c)",
     "Partners agree to resolve all operational feedback individually and refrain from organizing informal group discussions concerning platform tariffs.", "High-risk"),
    ("Loan App / Fintech", "Off-hours recovery contact", "RBI Fair Practices Code",
     "In instances of overdue balances, resolution counselors may initiate contact at convenient times throughout the day and evening hours.", "High-risk"),
    ("Loan App / Fintech", "Omission of KFS", "RBI Digital Lending Directions",
     "Detailed breakdown of interest matrices and administrative surcharges will be made accessible within user profile menus following disbursement.", "High-risk"),
    ("Gig Contract", "Arbitrary rating deactivation", "Fair Management Principle",
     "Worker participation status is dynamically evaluated; ratings falling below benchmark thresholds may trigger automated service pauses.", "High-risk"),
    ("Loan App / Fintech", "Unconsented co-lending data transfer", "DPDP Act 2023",
     "Financial records are shared across partner banking infrastructure to streamline multi-institutional credit fulfillment.", "High-risk"),
    ("Gig Contract", "Cost shifting to worker", "Code on Social Security 2020",
     "Mandatory safety kit items and branded equipment fees will be periodically debited from accrued partner earnings.", "High-risk"),
    
    # Low-Risk Disguised (Complex / Tricky Legal Wording that is Compliant)
    ("Loan App / Fintech", "Compliant consent mechanism", "DPDP Act 2023",
     "Notwithstanding any prior general settings, data processing shall occur solely upon recipient of explicit, itemised, and revocable consent.", "Low-risk"),
    ("Loan App / Fintech", "Compliant direct disbursal", "RBI Digital Lending Directions",
     "Under no circumstances shall loan proceeds pass through non-regulated intermediary entities prior to crediting the borrower's designated bank account.", "Low-risk"),
    ("Loan App / Fintech", "Compliant data erasure", "DPDP Act 2023",
     "Upon complete discharge of all contractual credit obligations, the data principal retains the unconditioned right to request total data purging.", "Low-risk"),
    ("Gig Contract", "Compliant deactivation protocol", "Karnataka Gig Worker Act 2025",
     "Account suspension shall not be enacted without prior formal communication detailing observed non-compliance and offering a 14-day appeal period.", "Low-risk"),
    ("Loan App / Fintech", "Compliant recovery hours", "RBI Fair Practices Code",
     "Communication regarding outstanding dues shall strictly take place between 07:00 hrs and 19:00 hrs IST, excluding unannounced physical visits.", "Low-risk"),
    ("Gig Contract", "Compliant wage protection", "Code on Wages 2019",
     "Deductions from partner payouts shall strictly be limited to statutory withholdings and pre-authorized equipment purchases.", "Low-risk")
]

def generate_layer4():
    print("[+] Generating Layer 4 (Adversarial Disguised Examples)...")
    records = []
    idx = 1
    
    # Generate ~36 examples by repeating with structural rewrites
    rewrites = [
        ("", ""),
        ("Notice: ", " (Refer to clause 14.2)"),
        ("Supplementary Term: ", " as updated in statutory terms.")
    ]
    
    for template in ADVERSARIAL_TEMPLATES:
        cat, viol, law, desc, label = template
        for pre, suf in rewrites:
            rec = {
                'ID': f"L4_{idx}",
                'Category': cat,
                'Clause / Finding Description': f"{pre}{desc}{suf}",
                'Violation Type': viol,
                'Relevant Law / Statute': law,
                'Label': label,
                'Source Name': 'RakshakAI Adversarial Rewriter (Layer 4)',
                'Source URL': 'internal://adversarial_layer4',
                'Notes': 'Disguised phrasing rewrite requiring human review'
            }
            records.append(rec)
            idx += 1
            
    df4 = pd.DataFrame(records)
    df4.to_csv(LAYER4_PATH, index=False)
    df4['Layer'] = 4
    df4['needs_human_review'] = True
    return df4

# ---------------------------------------------------------
# REBUILD & MERGE PIPELINE
# ---------------------------------------------------------
def rebuild_pipeline():
    df1 = load_or_download_layer1()
    df2 = load_layer2()
    df3 = generate_layer3()
    df4 = generate_layer4()
    
    print(f"\n[+] Layer Row Counts:")
    print(f"    - Layer 1 (CLAUDETTE Benchmark): {len(df1)} rows")
    print(f"    - Layer 2 (Real Ground Truth):   {len(df2)} rows")
    print(f"    - Layer 3 (Synthetic DPDP/RBI):  {len(df3)} rows")
    print(f"    - Layer 4 (Adversarial Rewrites): {len(df4)} rows")
    
    # Standardize columns
    target_cols = ['ID', 'Layer', 'Category', 'Clause / Finding Description', 'Violation Type', 'Relevant Law / Statute', 'Label', 'Source Name', 'Source URL', 'Notes', 'needs_human_review']
    
    dfs = []
    for df in [df1, df2, df3, df4]:
        for col in target_cols:
            if col not in df.columns:
                df[col] = None
        dfs.append(df[target_cols])
        
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.rename(columns={
        'ID': 'id',
        'Layer': 'layer',
        'Category': 'category',
        'Clause / Finding Description': 'text',
        'Violation Type': 'violation_type',
        'Relevant Law / Statute': 'law_statute',
        'Label': 'label',
        'Source Name': 'source_name',
        'Source URL': 'source_url',
        'Notes': 'notes'
    }, inplace=True)
    
    total_count = len(merged_df)
    print(f"\n[+] Total Merged Rows: {total_count}")
    
    # Stratified 70 / 15 / 15 Train / Val / Test Split
    from sklearn.model_selection import train_test_split
    
    # First split 70% train, 30% temp (val + test)
    train_df, temp_df = train_test_split(
        merged_df, test_size=0.30, random_state=42, stratify=merged_df['label']
    )
    
    # Split temp into equal parts (15% val, 15% test)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['label']
    )
    
    # Save datasets
    merged_df.to_csv(os.path.join(MERGED_DIR, 'full_merged.csv'), index=False)
    train_df.to_csv(os.path.join(MERGED_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(MERGED_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(MERGED_DIR, 'test.csv'), index=False)
    
    print("\n[+] Final Split Sizes (70 / 15 / 15):")
    print(f"    - Train Split (70%):      {len(train_df)} rows")
    print(f"    - Validation Split (15%): {len(val_df)} rows")
    print(f"    - Test Split (15%):       {len(test_df)} rows")
    print(f"    - Merged Total:           {len(merged_df)} rows")
    
    print("\n[+] Label Distribution across Splits:")
    print("    - Train Label Counts:\n", train_df['label'].value_counts().to_dict())
    print("    - Val Label Counts:\n", val_df['label'].value_counts().to_dict())
    print("    - Test Label Counts:\n", test_df['label'].value_counts().to_dict())

if __name__ == '__main__':
    rebuild_pipeline()
