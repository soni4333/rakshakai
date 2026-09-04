# RakshakAI Evaluation Report

## 1. Overall Performance
- **Pipeline Accuracy:** 48.98%
- **Precision:** 0.4898
- **Recall (High-Risk):** 1.0000
- **F1-Score:** 0.6575

## 2. Confusion Matrix
- **True Positives (TP):** 24
- **True Negatives (TN):** 0
- **False Positives (FP):** 25
- **False Negatives (FN):** 0

## 3. Per-Layer Performance
- **Layer 1**: Precision=0.0, Recall=0.0, F1=0.0 (N=15)
- **Layer 2**: Precision=1.0, Recall=1.0, F1=1.0 (N=11)
- **Layer 3**: Precision=0.5333, Recall=1.0, F1=0.6957 (N=15)
- **Layer 4**: Precision=0.625, Recall=1.0, F1=0.7692 (N=8)

## 4. Hallucination Guard
- **Trigger Count:** 2
- **Trigger Rate:** 4.08%

## 5. Baseline Comparison (Pipeline vs. Direct Gemini)
- **RakshakAI Pipeline F1:** 0.6575 | **Direct Baseline F1:** 0.6857 (Delta: -0.0282)
- **RakshakAI Recall:** 1.0000 | **Direct Baseline Recall:** 1.0000 (Delta: +0.0000)
