# 📄 Invoice Data Parser using LayoutLM

An end-to-end **Document AI** system for extracting key information from **unstructured invoice images** using **OCR + LayoutLM**.

This project focuses on **real-world invoice understanding**, including layout-aware extraction, confidence handling, and production-style APIs.

---

## 🔍 Extracted Fields

- Invoice Number  
- Invoice Date  
- Total Amount  
- Vendor Name  

---

## 🧠 Architecture Overview
Invoice Image
-->
Tesseract OCR
-->
LayoutLMv3 (Token Classification)
-->
Post-processing + Heuristics
-->
Structured Invoice JSON


---

## 🛠 Tech Stack

- Python 3.11
- FastAPI
- Streamlit
- Tesseract OCR
- LayoutLMv3
- PyTorch
- Pytest

---

## 📈 Model Notes

OCR quality directly impacts extraction accuracy

LayoutLM improves robustness by using spatial layout

Heuristic fallback is applied when model confidence is low

Empty output indicates low confidence, not a failure

