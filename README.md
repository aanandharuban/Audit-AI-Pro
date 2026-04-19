# 🛡️ Audit-AI Pro — Hybrid Forensic Fraud Detection Engine

Audit-AI Pro is an AI-powered forensic auditing system designed for **Indian accounting standards**, enabling Chartered Accountants and financial analysts to detect fraud, anomalies, and manipulation patterns in financial data.

It combines **machine learning, statistical analysis, and automated training pipelines** to deliver intelligent, adaptive audit insights.

---

## 🚀 Overview

Financial fraud detection is complex and often reactive.
Audit-AI Pro introduces a **proactive, AI-driven approach** to:

* Detect anomalies in ledgers and transactions
* Identify suspicious financial patterns
* Automate audit workflows
* Continuously improve through feedback

---

## ✨ Key Features

### 🧠 Hybrid Intelligence Engine

* **XGBoost (Supervised Learning)** → Risk classification
* **Isolation Forest (Unsupervised Learning)** → Anomaly detection
* Combines both for high-accuracy fraud detection

---

### 🔄 Auto-Trainable System

* Feedback-driven learning pipeline
* Continuously improves with new audit data
* Separate training module for stability

---

### 📊 Forensic Analysis Tools

* **Benford’s Law analysis** for fraud detection
* Risk probability scoring
* Timeline-based anomaly tracking

---

### 🇮🇳 Built for Indian Accounting

* Handles TDS-related patterns
* Detects ledger manipulation
* Designed for real-world audit scenarios in India

---

### 🖥️ Interactive Application Layer

* Main interface via `app.py`
* Clean and modern UI (Black Shine / Glassmorphism style)
* Easy-to-use audit workflow

---

## 🏗️ Architecture

```bash id="n4hj6r"
Audit-AI-Pro/
│
├── app.py                     # Inference engine (main app)
├── trainer_module.py          # Autonomous training pipeline
├── audit_model.py             # ML model logic
├── audit_features.py          # Feature engineering
│
├── models/
│   ├── audit_brain_xgb.pkl
│   ├── anomaly_detector.pkl
│   └── text_processor.pkl
│
├── data/
│   ├── indian_audit_data.csv
│   ├── master_audit_dataset.csv
│   ├── training_data_v2.csv
│   └── client_ledger_demo.csv
│
├── scripts/
│   ├── train_auditor.py
│   ├── master_train_data.py
│   ├── generate_indian_data.py
│   └── generate_demo_ledger.py
│
├── outputs/
│   └── audit_report.csv
│
├── test_single_invoice.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

1. Input financial dataset (ledger / transactions)
2. Feature engineering extracts audit signals
3. ML models analyze:

   * Risk probability
   * Anomaly detection
4. Output:

   * Audit report
   * Fraud indicators
   * Risk scores

---

## ▶️ How to Run

### 1. Clone the repository

```bash id="a7yyxq"
git clone https://github.com/aanandharuban/Audit-AI-Pro.git
cd Audit-AI-Pro
```

---

### 2. Install dependencies

```bash id="szl91j"
pip install -r requirements.txt
```

---

### 3. Run the application

```bash id="pjbbpk"
python app.py
```

---

### 4. (Optional) Train the model

```bash id="h0kqne"
python trainer_module.py
```

---

## 📊 Sample Use Cases

* 🧾 Ledger fraud detection
* 📉 Suspicious transaction identification
* 🧠 AI-assisted auditing for CAs
* 📊 Financial anomaly detection
* 🏦 FinTech risk analysis systems

---

## 💡 Highlights

* Real-world financial datasets included
* Hybrid ML architecture (rare in student projects 🔥)
* Auto-learning system (feedback loop)
* Domain-specific (Indian accounting — strong niche advantage)

---

## 🔮 Future Improvements

* Web dashboard for visualization
* API integration for fintech platforms
* Real-time audit streaming
* Multi-country accounting support
* Explainable AI (XAI) insights

---

## 📫 Contact

GitHub: https://github.com/aanandharuban

---

## 🏁 Conclusion

Audit-AI Pro showcases the development of a **domain-focused AI system** that combines machine learning with financial intelligence to solve real-world auditing challenges.

This project demonstrates capabilities in:

* AI model building
* Data engineering
* System architecture
* Practical problem solving in fintech
