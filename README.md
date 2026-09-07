# 🧠 Arabic FAQ RAG System

> **AI-Powered Arabic Question Answering using Semantic Retrieval, Intent Classification, and Intent-Aware Reranking**

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-red)](https://arabic-faq-rag.streamlit.app/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20%26%20Artifacts-yellow)](https://huggingface.co/AbdelrahmanAkl/Arabic-FAQ-RAG)
[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)](https://github.com/facebookresearch/faiss)

---

## 🚀 Live Demo

### Try the application directly:

**Arabic FAQ RAG — Live Demo**

https://arabic-faq-rag.streamlit.app/

The deployed application allows users to ask Arabic questions and retrieve the most relevant FAQ answers using semantic search and intent-aware reranking.

---

## 📌 Project Overview

**Arabic FAQ RAG** is an intelligent Arabic question-answering and retrieval system designed to solve a common problem in FAQ and customer-support systems:

> How can we retrieve the correct answer when users ask the same question using completely different wording?

Traditional keyword-based search can fail when two questions have similar meanings but different words.

For example:

**Knowledge Base Question:**

> كيف يمكنني إغلاق حسابي؟

**User Question:**

> عايز أقفل حسابي نهائيًا، أعمل إيه؟

Although the wording is different, the intent and meaning are highly similar.

This project addresses this problem using:

* **Multilingual E5 embeddings**
* **Semantic vector search**
* **FAISS**
* **Arabic intent classification**
* **Intent-aware reranking**
* **Streamlit deployment**

---

## 🎯 Main Objective

The main goal is to build a retrieval system that can:

1. Understand the semantic meaning of Arabic questions.
2. Identify the user's intent.
3. Search a large FAQ knowledge base efficiently.
4. Retrieve the most semantically relevant candidates.
5. Re-rank the candidates according to both semantic similarity and intent compatibility.
6. Return the best matching FAQ answer and its source information.

---

# 🏗️ System Architecture

```text
                    User
                     │
                     ▼
             Arabic Question
                     │
                     ▼
        ┌─────────────────────────┐
        │   Intent Classification │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │ Multilingual E5 Encoder │
        │   Query Embedding       │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │      FAISS Search       │
        │  Semantic Vector Search │
        └────────────┬────────────┘
                     │
                     ▼
              Top-K Candidates
                     │
                     ▼
        ┌─────────────────────────┐
        │ Intent-Aware Reranking  │
        │ Similarity + Intent     │
        └────────────┬────────────┘
                     │
                     ▼
               Top Results
                     │
                     ▼
        Answer + Source Information
                     │
                     ▼
               Streamlit UI
```

---

# 🔍 How It Works

## 1. Intent Classification

The system first predicts the intent behind the user's question.

Examples:

```text
"أريد إغلاق حسابي"
→ Account / Subscription Management

"ليه اتخصم مني مبلغ؟"
→ Billing Dispute

"الدفع فشل عندي"
→ Troubleshooting
```

This provides an additional understanding signal beyond semantic similarity.

---

## 2. Semantic Embeddings

The system uses:

**`intfloat/multilingual-e5-base`**

to transform Arabic questions into dense vector representations.

Instead of relying only on exact keyword matching, the system compares the semantic meaning of questions.

```text
Question
   ↓
E5 Embedding
   ↓
Dense Vector
```

---

## 3. FAISS Vector Search

The generated query embedding is searched against the FAQ vector index using **FAISS**.

The knowledge base contains approximately:

**164,456 FAQ chunks**

with:

* 768-dimensional embeddings
* Arabic questions and answers
* company information
* domain information
* country information
* intent/question-type metadata

FAISS retrieves the most semantically similar candidates efficiently.

---

## 4. Intent-Aware Reranking

The initial retrieval stage returns the top candidates based on semantic similarity.

The system then applies an additional reranking step.

The final score is based on:

```text
Final Score =
E5 Similarity Score
+
Intent Matching Signal
```

This helps prioritize results that are not only semantically similar, but also aligned with the predicted user intent.

---

## 5. Final Answer

The system returns the best matching FAQ answer together with useful source metadata such as:

* FAQ ID
* Company
* Country
* Domain
* Intent
* Similarity score
* Final reranking score

This makes the retrieval process more transparent and easier to inspect.

---

# ✨ Key Features

* 🇦🇪 Arabic semantic question retrieval
* 🧠 Intent classification
* 🔎 Semantic search with Multilingual E5
* ⚡ FAISS vector similarity search
* 🎯 Intent-aware reranking
* 📚 Large-scale FAQ knowledge base
* 📊 Retrieval scores and source transparency
* 🌐 Streamlit web application
* 🤗 Hugging Face artifact hosting
* ☁️ Cloud deployment
* 💻 Local deployment support

---

# 🛠️ Tech Stack

| Component           | Technology                      |
| ------------------- | ------------------------------- |
| Language            | Python                          |
| UI                  | Streamlit                       |
| Embeddings          | `intfloat/multilingual-e5-base` |
| Vector Search       | FAISS                           |
| ML                  | Scikit-learn                    |
| Data Processing     | Pandas / PyArrow                |
| Model Serialization | Joblib                          |
| Artifact Hosting    | Hugging Face Hub                |
| Source Control      | Git / GitHub                    |
| Deployment          | Streamlit Community Cloud       |

---

# 📂 Project Structure

```text
Arabic-FAQ-RAG/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── intent.py
│   ├── retrieval.py
│   └── reranking.py
│
├── notebooks/
│   └── Arabic_FAQ_RAG.ipynb
│
├── data/
│   ├── train_chunks.faiss
│   └── train_chunks_metadata_v2.parquet
│
└── models/
    └── intent_classifier.joblib
```

Large artifacts are hosted separately on Hugging Face rather than stored directly in the Git repository.

---

# 📊 Knowledge Base

The retrieval index contains approximately:

* **164,456 indexed FAQ chunks**
* **768-dimensional embeddings**
* Arabic questions and answers
* Multiple companies
* Multiple domains
* Multiple countries
* Intent/question-type metadata

The system uses these indexed vectors to perform semantic retrieval.

---

# 🧪 Example Queries

### Example 1 — Account Management

```text
عايز أقفل حسابي نهائيًا، أعمل إيه؟
```

### Example 2 — Billing

```text
ليه لقيت رسوم زيادة في الفاتورة بتاعتي؟
```

### Example 3 — Payment

```text
الدفع فشل عندي والمبلغ اتخصم، هيرجع إمتى؟
```

### Example 4 — Refund

```text
إيه شروط استرجاع فلوسي؟
```

### Example 5 — Subscription

```text
هل فيه فترة مجانية قبل ما أبدأ الاشتراك؟
```

---

# 🧪 Evaluation & Testing

The system was tested using multiple Arabic queries covering different intents and user phrasings.

Testing focused on:

* Intent prediction
* Semantic retrieval quality
* Top-K candidate retrieval
* Reranking behavior
* Final answer relevance
* Edge cases
* Source transparency

The evaluation process was designed to verify that the system can retrieve semantically relevant FAQs even when the user's wording differs from the original knowledge-base question.

---

# 💻 Local Installation

Clone the repository:

```bash
git clone https://github.com/AbdelrhmanAkl/Arabic-FAQ-RAG.git
cd Arabic-FAQ-RAG
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will automatically use local artifacts when available.

If the artifacts are not present locally, the application downloads them from the Hugging Face repository.

---

# 🤗 Model & Data Artifacts

The large retrieval and classification artifacts are hosted on Hugging Face:

**Arabic FAQ RAG — Hugging Face**

https://huggingface.co/AbdelrahmanAkl/Arabic-FAQ-RAG

Hosted artifacts include:

```text
data/
├── train_chunks.faiss
└── train_chunks_metadata_v2.parquet

models/
└── intent_classifier.joblib
```

---

# 🌐 Deployment

The application is deployed using Streamlit Community Cloud.

### Live Application

https://arabic-faq-rag.streamlit.app/

### Source Code

https://github.com/AbdelrhmanAkl/Arabic-FAQ-RAG

### Hugging Face

https://huggingface.co/AbdelrahmanAkl/Arabic-FAQ-RAG

---

# 🔗 Project Links

| Resource        | Link                                                 |
| --------------- | ---------------------------------------------------- |
| 🚀 Live Demo    | https://arabic-faq-rag.streamlit.app/                |
| 💻 GitHub       | https://github.com/AbdelrhmanAkl/Arabic-FAQ-RAG      |
| 🤗 Hugging Face | https://huggingface.co/AbdelrahmanAkl/Arabic-FAQ-RAG |

---

# ⚠️ Limitations

This project is primarily a **retrieval-based Arabic FAQ system**.

It does not currently use a generative LLM to generate new answers from retrieved context.

Instead, it retrieves existing FAQ answers from the indexed knowledge base.

This design helps keep responses grounded in the available FAQ data and reduces the risk of generating unsupported information.

---

# 🔮 Future Improvements

Potential future improvements include:

* Adding an LLM generation layer
* Full Retrieval-Augmented Generation pipeline
* Conversation memory
* Query rewriting
* Hybrid keyword + semantic retrieval
* Advanced cross-encoder reranking
* Better Arabic dialect handling
* Retrieval evaluation metrics such as Recall@K and MRR
* Confidence estimation
* Feedback-based retrieval improvement
* Multilingual expansion

---

# 👨‍💻 Author

**Abdelrahman Ahmed Akl**

AI / ML Engineer focused on:

* Artificial Intelligence
* Machine Learning
* NLP
* LLMs
* AI Agents
* Retrieval-Augmented Generation

---

## ⭐ Project

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.
