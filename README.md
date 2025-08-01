# Intelligent Chatbot System

This project is the result of a summer internship at Quantabase Technologies, aimed at building a domain specific intelligent chatbot using state of the art Retrieval Augmented Generation (RAG) and Hybrid Search methodologies.

## 🚀 Overview

The Quantabase chatbot provides accurate, context aware answers in areas like:

- Product support
- Technical support
- Billing inquiries
- General FAQs

It is optimized for real-time performance and reliability using advanced NLP techniques and lightweight LLMs.

---

## 🔧 Tech Stack

- **Language Models**: `DialoGPT-medium`, `Flan-T5-base`, `Orca Mini 3B`
- **Retrieval Engine**: ChromaDB (with HNSW indexing)
- **Embeddings**: Sentence Transformers (`all-mpnet-base-v2`)
- **Vector Store**: ChromaDB
- **Text Splitter**: LangChain `RecursiveCharacterTextSplitter`
- **Memory**: `ConversationSummaryBufferMemory`
- **Search Strategy**: Hybrid (70% Semantic + 30% BM25 Keyword)
- **Interface**: CLI-based tester (`rag_chatbot.py`)

---

## 🧠 Core Architecture

### Week 1: MVP & SML Limitations

- Scoped chatbot capabilities
- Tested with SMLs like `DialoGPT` and `Flan-T5`
- Observed poor contextual retention and generic responses

### Week 2: RAG Implementation

- Switched to RAG with:
  - Chunking (~400 chars, 75 overlap)
  - Semantic embedding via HuggingFace
  - Storage in ChromaDB
  - Prompt generation for LLM response

### Week 3: Hybrid Search Development

- Combined semantic and BM25-based keyword matching
- Weighted retrieval (70:30) with overlap boosting
- Score normalization and merging logic

### Week 4: QA Optimization & Debug Tools

- Summarized memory buffers
- Real-time query scoring and classification
- Debug tools for latency, CPU, and RAM monitoring
- Retail synonym mapping and domain-term extraction

---

## 📂 Dataset Structure

- `product_info.txt`
- `technical_support.txt`
- `billing_support.txt`
- `general_faq.txt`

Each document is processed and embedded into vector format for fast retrieval.

---

## 🔮 Future Enhancements

- Build a user-friendly UI/UX for frontend deployment
- Reduce latency further via optimized hardware and batching
- Expand domain coverage and language capabilities

---

## 🧑‍💻 Contributors

- **Harish Raj S**
- **Paranthagan S**

---
