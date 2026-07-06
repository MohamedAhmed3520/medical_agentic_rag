# 🩺 Medical Agentic RAG

An end-to-end **Agentic Retrieval-Augmented Generation (RAG)** system for answering medical questions from uploaded PDF documents using **LangGraph**, **FAISS**, **Sentence Transformers**, and **OpenRouter LLMs**.

The system retrieves relevant medical evidence, builds contextual information, generates accurate answers, and provides complete citations including the **PDF filename** and **page number** for every response.

---

# 🚀 Features

- 📄 Upload one or multiple medical PDF documents
- 🔍 Semantic document retrieval using FAISS
- 🧠 Agentic workflow built with LangGraph
- 🤖 LLM-powered medical question answering
- 📚 Source attribution with:
  - PDF filename
  - Page number
  - Source references
- 📖 Context-aware generation
- ⚡ Streamlit chat interface
- 🏗️ Modular architecture for easy extension
- 🔄 Ready for Hybrid Search (FAISS + BM25)
- 🎯 Ready for Reranking integration
- 🪞 Reflection and validation nodes
- 📑 Citation generation

---

# 🏛️ Architecture

```
                User Question
                      │
                      ▼
                 Router Agent
                      │
                      ▼
                 Document Retrieval
                      │
                      ▼
              Hybrid Search (Ready)
                      │
                      ▼
                  Reranker (Ready)
                      │
                      ▼
              Medical Research Agent
                      │
                      ▼
                Reflection Agent
                      │
                      ▼
                Validation Agent
                      │
                      ▼
                 Citation Builder
                      │
                      ▼
               OpenRouter Generator
                      │
                      ▼
                 Final Response
```

---

# 🧠 Tech Stack

- Python
- LangGraph
- LangChain
- FAISS
- Sentence Transformers
- OpenRouter API
- HuggingFace Embeddings
- Streamlit
- PyPDF
- Requests

---

# 📂 Project Structure

```
medical_agentic_rag/
│
├── app.py
├── config.py
├── requirements.txt
│
├── graph/
│   ├── state.py
│   └── workflow.py
│
├── rag/
│   ├── pipeline.py
│   ├── retriever.py
│   ├── generator.py
│   ├── embeddings.py
│   └── vectorstore.py
│
├── ui/
│   └── app.py
│
├── utils/
│   └── logging.py
│
├── data/
├── cache/
├── logs/
└── prompts/
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/medical_agentic_rag.git

cd medical_agentic_rag
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file.

```env
OPENROUTER_API_KEY=your_api_key

OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

OPENROUTER_MODEL=openai/gpt-4o-mini

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

RERANKER_MODEL=BAAI/bge-reranker-large

TEMPERATURE=0.1

MAX_TOKENS=700

CHUNK_SIZE=800

CHUNK_OVERLAP=120

TOP_K=8

HYBRID_ALPHA=0.6

ENABLE_ASYNC=true
```

---

# ▶️ Run

```bash
streamlit run app.py
```

---

# 💬 Example

### Question

```
What are the symptoms of acute appendicitis?
```

### Response

```
Acute appendicitis commonly presents with
right lower quadrant abdominal pain,
nausea, vomiting, fever, and loss of appetite.

Source:
Bailey & Love's Short Practice of Surgery.pdf

Page:
421
```

---

# 📌 Workflow

1. User asks a medical question
2. Router determines the query type
3. Relevant medical documents are retrieved
4. Medical context is constructed
5. Evidence is validated
6. Citations are prepared
7. OpenRouter generates the final answer
8. The answer is returned with:
   - Medical explanation
   - PDF filename
   - Page number
   - Source references

---

# 📈 Current Capabilities

- ✅ Medical PDF Retrieval
- ✅ Semantic Search
- ✅ LangGraph Agentic Workflow
- ✅ OpenRouter Integration
- ✅ Citation Generation
- ✅ PDF Filename Extraction
- ✅ Page Number Detection
- ✅ Multi-document Support
- ✅ Streamlit Chat Interface

---

# 🔮 Future Improvements

- Hybrid Search (FAISS + BM25)
- Cross-Encoder Reranking
- RAGAS Evaluation
- Query Rewriting Agent
- Multi-Agent Collaboration
- Medical Knowledge Graph Integration
- Conversation Memory
- Streaming Responses
- OCR Support for Scanned PDFs
- Docker Deployment
- REST API
- Authentication
- Feedback Collection

---

# 🎯 Use Cases

- Medical Education
- Clinical Decision Support
- Medical Literature Search
- Research Assistance
- Medical Document Question Answering
- Hospital Knowledge Bases

---

# 📜 License

This project is intended for educational and research purposes.

The generated responses should **not** replace professional medical advice or clinical judgment.

---

# 👨‍💻 Author

**Mohamed Ahmed Mohamed**

- LinkedIn: https://www.linkedin.com/in/mohamed-ahmed-372730373/
- GitHub: https://github.com/MohamedAhmed3520
