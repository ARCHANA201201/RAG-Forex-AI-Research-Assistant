# 📊 RAG Forex AI Research Assistant

An AI-powered Forex market research assistant that uses **Retrieval-Augmented Generation (RAG)** to analyze forex articles and answer trading-related questions.

The application allows users to upload forex news or article URLs and ask questions about the market. The AI retrieves relevant information from indexed documents and generates intelligent responses.

---

## 🚀 Features

- 🤖 AI-powered Forex research assistant
- 🔎 Retrieval-Augmented Generation (RAG)
- 📰 Analyze Forex news and articles
- 💬 Interactive chatbot interface
- 📊 Streamlit web dashboard
- 📚 Document embeddings using HuggingFace
- 🗂 Vector database using ChromaDB

---

## 🧠 Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Newspaper3k
- Groq LLM API

---

## 📂 Project Structure
RAG_Project


├── main.py # Streamlit application

├── rag.py # RAG backend pipeline

├── requirements.txt # Project dependencies

├── .gitignore # Ignored files

├── .env # API keys (not uploaded)

└── resources/ # Vector database storage

---

## ⚙️ Installation

### 1️⃣ Clone the repository
git clone https://github.com/ARCHANA201201/rag-forex-ai-research-assistant.git

cd rag-forex-ai-research-assistant

### 2️⃣ Install dependencies
pip install -r requirements.txt

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

GROQ_API_KEY=your_api_key_here

---

## ▶️ Run the Application
streamlit run main.py

The application will open in your browser.

---

## 💬 Example Questions

You can ask questions like:

- Why is USD rising today?
- What affects EUR/USD?
- Why is Japanese Yen weak?
- What factors move GBP/USD?

---

## 📊 How It Works

1. User adds Forex article URLs
2. Articles are downloaded and processed
3. Text is split into chunks
4. Embeddings are created using HuggingFace
5. Data stored in Chroma vector database
6. RAG pipeline retrieves relevant context
7. AI generates answer using Groq LLM

---

## 🔒 Security

API keys are stored in `.env` and excluded from the repository using `.gitignore`.

---



