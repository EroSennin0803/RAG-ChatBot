# RAG Chatbot - Agentic AI

A Retrieval-Augmented Generation (RAG) based AI Chatbot built in Python. This chatbot strictly answers questions based on a provided knowledge base (the Agentic AI eBook) using Google's Gemini models and LangGraph.

## Features
- **Strict Grounding**: The LLM is configured with `temperature=0` and a strict system prompt to ensure it only answers using the retrieved context from the PDF, preventing hallucinations.
- **Advanced RAG Pipeline**: Built using **LangGraph** to construct a clear Retrieve → Generate node architecture.
- **Local Vector Database**: Uses **ChromaDB** to store and query document embeddings locally without needing external cloud databases.
- **Smart Chunking**: Processes the PDF in safe batches using the `gemini-embedding-2` model to respect API rate limits.
- **Interactive UI**: Built with **Streamlit** to provide a clean chat interface.
- **Transparency**: Every answer includes an expandable section showing the exact context chunks retrieved from the PDF and the vector distance score.

## Technology Stack
- **Language**: Python 3
- **LLM & Embeddings**: Google Gemini (`gemini-3.8-flash` for generation, `gemini-embedding-2` for embeddings)
- **Frameworks**: LangChain, LangGraph
- **Vector Database**: Chroma
- **Frontend**: Streamlit

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/EroSennin0803/RAG-ChatBot.git
   cd RAG-ChatBot
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   - Rename the `.env` template or create a new `.env` file in the root directory.
   - Add your Google Gemini API Key:
     ```env
     GOOGLE_API_KEY=your_google_api_key_here
     ```

4. **Ingest the Knowledge Base:**
   Run the ingestion script to chunk the PDF and populate the local Chroma vector database:
   ```bash
   python ingest.py
   ```
   *(Note: This script has built-in batching and delays to safely handle free-tier API rate limits).*

5. **Start the Chatbot UI:**
   ```bash
   streamlit run app.py
   ```
   The application will automatically open in your web browser.

## Architecture Flow
1. **Ingestion**: `PyPDFLoader` extracts text -> `RecursiveCharacterTextSplitter` chunks it -> Gemini embeds it -> Stored in Chroma DB.
2. **Retrieval**: User asks a question -> Converted to vector -> Chroma DB returns top 4 closest chunks using L2/Cosine similarity.
3. **Generation**: LangGraph routes the context chunks to the Gemini LLM with strict instructions to generate the final answer.

## Sample Queries to Try
To test the chatbot's grounding and retrieval capabilities, try asking these questions:
- *"Who are the members of the authoring team?"*
- *"What are the practical applications of Agentic AI?"*
- *"How does Agentic AI differ from traditional AI models?"*
- *"Can you explain the workflow automation examples mentioned in the book?"*
- *"What is the recipe for chocolate chip cookies?"* (To test the strict guardrails—the bot should refuse to answer since it's not in the PDF!)
