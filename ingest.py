import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# We need GOOGLE_API_KEY set in environment or .env
if "GOOGLE_API_KEY" not in os.environ:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")

PDF_PATH = "Ebook-Agentic-AI.pdf"
CHROMA_PATH = "./chroma_db"

def ingest_pdf():
    print(f"Loading PDF from {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print(f"Loaded {len(docs)} pages. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} text chunks.")

    print("Generating embeddings and storing in Chroma in batches to avoid rate limits...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    # Process in batches of 50 to stay under the 100 requests per minute limit
    import time
    vectorstore = None
    batch_size = 10
    
    print("Waiting 30 seconds before starting to let previous rate limits clear...")
    time.sleep(30)
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(len(chunks)-1)//batch_size + 1}...")
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_PATH
            )
        else:
            vectorstore.add_documents(batch)
            
        if i + batch_size < len(chunks):
            print("Sleeping 20 seconds to respect rate limits...")
            time.sleep(20)
    
    print(f"Successfully ingested and saved to {CHROMA_PATH}")

if __name__ == "__main__":
    ingest_pdf()
