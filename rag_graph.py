import os
from typing import List, Dict, Any, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "./chroma_db"

# 1. Define State
class GraphState(TypedDict):
    question: str
    documents: List[Dict[str, Any]]
    final_answer: str
    confidence_score: float

# 2. Nodes
def retrieve(state: GraphState):
    print("---RETRIEVE---")
    question = state["question"]
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Retrieve top 4 chunks with similarity scores
    # In Chroma, lower distance score means more similar (L2 distance by default)
    # But similarity_search_with_relevance_scores can normalize it, but sometimes it depends on the distance metric.
    # Let's just use similarity_search_with_score
    docs_and_scores = vectorstore.similarity_search_with_score(question, k=4)
    
    documents = []
    scores = []
    for doc, score in docs_and_scores:
        documents.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        })
        scores.append(score)
    
    # Compute a simple mock confidence score based on the distances
    # Distance usually 0 to 1, or higher. We will just pass the best (lowest) distance score, or normalize.
    # We will just use the first document's score as the "confidence/distance score".
    best_score = float(scores[0]) if scores else 0.0

    return {"documents": documents, "question": question, "confidence_score": best_score}

def generate(state: GraphState):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    best_score = state["confidence_score"]
    
    context_text = "\n\n".join([f"Chunk: {doc['content']}" for doc in documents])
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
    
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer or the answer is not contained in the context, just say that you don't know. 
Do not make up any information outside of the provided context. Ground your answer strictly in the context.

Question: {question} 
Context: {context} 
Answer:""",
        input_variables=["question", "context"],
    )
    
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context_text})
    
    import ast
    content = response.content
    
    # If LangChain returned a stringified list of dicts, parse it back to a list
    if isinstance(content, str) and content.strip().startswith("[{") and content.strip().endswith("}]"):
        try:
            content = ast.literal_eval(content)
        except:
            pass

    if isinstance(content, list):
        final_text = "".join([block.get("text", "") if isinstance(block, dict) else str(block) for block in content])
    else:
        final_text = str(content)
        
    return {"final_answer": final_text}

# 3. Build Graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()

def run_chat(question: str):
    initial_state = {"question": question, "documents": [], "final_answer": "", "confidence_score": 0.0}
    result = app.invoke(initial_state)
    return {
        "final_answer": result["final_answer"],
        "context_chunks": result["documents"],
        "confidence_score": result["confidence_score"]
    }
