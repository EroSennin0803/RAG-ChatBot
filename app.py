import streamlit as st
import os
from rag_graph import run_chat

st.set_page_config(page_title="RAG Chatbot - Agentic AI", layout="wide")

st.title("RAG-based AI Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message:
            with st.expander("Retrieved Context & Score"):
                st.write(f"**Confidence / Distance Score:** {message['metadata']['score']}")
                for i, chunk in enumerate(message['metadata']['chunks']):
                    st.markdown(f"**Chunk {i+1}** (Page: {chunk['metadata'].get('page', 'N/A')}):")
                    st.info(chunk['content'])

# React to user input
if prompt := st.chat_input("Ask a question..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call LangGraph RAG pipeline
    with st.spinner("Searching and generating answer..."):
        try:
            result = run_chat(prompt)
            final_answer = result["final_answer"]
            chunks = result["context_chunks"]
            score = result["confidence_score"]
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(final_answer)
                with st.expander("Retrieved Context & Score"):
                    st.write(f"**Confidence / Distance Score:** {score:.4f}")
                    for i, chunk in enumerate(chunks):
                        st.markdown(f"**Chunk {i+1}** (Page: {chunk['metadata'].get('page', 'N/A')}):")
                        st.info(chunk['content'])
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_answer,
                "metadata": {
                    "score": round(score, 4),
                    "chunks": chunks
                }
            })
        except Exception as e:
            st.error(f"Error processing query: {e}")
            st.info("Make sure you have ingested the PDF and set the GOOGLE_API_KEY in the .env file.")
