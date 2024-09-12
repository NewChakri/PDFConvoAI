import os
import streamlit as st
from PyPDF2 import PdfReader
from fixthaipdf import clean
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub



# Set Hugging Face API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Function to extract text from PDF files
def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    text = clean(text)
    return text

# Function to split text into chunks
def get_chunk_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store from text chunks
def get_vector_store(text_chunks, model_name="hkunlp/instructor-xl"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create conversation chain
def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Streamlit App
def main():
    st.title("ConvoDocsAI: Conversational AI for PDFs")
    st.write("Upload your PDF files and chat with AI to retrieve information from them.")

    # Two columns layout for file upload and conversation
    col1, col2 = st.columns([1, 2])

    with col1:
        # Upload PDF Files
        uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
    
        if uploaded_files:
            st.write("Processing PDF files...")

            # Extract text from PDFs
            pdf_text = get_pdf_text(uploaded_files)

            # Split text into chunks
            chunks = get_chunk_text(pdf_text)

            # Create vector store
            vectorstore = get_vector_store(chunks)

            # Create conversation chain
            conversation_chain = get_conversation_chain(vectorstore)

            st.session_state['conversation_chain'] = conversation_chain  # Save to session state for multiple messages

    with col2:
        # Initialize session state to hold the chat history
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        st.header("Chat with AI")
        
        # Multi-message conversation: Display chat history
        for message in st.session_state['chat_history']:
            st.write(f"You: {message['user']}")
            st.write(f"AI: {message['ai']}")
        
        # Input for user message
        user_input = st.text_input("Ask something:")
        
        if user_input and 'conversation_chain' in st.session_state:
            # Get the conversation chain from session state
            conversation_chain = st.session_state['conversation_chain']
            
            # Use conversation history for the conversation chain
            response = conversation_chain.run({"question": user_input, "chat_history": st.session_state['chat_history']})
            
            # Add user input and AI response to the chat history
            st.session_state['chat_history'].append({"user": user_input, "ai": response})
            
            # Display the latest AI response
            st.write(f"AI: {response}")

if __name__ == "__main__":
    main()
