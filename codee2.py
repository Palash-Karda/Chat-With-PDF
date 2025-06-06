import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Load .env variables
load_dotenv()
Cohere_API_KEY = os.getenv("Cohere_API_KEY")

# Initialize SpaCy embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to read PDF files and extract text
def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# Function to split text into chunks
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create and save FAISS vector store
def vector_store(text_chunks):
    vector_db = FAISS.from_texts(text_chunks, embedding=embeddings)
    os.makedirs("faiss_db", exist_ok=True)
    vector_db.save_local("faiss_db")


# Function to create and run the conversational agent
def get_conversational_chain(tool, question):
    llm = ChatCohere(cohere_api_key=Cohere_API_KEY, model="command-r-plus",
        temperature=0,
        api_key=Cohere_API_KEY,
        verbose=True
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Answer the question as detailed as possible from the provided context. "
            "If the answer is not in the provided context, just say 'Answer is not available in the context.'"
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    tools = [tool] if not isinstance(tool, list) else tool
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": question})
    st.write("### 🤖 Reply:", response['output'])


# Function to process user question
def user_input(user_question):
    if not os.path.exists("faiss_db/index.faiss"):
        st.error("Please upload and process a PDF first.")
        return

    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    retrieval_tool = create_retriever_tool(
        retriever,
        name="pdf_extractor",
        description="Useful for answering questions from the uploaded PDF."
    )
    get_conversational_chain(retrieval_tool, user_question)


# Streamlit UI
def main():
    st.set_page_config(page_title="Chat with PDF", layout="centered")
    st.header("🧠 Chat with your PDF (Cohere + FAISS + LangChain)")

    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📄 Upload PDFs")
        pdf_doc = st.file_uploader("Upload PDF(s)", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if pdf_doc:
                with st.spinner("🔄 Processing..."):
                    raw_text = pdf_read(pdf_doc)

                    if not raw_text.strip():
                        st.error("❌ Could not extract any text from the uploaded PDFs.")
                        return

                    chunks = get_chunks(raw_text)
                    vector_store(chunks)
                    st.success("✅ Text processed and indexed successfully!")
            else:
                st.error("❌ Please upload at least one PDF.")


if __name__ == "__main__":
    main()
