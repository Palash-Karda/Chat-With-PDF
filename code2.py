import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.readers.file import(PDFReader)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Process the uploaded file using SimpleDirectoryReader
    pdf_reader = PDFReader()
    documents = pdf_reader.load_data("uploaded_file.pdf")


    # Display success message
    st.success("PDF uploaded and processed successfully!")

    Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
    )

    index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)

    HF_TOKEN = "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"
    remotely_run = HuggingFaceInferenceAPI(
    model_name="HuggingFaceH4/zephyr-7b-alpha", token=HF_TOKEN
    )
    Settings.llm = remotely_run

    st.title("GenAI Query Application")

    query = st.text_input("Enter your query:")
    if query:
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        st.write("Response:")
        st.write(response.response)
