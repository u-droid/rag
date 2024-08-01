import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

def read_pdf(filename, file):
    docs = []
    reader = PdfReader(file)  # Use the file-like object directly
    for i in range(len(reader.pages)):
        st.progress(i / len(reader.pages))  # Update progress ratio
        page = reader.pages[i]
        docs.append(
            Document(
                page_content=page.extract_text(),  # Call the method to extract text
                metadata={
                    "source": filename,
                    "page": i + 1
                }
            )
        )
    return docs

def main():
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.environ.get('OPENAI_API_KEY')
    )
    
    # Initialize Pinecone
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = 'exam'
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )
    
    uploaded_file = st.file_uploader(label="Upload")
    if uploaded_file is not None:
        with st.spinner("Vectorizing"):
            # Use the file-like object directly
            docs = read_pdf(uploaded_file.name, uploaded_file)
            vectorstore.add_documents(docs)
        st.write('Uploaded and vectorized successfully.')

if __name__ == "__main__":
    main()
