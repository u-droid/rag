import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

def vectorizer(filename, file, vectorstore: PineconeVectorStore):
    reader = PdfReader(file)  # Use the file-like object directly
    progress_bar = st.progress(0)
    docs = []
    for i in range(len(reader.pages)):
        progress_bar.progress(i / len(reader.pages), text=f"Vectorized {i} pages")  # Update progress ratio
        page = reader.pages[i]
        page_content = ''
        page_content += page.extract_text()
        docs.append(
            Document(
                page_content=page_content,  # Call the method to extract text
                metadata={
                    "source": filename,
                    "page": i + 1
                }
            )
        )
        if len(docs)==100:
            vectorstore.add_documents(docs)
            docs = []

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
            vectorizer(uploaded_file.name, uploaded_file, vectorstore)
        st.write('Uploaded and vectorized successfully.')

if __name__ == "__main__":
    main()
    
