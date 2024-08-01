import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

def read_pdf(file):
    docs = []
    reader = PdfReader(file)
    for i in range(len(reader.pages)):
        st.progress(i)
        docs.append(
            Document(
                page_content=reader.pages[i].extract_text,
                metadata={
                    "source":file,
                    "page":i+1
                }
            )
        )
    return docs

def main():
    st.write(os.environ.items())
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.environ.get('OPENAI_API_KEY')
    )
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    print(pc.list_indexes())
    index_name = 'airline'
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )
    uploaded_file = st.file_uploader(label="Upload")
    if uploaded_file is not None:
        with st.spinner("Vectorizing"):
            docs = read_pdf(uploaded_file.name)
            vectorstore.add_documents(docs)
        st.write('uploaded')

if __name__=="__main__":
    main()