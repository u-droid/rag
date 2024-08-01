import os
os.environ.clear()
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

os.environ['OPENAI_API_KEY'] = st.secrets.OPENAI_API_KEY
os.environ['PINECONE_API_KEY'] = st.secrets.PINECONE_API_KEY

def retreive_context(prompt):
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    index_name = 'exam'
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.environ.get('OPENAI_API_KEY')
    )
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )
    retriever = vectorstore.as_retriever()
    matched_docs = retriever.invoke(prompt)
    with st.sidebar:
        for doc in matched_docs:
            st.success(doc.page_content)
            st.write(doc.metadata)
    return matched_docs[0]

def get_completion(prompt):
    client = OpenAI(
        api_key=os.environ.get('OPENAI_API_KEY')
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def show_history():
    for conv in st.session_state['history']:
        with st.chat_message("user"):
            st.write(conv.get('user'))
        with st.chat_message("assistant"):
            st.write(conv.get('assistant'))

def main():
    #st.write(os.environ.items())
    prompt = st.chat_input("Say something")
    if prompt:
        context = retreive_context(prompt)
        final_prompt = f"Answer the below question by levraging the context.\nContext: {context.page_content}\nQuestion{prompt}"
        completion  = get_completion(prompt=final_prompt)
        completion = completion +"\nMetadata:\n"+ str(context.metadata)
        st.session_state['history'].append(
            {
                'user':prompt,
                'assistant':completion
            }
        )
        show_history()
        #st.write(context.metadata)

if __name__=="__main__":

    if 'history' not in st.session_state:
        st.session_state['history'] = []
    main()
        





