#!/bin/env python3
import os
import uuid
import streamlit as st
from streamlit_chat import message
from langchain_community.vectorstores import DeepLake
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

st.set_page_config(page_title="Asthmatic Information A.I", page_icon=":lungs:")
#make a sidebar with information about the project
st.sidebar.title("Asthmatic Information A.I")
st.sidebar.write("This project is to provide information about asthma, its treatments and how to manage it.")
st.sidebar.write("The information is collected from Indonesia (from the Ministry of Health), Singapore  and more information will be added.")
st.sidebar.write("Let's connect: [LinkedIn](https://www.linkedin.com/in/rianrachmanto/)")


# Environment Variables and Initial Setup
api_key = st.secrets["OPENAI_API_KEY"]
ACTIVELOOP_TOKEN = st.secrets["ACTIVELOOP_TOKEN"]
embeddings_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4o-mini", max_tokens=1000)

# Initialize DeepLake DB and related session state setup
if 'retriever' not in st.session_state or 'assistant' not in st.session_state:
    db = DeepLake(dataset_path="hub://rian/medicaldoc", embedding=embeddings_model, read_only=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    chain = load_qa_chain(llm, chain_type="stuff")
    st.session_state['retriever'] = retriever
    st.session_state['assistant'] = chain
    st.session_state['messages'] = []

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state['messages']):
        message(msg, is_user=is_user, key=str(uuid.uuid4()))

def ask(question):
    context = st.session_state['retriever'].get_relevant_documents(question)
    prompt_template = """Answer the following question based only on the provided context.
    Your answers must be based on the document and please provide detailed answers.
    If you don't know the answer, just say you don't know. Don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer: Let's approach this step by step:"""
    prompt = prompt_template.format(context=context, question=question)
    return st.session_state['assistant'].run(input_documents=context, question=prompt)

def process_input():
    user_input = st.session_state['user_input']
    if user_input and user_input.strip():
        user_msg_id = str(uuid.uuid4())
        st.session_state['messages'].append((user_input, True))
        answer = ask(user_input)
        st.session_state['messages'].append((answer, False))

def setup_page():
    st.header("Chat with Medical Documents")
    st.text_input("Message", key="user_input", on_change=process_input)
    display_messages()

if __name__ == "__main__":
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    setup_page()
