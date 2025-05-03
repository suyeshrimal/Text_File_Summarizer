import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
import pandas as pd
from io import StringIO
from langchain.prompts import PromptTemplate

def llm_loader(groq_api_key):
    llm = ChatGroq(temperature=0,model="llama3-70b-8192",api_key=groq_api_key)
    return llm

st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")

st.markdown("## Enter Your GROQ API Key")

def get_groq_api_key():
    input_text=st.text_input(label="Groq ai API key",placeholder="Ex: gsk_euXLPpIQlMXi......",key="groq_api_key_input", type="password")
    return input_text

groq_api_key = get_groq_api_key()

st.markdown("## Upload the text file you want to summarize")
uploaded_file = st.file_uploader("Choose a file", type="txt")


st.markdown("# Here is your Summary:")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
    string_data = stringio.read()

    file_input = string_data

    if len(file_input.split(" ")) > 20000:
        st.write("Please enter a shorter file. The maximum length is 20000 words.")
        st.stop()

    if file_input:
        if not groq_api_key:
            st.warning('Please insert GROQ API Key.', 
            icon="⚠️")
            st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], 
        chunk_size=5000, 
        chunk_overlap=350
        )

    splitted_documents = text_splitter.create_documents([file_input])

    llm = llm_loader(groq_api_key=groq_api_key)

    summarize_chain = load_summarize_chain(
        llm=llm, 
        chain_type="map_reduce"
        )

    summary_output = summarize_chain.run(splitted_documents)

    st.write(summary_output)