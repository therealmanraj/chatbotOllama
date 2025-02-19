from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPEN_AI_KEY"]= os.getenv("OPEN_AI_KEY")

os.environ["LANGCHAIN_API_KEY"]= os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]= os.getenv("LANGSMITH_TRACING")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please provide a helpful response to the user query."),
        ("user","Question:{question}")
    ]
)

st.title("Langchain Demo with Open AI API")
input_text = st.text_input("Search the topic you want")

llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()

chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))