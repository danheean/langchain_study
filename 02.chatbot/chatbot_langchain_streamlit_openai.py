"""
- https://github.com/gilbutITbook/080413/blob/main/%EC%8B%A4%EC%8A%B5/5%EC%9E%A5/5_1_%EA%B0%84%EB%8B%A8%ED%95%9C_%EC%B1%97%EB%B4%87_%EB%A7%8C%EB%93%A4%EA%B8%B0.py
- run: streamlit run chatbot_langchain_streamlit_openai.py --server.port 8001
"""
import os
from dotenv import load_dotenv
import streamlit as st
#from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv('../envls')

#print(os.getenv('OPENAI_API_KEY'))

st.set_page_config(page_title='🦜🔗뭐든지 질문하세요~')
st.title('🦜🔗뭐든지 질문하세요~')

# def generate_response(input_text):
#     llm = OpenAI(model='gpt-3.5-turbo', temperature=0)
#     st.info(llm(input_text))


prompt = ChatPromptTemplate.from_messages([
    ("user", "{messages}")
])

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
parser = StrOutputParser()
chain = prompt | model | parser

def generate_response(input_text):
     st.info(chain.invoke(input_text))

with st.form('Question'):
    text = st.text_area('질문 입력:', 'What types of text models does OpenAI provide?')
    submitted = st.form_submit_button('보내기')
    generate_response(text)

