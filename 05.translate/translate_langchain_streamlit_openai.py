"""
- https://github.com/gilbutITbook/080413/blob/main/%EC%8B%A4%EC%8A%B5/5%EC%9E%A5/5_6_%EB%B2%88%EC%97%AD_%EC%84%9C%EB%B9%84%EC%8A%A4_%EB%A7%8C%EB%93%A4%EA%B8%B0.py
- run: streamlit run langchain_streamlit_openai.py --server.port 8001
"""
from dotenv import load_dotenv
load_dotenv('../envls')
import streamlit as st
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
#from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# import os
# os.environ["OPENAI_API_KEY"] = "sk-" #openai 키 입력

# 웹페이지에 보여질 내용
langs = ["Korean", "Japanese", "chinese", "English"]  #번역을 할 언어를 나열
left_co, cent_co, last_co = st.columns(3)

#웹페이지 왼쪽에 언어를 선택할 수 있는 라디오 버튼
with st.sidebar:
    language = st.radio('번역을 원하는 언어를 선택해주세요.:', langs)

st.markdown('### 언어 번역 서비스예요~')
prompt = st.text_input('번역을 원하는 텍스트를 입력하세요')  #사용자의 텍스트 입력

trans_template = PromptTemplate(
    input_variables=['trans'],
    template='Your task is to translate this text to ' + language + 'TEXT: {trans}'
)  #해당 서비스가 번역에 대한 것임을 지시

#momory는 텍스트 저장 용도
memory = ConversationBufferMemory(input_key='trans', memory_key='chat_history')

llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo')
trans_chain = LLMChain(llm=llm, prompt=trans_template, verbose=True, output_key='translate', memory=memory)

# 프롬프트(trans_template)가 있으면 이를 처리하고 화면에 응답을 작성
if st.button("번역"):
    if prompt:
        response = trans_chain({'trans': prompt})
        st.info(response['translate'])
