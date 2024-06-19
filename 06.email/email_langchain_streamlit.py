"""
- https://github.com/gilbutITbook/080413/blob/main/%EC%8B%A4%EC%8A%B5/5%EC%9E%A5/5_7_%EB%A9%94%EC%9D%BC_%EC%9E%91%EC%84%B1%EA%B8%B0_%EB%A7%8C%EB%93%A4%EA%B8%B0.py
- run: streamlit run email_langchain_streamlit.py --server.port 8001
"""

# import os
# os.environ["OPENAI_API_KEY"] = "sk-" #openai 키 입력

from dotenv import load_dotenv
load_dotenv('../envls')
import streamlit as st

st.set_page_config(page_title="이메일 작성 서비스예요~", page_icon=":robot:")
st.header("이메일 작성기")

def getEmail():
    input_text = st.text_area(label="메일 입력", label_visibility='collapsed',
                              placeholder="당신의 메일은...", key="input_text")
    return input_text

input_text = getEmail()

# 이메일 변환 작업을 위한 템플릿 정의
query_template = """
    메일을 작성해주세요.
    아래는 이메일입니다:
    이메일: {email}
"""

from langchain_core.prompts import ChatPromptTemplate
# PromptTemplate 인스턴스 생성
prompt = ChatPromptTemplate.from_template("""
    메일을 작성해주세요.
    아래는 이메일입니다:
    이메일: {email}
"""
)

from langchain_openai import ChatOpenAI
# 언어 모델을 호출합니다
def loadLanguageModel():
    llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo')
    return llm

# 예시 이메일을 표시
st.button("*예제를 보여주세요*", type='secondary', help="봇이 작성한 메일을 확인해보세요.")
st.markdown("### 봇이 작성한 메일은:")

if input_text:
    llm = loadLanguageModel()
    # PromptTemplate 및 언어 모델을 사용하여 이메일 형식을 지정
    prompt_with_email = prompt.format(email=input_text)
    formatted_email = llm.predict(prompt_with_email)
    # 서식이 지정된 이메일 표시
    st.write(formatted_email)
