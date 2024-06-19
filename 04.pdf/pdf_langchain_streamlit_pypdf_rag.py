"""
- https://github.com/gilbutITbook/080413/blob/main/%EC%8B%A4%EC%8A%B5/5%EC%9E%A5/5_3_PDF_%EC%9A%94%EC%95%BD_%EC%9B%B9%EC%82%AC%EC%9D%B4%ED%8A%B8_%EB%A7%8C%EB%93%A4%EA%B8%B0.py
- run: streamlit run pdf_langchain_streamlit_pypdf_rag.py --server.port 8001
"""
import os
from dotenv import load_dotenv
load_dotenv('../envls')
from PyPDF2 import PdfReader
import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
#from langchain import FAISS
from langchain_community.vectorstores import FAISS
#from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
#from langchain.callbacks import get_openai_callback
from langchain_community.callbacks.manager import get_openai_callback

def process_text(text):
#CharacterTextSplitter를 사용하여 텍스트를 청크로 분할
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    #임베딩 처리(벡터 변환), 임베딩은 HuggingFaceEmbeddings 모델을 사용합니다.
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    documents = FAISS.from_texts(chunks, embeddings)
    return documents

parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template("""너는 훌륭한 비서이다. 아래 [질문]에 대해서 [내용]안에서 답변해줘
[내용] ~~ 시작 ~~

{context}

[내용] ~~ 끝 ~~

[질문]
{question}
""")


def main():  #streamlit을 이용한 웹사이트 생성
    st.title("📄PDF 요약하기")
    st.divider()

    # try:
    #     os.environ["OPENAI_API_KEY"] = "sk-" #openai api 키 입력
    # except ValueError as e:
    #     st.error(str(e))
    #     return

    pdf = st.file_uploader('PDF파일을 업로드해주세요', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""   # 텍스트 변수에 PDF 내용을 저장
        for page in pdf_reader.pages:
            text += page.extract_text()

        documents = process_text(text)
        retriever =  documents.as_retriever(search_kwargs={"k": 3})

        query = "업로드된 PDF 파일의 내용을 약 3~5문장으로 요약해주세요."  # LLM에 PDF파일 요약 요청

        if query:
            #docs = documents.similarity_search(query)

            model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

            #llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.1)
            #chain = load_qa_chain(llm, chain_type='stuff')

            rag_chain = (
                {'context': retriever, 'question': RunnablePassthrough()}
                | prompt
                | model
                | parser
            )

            with get_openai_callback() as cost:
                # response = chain.run(input_documents=docs, question=query)
                query = "요약해 주세요"
                #response = rag_chain.invoke(question=query)
                response = rag_chain.invoke(query)
                print(cost)

            st.subheader('--요약 결과--:')
            st.write(response)

if __name__ == '__main__':
    main()
