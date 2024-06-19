"""

"""

from dotenv import load_dotenv, find_dotenv
load_dotenv('../envls')

#from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
#from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 문서 로드
documents = TextLoader("./data/AI.txt").load()

# 문서를 청크로 분할
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

# docs 변수에 분할 문서를 저장
docs = split_docs(documents)

#print(docs)

#embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Chromdb에 벡터 저장
vector_store = Chroma.from_documents(docs, embeddings)

model_name = "gpt-3.5-turbo"  #gpt-3.5-turbo 모델 사용
model = ChatOpenAI(model_name=model_name, temperature=0)

# Q&A 체인을 사용하여 쿼리에 대한 답변 얻기
#chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
prompt = ChatPromptTemplate.from_template("""너는 훌륭한 비서이다. 아래 [질문]에 대해서 [내용]을 참고하여 답변해줘
[내용] ~~ 시작 ~~

{context}

[내용] ~~ 끝 ~~

[질문]
{question}
""")

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
parser = StrOutputParser()

qa_chain = ({'context': retriever, 'question': RunnablePassthrough()}
            | prompt
            | model
            | parser
)

# # 쿼리를 작성하고 유사성 검색을 수행하여 답변을 생성,따라서 txt에 있는 내용을 질의해야 합니다
# query = "AI란?"
# matching_docs = db.similarity_search(query)
# answer =  chain.run(input_documents=matching_docs, question=query)
# answer

#print(qa_chain.invoke("AI란?"))
print(qa_chain.invoke("Alan Turing에 관해 알려줘"))
