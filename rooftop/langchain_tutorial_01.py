"""
- https://python.langchain.com/v0.2/docs/tutorials/llm_chain/
"""
import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes

# 환경 변수 인식
dotenv.load_dotenv('envls')

prompt = ChatPromptTemplate.from_messages([
    ("system", " 다음에 주어진 단어를 보고 연상되는 화가와 화가의 생애를 애개해줘."),
    ("user", "단어 : {word}")
])

model = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()

chain = prompt | model | parser

#print(chain.invoke({"word": "사람"}))

app = FastAPI(
    title="Tutl Test",
    version="0.989",
    description="my tutl test..."
)

add_routes(
    app,
    chain,
    path="/paint"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

