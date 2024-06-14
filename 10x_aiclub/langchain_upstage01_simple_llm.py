"""
- https://developers.upstage.ai/docs/apis/chat
- https://www.youtube.com/watch?v=FouUOftcn70
"""
import os
from dotenv import load_dotenv

print(load_dotenv('./envls'))

from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{messages}")
])

model = ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"))
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi, how are you")
]

# response = model.invoke(messages)
# print(response.content)

chain = prompt | model | StrOutputParser()

response = chain.invoke("Hi, how are you")

print(response)
