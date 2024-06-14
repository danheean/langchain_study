"""
- https://python.langchain.com/v0.2/docs/tutorials/chatbot/
- https://www.youtube.com/watch?v=0PRmLRSi374
"""
import dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


dotenv.load_dotenv('envls')

model = ChatOpenAI(model="gpt-4o")

store = {}

def get_session_history(sid: str) -> BaseChatMessageHistory:
    if sid not in store:
        store[sid] = ChatMessageHistory()
    return store[sid]

prompt = ChatPromptTemplate.from_messages([
    ("system", "넌 나의 친한 친구야, 너의 이름은 {name}이야, Answer all questions to the best of your ability in {language}"),
    MessagesPlaceholder(variable_name="history")
])

# resp3 = prompt.invoke({
#     "history": [("user", "Hi my name is togepi"), ("user", "What's my name?")]
# })

# print(resp3)

runnable_with_history = RunnableWithMessageHistory(prompt | model, get_session_history, input_messages_key="history")

chain = runnable_with_history | StrOutputParser()
config = {"configurable": {"session_id": "test1"}}

resp1 = chain.invoke({"history":[("user", "Hi my name is togepi")], "name": "eevee", "language": "Korean"}, config)
print(resp1)

resp2 = chain.invoke({"history": [
    #("user", "Hi my name is togepi"),
    ("user", "What's my name?")
], "name": "eevee", "language": "Korean"}, config)
print(resp2)

resp3 = chain.invoke({"history": [
    #("user", "Hi my name is togepi"),
    ("user", "What's your name?")
], "name": "eevee", "language": "Korean"}, config)
print(resp3)
