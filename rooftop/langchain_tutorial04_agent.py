"""
- https://python.langchain.com/v0.2/docs/tutorials/agents/
- https://www.youtube.com/watch?v=dGk83z2RkO4
"""

import os
import dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent

print(dotenv.load_dotenv('envls'))
# print(os.getenv('TAVILY_API_KEY'))

model = ChatOpenAI(model="gpt-4o")
search = TavilySearchResults(max_results=2)
memory = SqliteSaver.from_conn_string('agent1.sqlite')

agent1 = create_react_agent(model, [search], checkpointer=memory)

# print(agent1)
# results = search.invoke("비트 코인 가격")
# print(results)

# with open("./agent1.png", "wb") as f:
#     f.write(agent1.get_graph().draw_mermaid_png())

# print(agent1)

config = {"configurable": {"thread_id": "thread01"}}

for chunk in agent1.stream({
    "messages": [("user", "오늘의 비트코인의 가격을 알려줘... 1비트코인은 몇불이지?")]
}, config):
    print(chunk)



