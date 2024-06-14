from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

docs = [
    Document(
        page_content="금융보험은 장기적인 자산 관리와 위험 대비를 목적으로 고안된 금융 상품입니다."
    ),
    Document(
        page_content="금융저축보험은 규칙적인 저축을 통해 목돈을 마련할 수 있으며, 생명보험 기능도 겸비하고 있습니다."
    ),
    Document(
        page_content="저축금융보험은 저축과 금융을 통해 목돈 마련에 도움을 주는 보험입니다. 또한, 사망 보장 기능도 제공합니다."
    ),
    Document(
        page_content="금융저축산물보험은 장기적인 저축 목적과 더불어, 축산물 제공 기능을 갖추고 있는 특별 금융 상품입니다."
    ),
    Document(
        page_content="금융단폭격보험은 저축은 커녕 위험 대비에 초점을 맞춘 상품입니다. 높은 위험을 감수하고자 하는 고객에게 적합합니다."
    ),
    Document(
        page_content="금보험은 저축성과를 극대화합니다. 특히 노후 대비 저축에 유리하게 구성되어 있습니다."
    ),
    Document(
        page_content="금융보씨 험한말 좀 하지마시고, 저축이나 좀 하시던가요. 뭐가 그리 급하신지 모르겠네요."
    ),
]
