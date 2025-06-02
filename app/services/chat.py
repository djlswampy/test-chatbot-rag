from dotenv import load_dotenv
# OpenAI GPT 기반 LLM 객체를 생성하는 LangChain 래퍼
from langchain_openai import ChatOpenAI
# 시스템/사용자 역할을 나눈 대화형 프롬프트 템플릿을 정의할 수 있는 클래스
from langchain_core.prompts import ChatPromptTemplate
# LLM 출력 결과를 문자열로 파싱해주는 기본 OutputParser
from langchain_core.output_parsers import StrOutputParser
# LangChain의 체이닝 파이프라인을 함수형으로 구성하는 기본 추상 객체
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# 이미 생성된 FAISS 인덱스를 로드
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    "./faiss_index_박철수",      # 인덱스 폴더 경로
    embedding_model,
    allow_dangerous_deserialization=True  # ← 신뢰 표시
)
retriever = vectorstore.as_retriever()

# 모델 생성
llm = ChatOpenAI(model="gpt-4o-mini")

# 프롬프트 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "당신은 질문-답변(Question-Answer) Task 를 수행하는 AI 어시스턴트입니다. "
     "검색된 문맥(context)를 사용하여 질문(question)에 답하세요. "
     "만약, 문맥(context)으로부터 답을 찾을 수 없다면 '모른다'고 말하세요. "
     "한국어로 대답하세요."
    ),
    ("user", "# Question:\n{question}\n\n# Context:\n{context}")
])

# 결과 파싱
parser = StrOutputParser()


# RAG 체인 정의 (LCEL)
chain: Runnable = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask(question: str) -> str:
    """
    RAG 기반 질문-답변 실행
    """
    result = chain.invoke(question)
    return result.content if hasattr(result, "content") else str(result)
