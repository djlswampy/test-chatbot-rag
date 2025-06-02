# Test Chatbot RAG

- RAG 기능 테스트용 챗봇 프로젝트
  

## 프로젝트 개요

PDF 문서를 활용한 질의응답 시스템
- PDF 문서를 청킹하여 FAISS 벡터 데이터베이스에 저장
- 사용자 질문에 대해 관련 문서를 검색하고 GPT 모델로 답변 생성
- FastAPI를 통한 REST API 제공


## 기술 스택
- **Backend**: FastAPI
- **LLM**: OpenAI GPT-4o-mini
- **Vector Store**: FAISS
- **Framework**: LangChain
- **Document Processing**: PyPDF


## 설치 및 실행

### 1. 가상환경 설정
```bash
python -m venv venv
source venv/bin/activate
```

### 2. 의존성 설치
- pip install -r requirements.txt


### 3. 환경변수 설정
- 프로젝트 루트 디렉토리에 .env 파일을 생성하고 OpenAI API 키 설정
- OPENAI_API_KEY=your_api_key_here

### 4. pdf 문서 전처리
```bash
python rag_preprocessing.py
```

### 5. 서버 실행
uvicorn app.main:app --reload --port 8000


## 테스트 예시 질문
"박철수는 누구인가요?"
"박철수의 나이는 몇 살인가요?"
"박철수가 다니는 회사는 어디인가요?"


