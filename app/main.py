"""
[FastAPI 서버 실행 방법]

uvicorn app.main:app --reload --host 0.0.0.0 --port 8500

- 이 명령은 main.py 파일 내의 'app = FastAPI()' 객체를 실행하여 FastAPI 웹 서버를 구동합니다.
- 각 옵션의 의미는 다음과 같습니다:

  --reload
    코드가 수정될 때마다 자동으로 서버를 재시작합니다.
    (개발 환경에서 매우 유용함)

  --host 0.0.0.0
    외부 IP에서도 접속할 수 있도록 모든 네트워크 인터페이스에서 요청을 받습니다.
    (기본값은 127.0.0.1로, 내부에서만 접근 가능)

  --port 8500
    서버가 열릴 포트 번호를 지정합니다. (http://서버IP:8500)

예:
  내부에서 실행 시: http://127.0.0.1:8500
  외부에서 접속 시: http://<서버공인IP>:8500/chat
"""
from fastapi import FastAPI
from app.routes import router

app = FastAPI()

# 라우터 등록
app.include_router(router)

