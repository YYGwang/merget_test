from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.daily_agent import api_prac_router
from app.core.config import settings
from app.core.security import verify_cognito_token, mocked_verify_cognito_token

'''
## Local test 시 실행 방법 ##
    app 폴더와 동일한 위치에 .env.local 파일을 생성하고, ENV=local 작성 후 저장
    app 폴더와 동일한 위치에서 아래 커맨드 실행
    - uvicorn app.main:app
    
    터미널에 'local'이라는 텍스트가 나오면 aws token 검증 단계를 빼고 테스트 가능
    - authorization에 넣은 값을 uid로 사용해 테스트 진행됨
    
    만약 로컬 프론트엔드에서 직접 로그인해 토큰 인증을 거친 테스트를 진행하고 싶을 경우
    - .env.local 파일에 ENV=server 작성 후 저장. uvicorn 재실행 
'''

app = FastAPI(title="Noton API Server")
# local에서는 aws 토큰 검증을 더미로 진행하도록 함
if settings.ENV == "local":
    app.dependency_overrides[verify_cognito_token] = mocked_verify_cognito_token    # type: ignore
print(settings.ENV)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 통합된 v1 라우터 등록
app.include_router(api_prac_router)

@app.get("/")
async def root():
    return {"status": "ok"}