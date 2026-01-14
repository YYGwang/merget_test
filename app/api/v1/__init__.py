from fastapi import APIRouter
from .draft import router as draft_router
from .wiki import router as wiki_router
from .daily import router as daily_router

# v1 버전의 모든 라우터를 통합하는 메인 라우터
api_v1_router = APIRouter()

# 각 모듈에서 정의한 router를 등록
api_v1_router.include_router(draft_router, prefix="/draft", tags=["Draft"])
api_v1_router.include_router(wiki_router, prefix="/wiki", tags=["Wiki"])
api_v1_router.include_router(daily_router, prefix="/daily", tags=["Daily"])
