from fastapi import APIRouter
from .daily import router as daily_prac_router


# daily_prac 버전의 모든 라우터를 통합하는 메인 라우터
api_prac_router = APIRouter()

# 각 모듈에서 정의한 router를 등록
api_prac_router.include_router(daily_prac_router, prefix="/agent/daily", tags=["daily"])

