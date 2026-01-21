# app/api/daily_agent/__init__.py
from .graph import app_graph
# app/api/daily_agent/__init__.py

# daily.py에 정의된 router를 api_prac_router라는 이름으로 노출시킵니다.
from .daily import router as api_prac_router

# 다른 곳에서 이 패키지를 불러올 때 사용할 수 있도록 설정
__all__ = ["api_prac_router"]