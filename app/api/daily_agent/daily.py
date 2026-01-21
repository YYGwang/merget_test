import time
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.core.security import verify_cognito_token
from app.core.database import get_table
from .graph import app_graph

router = APIRouter()
DAILY_TABLE = get_table('daily_table')


# 명세서: Request Parameters - body { content: str }
class DailyReportRequest(BaseModel):
    content: str


# 명세서: POST /dailly (또는 /daily)
# 경로 파라미터 {history_date}를 제거하여 명세와 일치시킵니다.
@router.post("/generate-report")
async def create_daily_report(
        request: DailyReportRequest,
        uid: str = Depends(verify_cognito_token)
):
    try:
        if not request.content:
            raise HTTPException(status_code=400, detail="content가 없습니다.")

        # 에이전트 실행
        config = {"configurable": {"thread_id": uid}}
        final_state = app_graph.invoke({
            "user_request": request.content,
            "category": "info",
            "refined_note": "",
            "title": "",
            "user_decision": ""
        }, config)

        # 결과 추출
        refined_title = final_state.get("title", "제목 없음")
        refined_text = final_state.get("refined_note", "정리 실패")

        # 명세서 응답 규격: 최초 생성된 시간 Unix epoch time (초)
        # 서버에서 현재 시간을 생성하여 저장 및 반환합니다.
        current_unix_time = int(time.time())

        # daily_table에 최종 결과 저장
        DAILY_TABLE.put_item(Item={
            "user_key": uid,
            "creation_date": current_unix_time,  # 테이블 스키마 키 명칭 준수
            "title": refined_title,
            "content": refined_text
        })

        # 명세서 Response 규격 준수
        return {
            "creation_date": current_unix_time,
            "title": refined_title,
            "content": refined_text,
            "keyword": []  # 명세서상 옵션 필드
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))