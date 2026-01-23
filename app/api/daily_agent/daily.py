import time
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.core.security import verify_cognito_token
from app.core.database import get_table

# 에이전트 및 그래프 관련 (이미 정의된 객체들)
from .graph import app_graph

router = APIRouter()
DAILY_TABLE = get_table('daily_table')


# [명세서 반영] Request Body 모델
class DailyNoteRequest(BaseModel):
    content: str


@router.post("")
async def create_daily_report(
        request: DailyNoteRequest,
        uid: str = Depends(verify_cognito_token)
):
    try:
        if not request.content:
            raise HTTPException(status_code=400, detail="content가 없습니다.")

        # 1. AI 에이전트 실행
        config = {"configurable": {"thread_id": uid}}
        final_state = app_graph.invoke({
            "user_request": request.content,
            "iteration_count": 0,
            "research_plan": "",
            "reflection_feedback": ""
        }, config)

        # 2. 결과 추출 및 에러 방지 처리
        # final_state["refined_note"]가 FinalReport 객체인지 확인 후 추출
        report_data = final_state.get("refined_note")

        # 만약 report_data가 객체가 아니라 문자열로 넘어올 경우를 대비한 안전장치
        if hasattr(report_data, 'refined_text'):
            refined_title = report_data.title
            refined_text = report_data.refined_text
        elif isinstance(report_data, dict):
            refined_title = report_data.get("title", "제목 없음")
            refined_text = report_data.get("refined_text", "내용 없음")
        else:
            # 에러 로그에 찍혔던 상황: report_data가 str일 경우
            refined_title = "요약 리포트"
            refined_text = str(report_data)

        # ---------------------------------------------------------
        # 3. [명세서 핵심] 새로운 creation_date 생성 (Unix epoch time)
        # ---------------------------------------------------------
        current_unix_time = int(time.time())

        # 4. [daily_table] 저장 (이미지 스키마 준수)
        # key: user_key(string), creation_date(number), title(string), content(string)
        item_to_store = {
            "user_key": uid,  # user의 uid
            "creation_date": current_unix_time,  # 최초 생성 시간 (초 단위)
            "title": refined_title,  # 제목 (주제)
            "content": refined_text  # 정리된 본문
        }

        DAILY_TABLE.put_item(Item=item_to_store)

        # 5. 결과 반환 (프론트엔드 전달용)
        return item_to_store

    except Exception as e:
        print(f"Error details: {e}")
        # 구체적인 에러 메시지를 detail에 담아 보냅니다.
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")