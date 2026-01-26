import time
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.core.security import verify_cognito_token
from app.core.database import get_table

# 에이전트 및 그래프 관련 (이미 정의된 객체들)
from .graph import app_graph

router = APIRouter()
ORIGIN_TABLE = get_table('origin_table')
PRE_TABLE = get_table('pre_table')
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
            "reflection_feedback": ""
        }, config)

        # 2. 결과 데이터 추출
        report_data = final_state.get("refined_note")
        # 에이전트가 생성한 제목을 모든 테이블의 공통 제목으로 사용합니다.
        refined_title = final_state.get("title", "제목 없음")
        refined_text = str(report_data)

        # 3. 공통 타임스탬프 생성 (세 테이블 간 연결 고리)
        current_unix_time = int(time.time())

        # ---------------------------------------------------------
        # 4. [Origin Table] 순수 원본 저장 (제목 추가)
        # ---------------------------------------------------------
        ORIGIN_TABLE.put_item(Item={
            "user_key": uid,
            "creation_date": current_unix_time,
            "content": request.content
        })

        # ---------------------------------------------------------
        # 5. [Pre Table] 정제된 전처리본 저장 (제목 추가)
        # ---------------------------------------------------------
        PRE_TABLE.put_item(Item={
            "user_key": uid,
            "creation_date": current_unix_time,
            "title": refined_title,  # 공통 제목 추가
            "content": final_state.get("preprocessed_request", request.content)
        })

        # ---------------------------------------------------------
        # 6. [Daily Table] 최종 정리본 저장
        # ---------------------------------------------------------
        item_to_store = {
            "user_key": uid,
            "creation_date": current_unix_time,
            "title": refined_title,
            "content": refined_text
        }
        DAILY_TABLE.put_item(Item=item_to_store)

        return item_to_store

    except Exception as e:
        print(f"Error details: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")