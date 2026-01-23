import time
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
# 기존에 정의된 app_graph, get_table, verify_cognito_token 등을 사용합니다.
from app.core.security import verify_cognito_token
from app.core.database import get_table
from .graph import app_graph


router = APIRouter()
DAILY_TABLE = get_table('daily_table')

# 명세서: Request Body { content: str }
class DailyNoteRequest(BaseModel):
    content: str

@router.post("/agent/daily")
async def generate_and_save_daily_note(
    request: DailyNoteRequest,
    uid: str = Depends(verify_cognito_token)
):
    try:
        # 1. content 존재 여부 체크
        if not request.content:
            raise HTTPException(status_code=400, detail="분석할 내용이 없습니다.")

        # 2. AI 에이전트(LangGraph) 실행
        # 프론트에서 넘어온 history_table.content를 분석합니다.
        config = {"configurable": {"thread_id": uid}}
        final_state = app_graph.invoke({
            "user_request": request.content,
            "iteration_count": 0,
            "research_plan": "",
            "reflection_feedback": ""
        }, config)

        # AI 결과물 추출
        refined_note = final_state.get("refined_note")
        if not refined_note:
            raise HTTPException(status_code=500, detail="AI 리포트 생성 실패")

        # ---------------------------------------------------------
        # 3. 새로운 creation_date 생성 (명세서의 Unix epoch time 형식)
        # ---------------------------------------------------------
        new_creation_date = int(time.time())

        # 4. daily_table 스키마에 맞춰 저장
        # 이미지에 제공된 [daily_table] 구조를 따릅니다.
        DAILY_TABLE.put_item(Item={
            "user_key": uid,                 # user의 uid
            "creation_date": new_creation_date, # 새로 생성된 시간 (number)
            "title": refined_note.title,      # 제목 (주제)
            "content": refined_note.refined_text # 정리된 본문
        })

        # 5. 명세서 return 규격에 맞춰 반환
        return {
            "user_key": uid,
            "creation_date": new_creation_date,
            "title": refined_note.title,
            "content": refined_note.refined_text
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))