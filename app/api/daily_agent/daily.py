import re
from fastapi import APIRouter, HTTPException, Depends, Path
from app.core.security import verify_cognito_token
from app.core.database import get_table

# 모듈화된 LangGraph 객체 임포트
from .graph import app_graph

# app/api/daily_agent/daily.py
# 기존: router = APIRouter()
router = APIRouter()

# 테이블 정의 변경: history(원본) -> daily(결과 저장)
HISTORY_TABLE = get_table('history_table')
DAILY_TABLE = get_table('daily_table')  # 기존 confirm에서 daily로 변경


@router.post("/generate-report/{creation_date}")
async def create_report(
        creation_date: int = Path(..., description="history -> content를 가져오기 위한 유닉스 타임스탬프"),
        uid: str = Depends(verify_cognito_token)
):

    try:
        # 1. history 테이블에서 사용자의 원본 데이터 조회
        response = HISTORY_TABLE.get_item(
            Key={"user_key": uid, "creation_date": creation_date}
        )
        item = response.get('Item')

        if not item or not item.get('content'):
            raise HTTPException(status_code=404, detail="history에서 해당 메모를 찾을 수 없습니다.")

        raw_content = item.get('content')

        # 2. 모듈화된 LangGraph 에이전트 가동
        config = {"configurable": {"thread_id": uid}}

        # 그래프 실행 (내부 분류 -> 전문가 정리 -> JSON 추출)
        final_state = app_graph.invoke({
            "user_request": raw_content,
            "category": "info",
            "refined_note": "",
            "title": "",
            "user_decision": ""
        }, config)

        # 3. 결과 추출
        refined_title = final_state.get("title", "제목 없음")
        refined_text = final_state.get("refined_note", "정리 실패")

        # 4. daily 테이블에 최종 결과 저장 (confirm -> daily 변경 반영)
        DAILY_TABLE.put_item(Item={
            "user_key": uid,
            "creation_date": creation_date,
            "title": refined_title,
            "content": refined_text
        })

        # 5. [API 명세서 Response 규격 준수]
        return {
            "creation_date": creation_date,
            "title": refined_title,
            "content": refined_text
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="리포트 생성 및 daily 저장 실패")