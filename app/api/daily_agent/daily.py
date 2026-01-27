import time
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.core.security import verify_cognito_token
from app.core.database import get_table

# 에이전트 및 그래프 관련
from .graph import app_graph

router = APIRouter()

# 테이블 객체 로드
ORIGIN_TABLE = get_table('origin_table')
PRE_TABLE = get_table('pre_table')
DAILY_TABLE = get_table('daily_table')
KEYWORD_TABLE = get_table('keyword_table') # 키워드 통계용 테이블 추가


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

        # 1. AI 에이전트 실행 (GraphState에 따라 제목, 본문, 키워드 추출)
        config = {"configurable": {"thread_id": uid}}
        final_state = app_graph.invoke({
            "user_request": request.content,
            "iteration_count": 0,
            "reflection_feedback": ""
        }, config)

        # 2. 결과 데이터 추출
        report_data = final_state.get("refined_note")
        refined_title = final_state.get("title", "제목 없음") # AI가 생성한 구체적 제목
        refined_text = str(report_data)
        extracted_keywords = final_state.get("keywords", []) # AI가 추출한 가변적 키워드 리스트

        # 3. 공통 타임스탬프 생성 (모든 테이블의 연결 고리)
        current_unix_time = int(time.time())

        # 4. [Origin Table] 순수 원본 저장
        ORIGIN_TABLE.put_item(Item={
            "user_key": uid,
            "creation_date": current_unix_time,
            "content": request.content
        })

        # 5. [Pre Table] 정제된 전처리본 저장
        PRE_TABLE.put_item(Item={
            "user_key": uid,
            "creation_date": current_unix_time,
            "title": refined_title,
            "content": final_state.get("preprocessed_request", request.content)

        })

        # 6. [Daily Table] 최종 정리본 저장
        item_to_store = {
            "user_key": uid,
            "creation_date": current_unix_time,
            "title": refined_title,
            "content": refined_text,
            "keywords": extracted_keywords

        }
        DAILY_TABLE.put_item(Item=item_to_store)

        # ---------------------------------------------------------
        # 7. [Keyword Table] 키워드별 통계 업데이트 (Upsert 로직)
        # ---------------------------------------------------------
        # 작동 원리:
        # - 신규 키워드: count 1로 생성 및 리스트 초기화
        # - 기존 키워드: count 원자적 증가 및 최신 시간 갱신
        for word in extracted_keywords:
            try:
                KEYWORD_TABLE.update_item(
                    Key={
                        'user_key': uid,
                        'key_word': word
                    },
                    # #c, #d, #u는 count, data, update_date 예약어 충돌 방지용 별칭
                    UpdateExpression="""
                        SET #c = if_not_exists(#c, :zero) + :inc,
                            #d = if_not_exists(#d, :empty_list),
                            #u = :now
                    """,
                    ExpressionAttributeNames={
                        '#c': 'key_count',
                        '#d': 'data',
                        '#u': 'update_date'
                    },
                    ExpressionAttributeValues={
                        ':inc': 1,
                        ':zero': 0,
                        ':empty_list': [],
                        ':now': current_unix_time # daily_table과 동일한 시간 사용
                    }
                )
            except Exception as kw_e:
                # 키워드 업데이트 실패가 전체 프로세스를 중단시키지 않도록 예외 처리
                print(f"Keyword update failed for '{word}': {kw_e}")

        return item_to_store

    except Exception as e:
        print(f"Error details: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")