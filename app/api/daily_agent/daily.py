from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.core.security import verify_cognito_token
from app.core.database import get_table
import time
from fastapi import File, UploadFile

# 에이전트 및 그래프 관련
from .graph import app_graph
from .utils.pdfparser import PDFParser
import os

import boto3
from urllib.parse import urlparse
# ... (기존 import 생략)

# S3 클라이언트 전역 선언 혹은 함수 내부 선언
s3_client = boto3.client('s3')

router = APIRouter()

# 테이블 객체 로드
ORIGIN_TABLE = get_table("origin_table")
PRE_TABLE = get_table("pre_table")
DAILY_TABLE = get_table("daily_table")
KEYWORD_TABLE = get_table("keyword_table")  # 키워드 통계용 테이블
TRIPLE_TABLE = get_table("triple_table")    # triple 저장용 테이블

class DailyNoteRequest(BaseModel):
    input_type: str  # text, file, image, audio
    content: str


def move_s3_file(s3_url: str, user_key: str, creation_date: int):
    """S3 내부: temp -> {creation_date} 폴더로 이동"""
    parsed = urlparse(s3_url)
    bucket = parsed.netloc.split('.')[0]
    old_key = parsed.path.lstrip('/')

    filename = os.path.basename(old_key)
    new_key = f"{user_key}/{creation_date}/{filename}"

    # 복사 후 삭제 (S3 Move 방식)
    s3_client.copy_object(Bucket=bucket, CopySource={'Bucket': bucket, 'Key': old_key}, Key=new_key)
    s3_client.delete_object(Bucket=bucket, Key=old_key)
    return f"s3://{bucket}/{new_key}"

@router.post("")
async def create_daily_report(
    request: DailyNoteRequest,
    uid: str = Depends(verify_cognito_token)
):
    try:
        if not request.content:
            raise HTTPException(status_code=400, detail="content가 없습니다.")

        # 1) AI 에이전트 실행
        # ✅ 변경: GraphState에 없는 iteration_count/reflection_feedback 제거
        config = {"configurable": {"thread_id": uid}}
        final_state = await app_graph.ainvoke(
            {
                "input_type": request.input_type,
                "content": request.content,
                "user_key":uid
            },
            config
        )

        # 2) 결과 데이터 추출
        refined_title = final_state.get("title", "제목 없음")
        refined_note = final_state.get("refined_note", "정리 실패")
        refined_text = str(refined_note)
        extracted_keywords = final_state.get("keywords", [])
        triples = final_state.get("triples", [])
        extracted_triples = [t.model_dump() for t in triples]
        extracted_abstract = final_state.get("abstract", "")

        # (선택) 디버깅용 - DB 저장은 안 하고 필요하면 응답에만 포함 가능
        category = final_state.get("category")

        # 3) 공통 타임스탬프 생성
        current_unix_time = int(time.time())

        # 2) 파일인 경우 S3 이동 처리 (추가된 부분)
        final_content_url = request.content
        # if request.input_type in ["file", "image", "audio"]:
        #    final_content_url = move_s3_file(request.content, uid, current_unix_time)


        # 3) 서버에 생성된 임시 로컬 파일 삭제 (서버 용량 관리)
        # graph.py의 각 노드가 반환한 'file_path'를 사용하여 삭제합니다.
        temp_local_path = final_state.get("file_path")
        if temp_local_path and os.path.exists(temp_local_path):
            os.remove(temp_local_path)

        # 4) 결과 데이터 추출 (기존 로직)
        refined_title = final_state.get("title", "제목 없음")  #
        # ... (이하 동일)

        # 5) [Origin Table] 저장 시 최종 이동된 S3 URL을 저장할지 선택
        # 원본 기록을 위해 이동된 경로(final_content_url)를 사용하는 것이 좋습니다.
        ORIGIN_TABLE.put_item(Item={
            "user_key": uid,
            "creation_date": current_unix_time,
            "content": final_content_url  # 이동된 경로로 업데이트
        })

        # 5) [Pre Table] 정제된 전처리본 저장
        PRE_TABLE.put_item(Item={
            "user_key": uid,
            "creation_date": current_unix_time,
            "title": refined_title,
            "content": final_state.get("preprocessed_request", request.content)
        })

        # 6) [Daily Table] 최종 정리본 저장
        item_to_store = {
            "user_key": uid,
            "creation_date": current_unix_time,
            "title": refined_title,
            "content": refined_text,
            "keywords": extracted_keywords,
            "triples": extracted_triples,
            "abstract": extracted_abstract
        }
        DAILY_TABLE.put_item(Item=item_to_store)

        # 7 [Triple Table] 트리플 테이블에 신규 트리플 저장
        TRIPLE_TABLE.put_item(Item={"user_key": uid,
                                    "creation_date": current_unix_time,
                                    "triples": extracted_triples})

        # 8) [Keyword Table] 키워드별 통계 업데이트 (Upsert)
        for word in extracted_keywords:
            try:
                KEYWORD_TABLE.update_item(
                    Key={
                        "user_key": uid,
                        "key_word": word
                    },
                    UpdateExpression="""
                        SET #c = if_not_exists(#c, :zero) + :inc,
                            #d = if_not_exists(#d, :empty_list),
                            #u = :now
                    """,
                    ExpressionAttributeNames={
                        "#c": "key_count",
                        "#d": "data",
                        "#u": "update_date"
                    },
                    ExpressionAttributeValues={
                        ":inc": 1,
                        ":zero": 0,
                        ":empty_list": [],
                        ":now": current_unix_time
                    }
                )
            except Exception as kw_e:
                print(f"Keyword update failed for '{word}': {kw_e}")

        return item_to_store

    except Exception as e:
        print(f"Error details: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
