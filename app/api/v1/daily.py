from fastapi import APIRouter, Query, Depends, HTTPException
from boto3.dynamodb.conditions import Key
from app.core.security import verify_cognito_token
from app.core.database import get_table
from datetime import datetime, time, timezone, timedelta

router = APIRouter()

# daily_note 테이블 객체 생성
# 환경 변수로 관리하는 것이 좋지만, 명시적으로 "daily_note"를 사용합니다.
DAILY_TABLE = get_table("daily_note")

@router.get("")
async def get_daily(date: int = Query(..., description="조회할 날짜의 Unix Timestamp (초)"),
                    uid: str = Depends(verify_cognito_token)):
    """
        Cognito UID를 기반으로 daily_note 테이블에서 해당 유저가 date일에 작성한 모든 노트를 가져옵니다.
    """
    try:
        # 1. KST 타임존 설정 (UTC + 9시간)
        KST = timezone(timedelta(hours=9))

        # 2. 입력된 Timestamp를 KST 기준으로 변환
        # (입력값이 UTC 기반 timestamp여도 KST 시간대의 '날짜'를 추출하기 위함)
        dt_kst = datetime.fromtimestamp(date, tz=KST)

        # 3. KST 기준 해당 날짜의 00:00:00와 23:59:59 설정
        start_dt = datetime.combine(dt_kst.date(), time.min).replace(tzinfo=KST)
        end_dt = datetime.combine(dt_kst.date(), time.max).replace(tzinfo=KST)
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())

        response = DAILY_TABLE.query(
            KeyConditionExpression=Key('user_key').eq(uid) &
                                   Key('creation_date').between(start_ts, end_ts),
            ScanIndexForward=False
        )

        items = response.get('Items', [])

        if not items:
            # 데이터가 없을 경우 빈 리스트 혹은 메시지 반환
            return {"message": "작성된 위키 노트가 없습니다.", "items": []}

        return items

    except Exception as e:
        print(f"DynamoDB Wiki Query Error: {e}")
        raise HTTPException(status_code=500, detail="위키 데이터를 가져오는 중 오류가 발생했습니다.")