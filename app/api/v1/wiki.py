from fastapi import APIRouter, Depends, HTTPException
from boto3.dynamodb.conditions import Key
from app.core.security import verify_cognito_token
from app.core.database import get_table

router = APIRouter()

# wiki_note 테이블 객체 생성
# 환경 변수로 관리하는 것이 좋지만, 명시적으로 "wiki_note"를 사용합니다.
WIKI_TABLE = get_table("wiki_note")

@router.get("")
async def get_wiki(uid: str = Depends(verify_cognito_token)):
    """
    Cognito UID를 기반으로 wiki_note 테이블에서 해당 유저의 모든 노트를 가져옵니다.
    """
    try:
        # partition key인 'user_key'가 uid와 일치하는 항목들을 쿼리
        response = WIKI_TABLE.query(
            KeyConditionExpression=Key('user_key').eq(uid)
        )

        items = response.get('Items', [])

        if not items:
            # 데이터가 없을 경우 빈 리스트 혹은 메시지 반환
            return {"message": "작성된 위키 노트가 없습니다.", "items": []}

        return items

    except Exception as e:
        print(f"DynamoDB Wiki Query Error: {e}")
        raise HTTPException(status_code=500, detail="위키 데이터를 가져오는 중 오류가 발생했습니다.")