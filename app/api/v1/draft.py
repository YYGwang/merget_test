from fastapi import APIRouter, Depends, HTTPException
from boto3.dynamodb.conditions import Key
from app.core.security import verify_cognito_token
from app.core.database import get_table

router = APIRouter()
DRAFT_TABLE = get_table('draft_note')

@router.get("")
async def get_draft(
        uid: str = Depends(verify_cognito_token)
):
    """
        Cognito UID를 기반으로 draft_note 테이블에서 해당 유저의 모든 노트를 가져옵니다.
    """
    try:
        response = DRAFT_TABLE.query(
            KeyConditionExpression=Key('user_key').eq(uid)
        )
        items = response.get('Items', [])

        if not items:
            return {"message": "데이터가 존재하지 않습니다.", "uid": uid}
        return items
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


