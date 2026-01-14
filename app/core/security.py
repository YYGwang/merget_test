import requests
from jose import jwk, jwt
from fastapi import Header, HTTPException, Depends
from app.core.config import settings
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# 서버 시작 시 1회 캐싱
KEYS_URL = f"https://cognito-idp.{settings.REGION}.amazonaws.com/{settings.USER_POOL_ID}/.well-known/jwks.json"
jwks = requests.get(KEYS_URL).json()["keys"]

# HTTPBearer 인스턴스 생성
security = HTTPBearer()

def verify_cognito_token(res: HTTPAuthorizationCredentials = Depends(security)):
    # res.credentials에 'Bearer '가 제거된 순수 토큰값만 들어옵니다.
    token = res.credentials

    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        key = next(k for k in jwks if k["kid"] == kid)
        public_key = jwk.construct(key)

        payload = jwt.decode(
            token, public_key, algorithms=["RS256"],
            issuer=f"https://cognito-idp.{settings.REGION}.amazonaws.com/{settings.USER_POOL_ID}"
        )

        if payload.get("client_id") != settings.APP_CLIENT_ID:
            raise Exception("Invalid Client ID")

        return payload.get("sub")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


# local 환경에서 테스트용으로 사용하는 가짜 검증 함수
async def mocked_verify_cognito_token(res: HTTPAuthorizationCredentials = Depends(security)):
    token = res.credentials
    # Authorization 헤더에 넣은 값이 그대로 UUID가 되도록 설정
    print(token)
    if token != "undefined":
        return token

    return "local-test-user-uuid-12345"
