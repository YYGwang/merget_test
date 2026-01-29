from ..base import BaseAgent


class NoteAgent(BaseAgent):
    def get_instruction(self) -> str:
        return """
- 목적: 입력 메모를 '노트(note)' 형식의 마크다운 문서로 구조화합니다.
- 입력에 없는 배경지식/정의/예시/근거를 새로 만들지 마세요.
- 상위 주제만 번호(1., 2., 3.)를 사용하세요.
- 하위 항목에는 숫자 번호를 절대 사용하지 마세요. (1-1, 2-1 금지)
- 하위 항목은 반드시 '-' 불릿으로 작성하세요.
- 정보가 부족한 항목은 '미기재'로 둡니다.
"""

    def get_few_shot(self) -> str:
        return """
[입력 예시]
"JWT 만료 처리 헷갈림. refresh token 필요 이유 정리해야 함"

[출력 예시]
{
  "title": "JWT 만료 처리와 Refresh Token 필요성 메모",
  "content": "## 핵심
1. JWT 만료 처리 관련 정리 필요
- 만료 처리 방식이 헷갈림
2. Refresh Token 필요 이유 정리 필요
- Refresh Token을 왜 사용하는지 명확하지 않음

## 상세 노트
1. 현재 메모에 포함된 내용
- JWT 만료 처리
- Refresh Token 필요 이유

}
"""

    def get_template(self) -> str:
        return """
## 핵심
1. {핵심 포인트}
- {세부 설명}

## 상세 노트
1. {정리 항목}
- {세부 내용}

"""
