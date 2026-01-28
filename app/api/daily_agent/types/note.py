from ..base import BaseAgent


class NoteAgent(BaseAgent):
    def get_instruction(self) -> str:
        return """
- 목적: 입력 메모를 '노트(note)' 템플릿에 맞게 구조화합니다.
- 입력에 없는 배경지식/정의/예시/근거를 새로 만들지 마세요.
- 정보가 부족한 섹션은 '미기재'로 둡니다.
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
  1-1. 만료 처리가 헷갈림
2. Refresh Token 필요 이유 정리 필요
  2-1. Refresh Token을 왜 쓰는지 정리해야 함

## 상세 노트
1. 현재 메모에 포함된 내용
  1-1. JWT 만료 처리
  1-2. Refresh Token 필요 이유

## 추가로 적으면 좋은 것
- 어떤 상황에서 헷갈렸는지
- 참고 링크/자료(있다면)",
  "keywords": ["JWT", "만료", "Refresh Token"]
}
"""

    def get_template(self) -> str:
        return """
## 핵심
1. {핵심 포인트 1}
  1-1. {세부}
2. {핵심 포인트 2}
  2-1. {세부}

## 상세 노트
1. {정리 항목}
  1-1. {세부}

## 추가로 적으면 좋은 것
- {추가로 필요해 보이는 정보(있다면)}
"""
