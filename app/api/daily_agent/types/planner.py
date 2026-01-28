from ..base import BaseAgent


class PlannerAgent(BaseAgent):
    def get_instruction(self) -> str:
        return """
- 목적: 해야 할 일/일정/마감을 'Planner' 형태로 구조화합니다.
- 입력에 없는 기한/우선순위/담당자를 절대 지어내지 마세요.
- 확인 가능한 경우에만 기한/우선순위를 적고, 없으면 '미기재'로 둡니다.
"""

    def get_few_shot(self) -> str:
        return """
[입력 예시]
"내일 아침 9시에 토익 접수. 이번주 내로 방 청소"

[출력 예시]
{
  "title": "개인 할 일 및 기한 정리",
  "content": "## Planner
1. 할 일 목록
  1-1. 토익 접수
    - 기한: 내일 09:00
    - 우선순위: 미기재
  1-2. 방 청소
    - 기한: 이번 주 내
    - 우선순위: 미기재

2. 비고
- 미기재",
  "keywords": ["토익", "접수", "청소"]
}
"""

    def get_template(self) -> str:
        return """
## Planner
1. 할 일 목록
  1-1. {할 일}
    - 기한: {기한 또는 미기재}
    - 우선순위: {우선순위 또는 미기재}
  1-2. {할 일}
    - 기한: {기한 또는 미기재}
    - 우선순위: {우선순위 또는 미기재}

2. 비고
- {추가 메모 또는 미기재}
"""
