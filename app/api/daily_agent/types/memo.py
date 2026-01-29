from ..base import BaseAgent


class MemoAgent(BaseAgent):
    def get_instruction(self) -> str:
        return """
- 목적: 정리 의도가 낮은 메모를 최소 가공하여 저장합니다.
- 입력에 없는 내용은 절대 추가하지 마세요.
- 본문은 입력 내용을 훼손하지 말고, 읽기 좋게만 정리하세요.
- 번호 구조는 사용하지 마세요.
- 불릿('-')만 사용해 간단히 표현하세요.
- 키워드는 입력에 등장한 단어만 사용하며, 생성하지 마세요.
"""

    def get_few_shot(self) -> str:
        return """
[입력 예시]
"운동 가야 함"

[출력 예시]
{
  "title": "운동 메모",
  "content": "- 운동 가야 함",
  "keywords": ["운동"]
}
"""

    def get_template(self) -> str:
        return """
- {원문 기반 메모}
"""
