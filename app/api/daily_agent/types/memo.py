from ..base import BaseAgent


class MemoAgent(BaseAgent):
    def get_instruction(self) -> str:
        return """
- 목적: 정리 의도가 낮은 메모를 최소 가공하여 저장합니다.
- 입력에 없는 내용은 절대 추가하지 마세요.
- 본문은 입력 내용을 훼손하지 말고, 읽기 좋게만 정리하세요.
- 구조(1., (1))는 사용하지 않아도 됩니다. 짧고 단순하게 유지하세요.
- 기본은 '-' 불릿으로 정리합니다.
- 다만 입력에 '중요/주의/필수/마감' 같은 표현이 있으면 해당 줄은 맨 위에 먼저 배치해 강조할 수 있습니다. (내용 추가 금지)
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
