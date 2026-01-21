# InfoAgent (정보 저장 특화 예시)
# app/api/daily_agent/types/info.py
from ..base import BaseAgent

class InfoAgent(BaseAgent):
    def get_few_shot(self) -> str:
        return """
[예시 1: 연락처/보안]
입력: "철수네 현관 비밀번호 0405* 이고 와이파이 비번은 chulsoo123 이래."
결과: ### 📌 정보 저장
- **분류**: 보안 및 네트워크 (철수네 집)
- **핵심 데이터**: 
    - 현관 PW: `0405*`
    - Wi-Fi: `chulsoo123`
- **비고**: 방문 시 참고

[예시 2: 맛집/장소 기록]
입력: "연남동에 '소이연남' 쌀국수 맛있음. 월요일은 휴무니까 피해서 가야 함."
결과: ### 📌 정보 저장
- **분류**: 맛집 리스트 (연남동)
- **핵심 데이터**: 소이연남 (태국 쌀국수)
- **비고**: 매주 월요일 정기 휴무
"""

    def get_template(self) -> str:
        return """### 📌 정보 저장
- **분류**: {어떤 종류의 정보인가요?}
- **핵심 데이터**: {데이터 본문}
- **비고**: {기타 참고 사항}"""