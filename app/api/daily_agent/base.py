import json
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class BaseAgent(ABC):
    def __init__(self):
        # 모델 설정 (GPT-4o-mini / watsonx.ai 연동 가능)
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    @abstractmethod
    def get_few_shot(self) -> str:
        """유형별 특화 Few-shot 예시를 정의합니다."""
        pass

    @abstractmethod
    def get_template(self) -> str:
        """유형별 마크다운 출력 형식을 정의합니다."""
        pass

    def organize(self, content: str) -> dict:
        """
        공통 프롬프트에 Few-shot과 템플릿을 주입하여 LLM을 호출하고,
        제목(title)과 본문(content)이 포함된 딕셔너리를 반환합니다.
        """
        system_prompt = f"""당신은 메모 정리 전문가입니다.
입력된 파편화된 메모를 분석하여 '제목'을 추출하고, 본문을 주어진 '템플릿'에 맞춰 정갈하게 정리하세요.

[정리 지침]
1. 제목(title): 메모의 핵심 내용을 관통하는 한 줄 요약 (예: [회의] 서비스 기획 로고 논의)
2. 본문(content): 아래 제공된 템플릿 형식을 반드시 준수하여 마크다운으로 작성하세요.
3. 결과는 반드시 JSON 형식으로만 출력하세요. 다른 설명은 생략합니다.

[Few-shot 예시]
{self.get_few_shot()}

[본문 출력 템플릿]
{self.get_template()}

[출력 JSON 스키마 예시]
{{
    "title": "추출된 제목",
    "content": "템플릿에 맞춰 정리된 본문"
}}
"""
        # 실제 LLM 호출
        response = self.model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"[입력된 메모]\n{content}")
        ])

        try:
            # LLM의 JSON 응답을 파이썬 딕셔너리로 변환
            # ```json ... ``` 같은 마크다운 태그가 붙어있을 경우를 대비해 strip() 처리
            raw_res = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(raw_res)
        except Exception as e:
            # 파싱 실패 시 기본 구조 반환 (Safe-guard)
            print(f"JSON Parsing Error: {e}")
            return {
                "title": "요약된 제목 없음",
                "content": response.content
            }