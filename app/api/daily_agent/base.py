import json
import re
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import settings
from dotenv import load_dotenv

load_dotenv()

class BaseAgent(ABC):
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )

    @abstractmethod
    def get_instruction(self) -> str:
        """노드별 특화된 상세 지침(Prompt)을 정의합니다."""
        pass

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
        각 노드에서 정의한 instruction, few-shot, template을 조합하여 실행합니다.
        """
        system_prompt = f"""당신은 파편화된 메모를 논리적이고 정갈한 리포트로 변환하는 전문가입니다.

[공통 정리 지침]
1. 사실성 유지: 입력된 메모에 없는 정보를 지어내지 마세요.
2. 데이터 보존: 날짜, 이름, 숫자 등 고유 정보는 왜곡 없이 반영하세요.
3. 형식 준수: 이모지와 체크박스 사용을 금지하며, 일반 글머리 기호(-)를 사용하세요.


[노드별 특화 상세 지침]
{self.get_instruction()}

[참고용 Few-shot 스타일]
{self.get_few_shot()}

[본문 출력 템플릿]
{self.get_template()}

[응답 형식]
반드시 아래 JSON 스키마를 따르는 JSON 데이터만 출력하세요.
{{
    "title": "추출된 구체적 제목",
    "content": "구조화된 본문"
}}
"""
        response = self.model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"[입력된 메모]\n{content}")
        ])

        try:
            res_content = response.content.strip()
            cleaned_res = re.sub(r"```json|```", "", res_content).strip()
            return json.loads(cleaned_res)
        except Exception as e:
            print(f"JSON Parsing Error: {e}")
            return {
                "title": "요약된 제목 없음",
                "content": response.content
            }