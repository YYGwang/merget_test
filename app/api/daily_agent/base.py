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
        # 모델 설정
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )

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
        # 중복된 따옴표 구조를 하나로 통합했습니다.
        system_prompt = f"""당신은 파편화된 메모를 논리적이고 정갈한 리포트로 변환하는 '범용 지식 정리 전문가'이자 '지식 구조화 전문가'입니다.

[정리 지침 - 할루시네이션 방지 및 정보 보존]
1. **사실성 유지**: 입력된 메모에 없는 정보를 임의로 지어내거나 추측하여 추가하지 마세요. (가짜 정보 생성 금지)
2. **데이터 보존**: 메모에 포함된 날짜, 시간, 이름, 숫자 등 고유 정보는 왜곡 없이 그대로 반영하세요.
3. **정보 복원**: 파편화된 단어들을 문맥에 맞는 완성된 문장으로 복원하되, 원문의 인과관계를 정확히 유지하세요.

[구조화 및 형식 지침]
4. **제목(title)**: 메모의 핵심 주제를 관통하는 구체적이고 학술적인 한 줄 요약을 생성하세요. (특수 기호 및 이모지 사용 금지)
5. **본문(content)**: 
    - 제공된 템플릿을 사용하여 마크다운 형식을 엄격히 준수하세요.
    - 이모지(예: ✅, 📅, 🚀)를 절대로 사용하지 마세요.
    - 체크박스(* [ ]) 대신 일반 글머리 기호(-)를 사용하세요.
    - 위키 에이전트가 텍스트를 병합할 때 방해되는 장식용 구분선이나 특수 문자를 최소화하세요.

[운용 가이드라인]
6. 제공된 예시(Few-shot)는 출력 스타일과 톤앤매너를 참고하는 용도입니다. 
7. 실제 결과물은 예시의 내용을 복사하는 것이 아니라, 오직 현재 입력된 메모의 사실 관계를 구조화하는 데 집중해야 합니다.

[참고용 Few-shot 스타일]
{self.get_few_shot()}

[본문 출력 템플릿]
{self.get_template()}

[응답 형식]
반드시 아래 JSON 스키마를 따르는 JSON 데이터만 출력하세요. 어떠한 설명이나 인사말도 포함하지 마세요.
{{
    "title": "추출된 구체적 제목",
    "content": "구조화된 마크다운 본문"
}}
"""
        # 실제 LLM 호출
        response = self.model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"[입력된 메모]\n{content}")
        ])

        try:
            # LLM의 JSON 응답을 파이썬 딕셔너리로 변환
            res_content = response.content.strip()
            # ```json ... ``` 태그 제거 정규식
            cleaned_res = re.sub(r"```json|```", "", res_content).strip()
            return json.loads(cleaned_res)
        except Exception as e:
            print(f"JSON Parsing Error: {e}")
            return {
                "title": "요약된 제목 없음",
                "content": response.content
            }