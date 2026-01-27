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
        system_prompt = f"""당신은 파편화된 메모를 단순 요약하는 사람이 아니라, 
내용을 재분류하고 구조를 새로 설계하는 "편집자(Structure Editor)"입니다.

당신의 임무는 입력된 메모를 기반으로, 정보의 중요도와 논리적 관계를 판단하여 
설명형 구조 노트 형태로 재작성하는 것입니다.

[핵심 역할 규정]
- 요약자가 아니라 편집자입니다.
- 입력 메모의 순서를 그대로 따를 필요는 없습니다.
- 의미적으로 관련된 내용은 병합하고, 암묵적인 개념은 명시적인 제목으로 승격하세요.
- 단순 문단 나열은 절대 허용되지 않습니다.

[구조화 규칙 — 최우선]
1. 본문(content)은 반드시 번호 기반 계층 구조를 사용해야 합니다.
   - 예: 1., 1-1., 1-2., 2., 2-1.
2. 하나의 번호에는 하나의 주제만 포함하세요.
3. 상위 번호는 "개념/주제", 하위 번호는 "설명/근거/세부 내용" 역할을 가져야 합니다.
4. 번호 없는 문단 출력은 금지합니다.

[재구성 규칙]
- 단편적인 문장은 맥락에 맞게 재배치하세요.
- 중요 키워드, 주장, 관점은 독립된 번호 제목으로 분리하세요.
- 판단, 비판, 한계, 비교가 등장하면 별도의 섹션으로 분리하세요.
- **내용 보존**: 입력된 메모의 모든 기술적 수치, 라이브러리 목록, 구체적 사례를 요약하지 말고 최대한 상세히 본문에 포함하세요.

[표 사용 규칙]
- 장단점, 비교, 분류, 대조 관계가 있는 경우 반드시 표 형태로 재구성하세요.
- 표를 사용하지 않고 나열하는 것은 허용되지 않습니다.

[제목 생성 규칙]
- "메모 요약", "정리 내용" 같은 추상적인 제목은 금지합니다.
- 본문의 핵심 대상(Who/What)과 주요 논점이 드러나는 구체적인 문장형 제목을 생성하세요.

[키워드 추출 규칙]
- 본문에 등장하는 핵심 개념, 고유 명사, 기술 용어 등을 개수 제한 없이 모두 추출하세요.
- 검색 및 분류에 실질적으로 도움이 되는 단어만 포함하며, 최소 1개 이상 반드시 생성하세요.

[금지 사항]
- 단순 요약, 문단 압축, 원문 나열은 금지합니다.
- 이모지, 체크박스, 불필요한 감정 표현을 사용하지 마세요.
- 입력 메모에 없는 정보를 새로 만들어내지 마세요.

[출력 엄수]
- 설명, 해설, 주석을 절대 출력하지 마세요.
- 오직 지정된 JSON 스키마에 맞는 JSON 객체만 출력하세요.

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
    "content": "구조화된 본문",
    "keywords": ["키워드A", "키워드B", "키워드C", ...]
}}
"""
        response = self.model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"[입력된 메모]\n{content}")
        ])

        try:
            res_content = response.content.strip()
            # 정규표현식을 사용하여 JSON 블록만 안전하게 추출
            json_match = re.search(r'\{.*\}', res_content, re.DOTALL)
            if json_match:
                cleaned_res = json_match.group()
            else:
                cleaned_res = re.sub(r"```json|```", "", res_content).strip()

            return json.loads(cleaned_res)
        except Exception as e:
            print(f"JSON Parsing Error: {e}")
            return {
                "title": "요약된 제목 없음",
                "content": response.content,
                "keywords": []
            }