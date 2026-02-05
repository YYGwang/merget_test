import json
import re
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings

load_dotenv()


class BaseAgent(ABC):
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )

    @abstractmethod
    def get_instruction(self) -> str:
        pass

    @abstractmethod
    def get_few_shot(self) -> str:
        pass

    @abstractmethod
    def get_template(self) -> str:
        pass

    # -------------------------
    # Helpers
    # -------------------------
    def _extract_json_block(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.strip()
        cleaned = re.sub(r"```json|```", "", cleaned).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        return match.group().strip() if match else cleaned


    def _normalize_md_hierarchy(self, text: str) -> str:
        """
        - 상위는 '1. 제목' 유지
        - '1-1.' 같은 하위 번호만 '(1)'로 바꿔서 보기 좋게
        - '-' 불릿은 유지(강제 변환하지 않음)
        """
        if not text:
            return text

        lines = text.splitlines()
        out = []
        sub_idx = 0
        in_top = False

        top_pat = re.compile(r"^\s*\d+\.\s+.+$")
        sub_hyphen_num_pat = re.compile(r"^\s*\d+\s*[-–]\s*\d+\.\s+(.+?)\s*$")  # 1-1. / 2-3.

        for raw in lines:
            line = raw.rstrip()

            if top_pat.match(line.strip()):
                out.append(line.strip())
                sub_idx = 0
                in_top = True
                continue

            if in_top:
                m = sub_hyphen_num_pat.match(line)
                if m:
                    sub_idx += 1
                    out.append(f"({sub_idx}) {m.group(1).strip()}")
                    continue

            out.append(line)

        text2 = "\n".join(out)
        text2 = re.sub(r"\n{3,}", "\n\n", text2).strip()
        return text2


    # -------------------------
    # Main
    # -------------------------
    def organize(self, content: str, *, category: str) -> dict:
        content = (content or "").strip()

        system_prompt = f"""당신은 파편화된 메모를 단순 요약하는 사람이 아니라,
내용을 재분류하고 구조를 새로 설계하는 "편집자(Structure Editor)"입니다.

[현재 카테고리]
- {category}

[핵심 역할 규정]
- 요약자가 아니라 편집자입니다.
- 입력 메모의 순서를 그대로 따를 필요는 없습니다.
- 의미적으로 관련된 내용은 병합하고, 암묵적인 개념은 명시적인 제목으로 승격하세요.
- 단, 입력에 없는 정보를 새로 만들어내지 마세요. (추측/상상 금지)

[구조화 규칙 — 최우선]
1. 상위 주제는 번호(1., 2., 3.)로 작성하세요.
2. 각 상위 주제 아래에 소주제가 있으면 괄호 번호((1), (2), (3)...)로 작성하세요.
3. 소주제 아래의 설명은 기본적으로 서술형 문장(문단)으로 작성하세요. (불릿 남발 금지)
4. '-' 불릿은 금지하지 않습니다. 다만 다음 경우에만 사용하세요:
   - 정말 중요한 요점/결론/주의사항을 짧게 나열할 때
   - 체크리스트 형태가 더 읽기 좋은 경우
   - 비교/장단점/항목 나열이 본문에 이미 존재하는 경우
5. 같은 줄에 여러 항목을 붙여 쓰지 말고, 줄바꿈을 지켜 가독성을 유지하세요.
6. 입력에 없는 내용을 새로 만들지 마세요. (추측/상상 금지)
7. 절대 내용을 압축하지 마세요. 입력된 모든 기술적 세부 사항, 통계 수치($10^{26}$ FLOPS 등), 기업별 동향을 각각 독립된 섹션으로 구성하여 최대한 상세하게 기술하세요. 문서가 길어지더라도 모든 정보를 보존하는 것이 최우선 과제입니다.

[표 사용 규칙]
- 장단점, 비교, 분류, 대조 관계가 있는 경우 표 형태로 재구성하세요.
- 단, 입력에 비교 근거가 없으면 표를 억지로 만들지 마세요.

[제목 생성 규칙]
- "메모 요약", "정리 내용" 같은 추상적인 제목은 금지합니다.
- 본문 핵심 대상(Who/What)이 드러나는 구체적 제목을 생성하세요.
- title에 입력에 없는 목적/의도 단어(계획/전략/로드맵/가이드/보고서 등)를 새로 넣지 마세요.

[금지 사항]
- 설명/해설/주석을 절대 출력하지 마세요. 오직 JSON만 출력하세요.
- 입력 메모에 없는 정보를 새로 만들어내지 마세요.
- 이모지, 체크박스, 감정 표현 금지.

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
            res_content = (response.content or "").strip()
            json_str = self._extract_json_block(res_content)
            parsed = json.loads(json_str)

            # 후처리 없이 바로 반환
            return parsed

        except Exception as e:
            print(f"JSON Parsing Error: {e}")
            # 에러 발생 시 최소한의 구조만 맞춰서 반환
            return {
                "title": "파싱 실패(원문 기반)",
                "content": content,
                "keywords": []
            }
