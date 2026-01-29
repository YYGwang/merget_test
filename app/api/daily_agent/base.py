import json
import re
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from langchain_openai import ChatOpenAI

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

    # -------------------------
    # Helpers
    # -------------------------
    def _extract_json_block(self, text: str) -> str:
        """
        응답에서 JSON 객체 블록만 최대한 안전하게 추출합니다.
        - ```json ... ``` 제거
        - 가장 바깥 { ... } 추출
        """
        if not text:
            return ""

        cleaned = text.strip()
        cleaned = re.sub(r"```json|```", "", cleaned).strip()

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        return match.group().strip() if match else cleaned

    def _fallback_keywords(self, content: str, max_k: int = 8) -> list[str]:
        """
        명사 전용 fallback 키워드 추출
        - 한글 명사 느낌만 허용
        - 조사/어미/동사/형용사 전부 제거
        - 의미 없는 일반어 제거
        """
        if not content:
            return []

        # 1) 토큰 후보 (한글 2글자 이상 + 영문/숫자)
        tokens = re.findall(r"[A-Za-z0-9_]+|[가-힣]{2,}", content)

        # 2) 제거할 패턴들 (❗핵심)
        banned_suffixes = (
            "합니다", "했습니다", "같습니다", "됩니다", "이다", "있다", "없다",
            "하기", "하기를", "하기에",
            "하는", "한", "할",
            "으로", "로", "에서", "에게", "께", "보다",
            "은", "는", "이", "가", "을", "를", "의", "와", "과", "도", "만"
        )

        # 의미 없는 일반 명사(검색가치 없음)
        stopwords = {
            "것", "수", "때", "경우", "부분", "관련", "기능", "내용",
            "문제", "방안", "제안", "정도", "사용"
        }

        seen = set()
        out: list[str] = []

        for t in tokens:
            t = t.strip()

            # 길이 컷
            if len(t) < 2:
                continue

            # 조사/어미/동사형 컷
            if t.endswith(banned_suffixes):
                continue

            # 불용 명사 컷
            if t in stopwords:
                continue

            # 중복 컷
            if t in seen:
                continue

            seen.add(t)
            out.append(t)

            if len(out) >= max_k:
                break

        return out

    def _sanitize_title(self, title: str, original: str, is_short: bool) -> str:
        """
        제목이 입력에 없는 '의도/문서형 단어(계획 등)'를 붙여서 과해석하는 문제 방지.
        - is_short일 때 더 강하게 적용
        """
        title = (title or "").strip()

        # 자주 튀는 의도/문서형 단어들
        banned = [
            "계획", "목표", "전략", "로드맵", "가이드", "보고서",
            "정리본", "요약본", "매뉴얼", "문서", "분석", "리포트",
            "플랜", "plan", "roadmap", "strategy", "report", "guide",
        ]

        if is_short:
            for w in banned:
                if w in title:
                    title = title.replace(w, "").strip()

            # 너무 짧아지거나 비면 입력 기반 중립 제목으로 재생성 (생성 최소)
            if len(title) < 4:
                kws = self._fallback_keywords(original, max_k=4)
                if kws:
                    if len(kws) == 1:
                        title = f"{kws[0]} 메모"
                    else:
                        title = f"{', '.join(kws)} 메모"
                else:
                    title = "메모"

            title = re.sub(r"\s{2,}", " ", title).strip()
            title = title.strip(" -:·")
            if len(title) > 60:
                title = title[:60] + "…"

        return title or "제목 미기재"

    def _make_summary_from_content(self, content: str, max_sentences: int = 3) -> str:
        """
        non-short에서 요약이 누락된 경우에만 사용.
        본문의 상위 헤더(1., 2., 3.)를 문장으로 이어
        최대 3문장 요약을 만든다. (생성 최소화)
        """
        heads = []
        for line in (content or "").splitlines():
            s = line.strip()
            m = re.match(r"^(\d+)\.\s+(.+)$", s)
            if m:
                heads.append(m.group(2).strip())
            if len(heads) >= max_sentences:
                break

        if not heads:
            return "요약: 메모를 구조화하여 정리했다."

        # 문장으로 변환
        sentences = [f"{h}." if not h.endswith(".") else h for h in heads]
        summary_text = " ".join(sentences[:max_sentences])
        return f"요약: {summary_text}"

    def _normalize_md_hierarchy(self, text: str) -> str:
        """
        Markdown 정리:
        - 1-1., 2-3. 같은 하위 번호를 '-' 불릿으로 변환
        - 상위 번호(1., 2.)는 유지
        """
        if not text:
            return text

        # 하위 번호 (1-1., 2-3., 10-2. 등) → '-'
        text = re.sub(
            r"(?m)^\s*\d+-\d+\.\s+",
            "- ",
            text
        )

        # 혹시 줄 중간에 끼어든 경우도 커버
        text = re.sub(
            r"\s+\d+-\d+\.\s+",
            "\n- ",
            text
        )

        # 정리
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _postprocess(self, data: dict, original: str, is_short: bool) -> dict:
        """
        모델 출력 후 최소 방어:
        - 키 누락 보정
        - keywords 최소 1개 보장
        - title 의도 단어 과해석 방지
        - non-short일 때 요약 섹션 강제(누락 시 보정)
        """
        if not isinstance(data, dict):
            data = {}

        title = data.get("title") if isinstance(data.get("title"), str) else ""
        content = data.get("content") if isinstance(data.get("content"), str) else ""
        keywords = data.get("keywords") if isinstance(data.get("keywords"), list) else []

        if not keywords:
            keywords = []

        title = self._sanitize_title(title, original=original, is_short=is_short)

        # non-short일 때만 요약 강제. short_memo(is_short=True)면 요약 안 함.
        if (not is_short) and content:
            if not content.lstrip().startswith("요약:"):
                summary = self._make_summary_from_content(content, max_sentences=3)
                content = f"{summary}\n\n{content}"

            # 요약이 너무 길어지지 않게 상한(안전장치)
            # (요약 섹션이 10줄 이상이면 앞부분 4줄까지만 유지)
            lines = content.splitlines()
            if lines and (lines[0].strip().startswith("요약")):
                # 요약 섹션이 끝나는 지점: 빈 줄 전까지
                summary_block = []
                rest_block = []
                in_summary = True
                for ln in lines:
                    if in_summary:
                        summary_block.append(ln)
                        if ln.strip() == "":
                            in_summary = False
                    else:
                        rest_block.append(ln)

                if len(summary_block) > 6:  # "요약:" + 최대 5줄 정도만 허용
                    summary_block = summary_block[:6]
                    if summary_block[-1].strip() != "":
                        summary_block.append("")
                    content = "\n".join(summary_block + rest_block)

        return {
            "title": title,
            "content": content.strip() or original.strip(),
            "keywords": keywords
        }

    # -------------------------
    # Main
    # -------------------------
    def organize(self, content: str, *, category: str, is_short: bool) -> dict:
        """
        category: note/meeting/planner/memo (router에서 결정)
        is_short: 짧은 글(정보 부족) 플래그 (graph에서 계산)
        """
        content = (content or "").strip()

        # SHORT MODE: 추출/재배치만 허용, 확장/생성 금지
        short_mode_rules = ""
        if is_short:
            short_mode_rules = """
[SHORT MODE — 정보가 부족한 짧은 입력 전용 규칙]
- 입력에 없는 사실/정의/배경/예시/근거를 절대 추가하지 마세요.
- 문단을 늘리거나 설명을 확장하지 마세요. (추출/재배치만 허용)
- 템플릿의 항목이 비면 '미기재'로 두세요.
- keywords는 반드시 입력에 등장한 단어에서만 추출하세요. (새 키워드 생성 금지)
- title에 입력에 없는 목적/의도 단어(예: 계획, 목표, 전략, 로드맵, 가이드, 보고서)를 추가하지 마세요.
"""

        # non-short일 때만 요약 섹션을 LLM이 직접 작성하도록 유도
        summary_rules = ""
        if not is_short:
            summary_rules = """
[요약 규칙]
- content의 맨 앞에 '요약:'으로 시작하는 요약 문단을 작성하세요.
- 요약은 최대 3문장까지만 허용합니다.
- 줄바꿈은 허용하되, bullet/번호/목록 형태는 사용하지 마세요.
- 요약은 본문에 작성한 내용을 바탕으로만 작성하세요.
- 입력에 없는 목적/의도 단어(계획/전략/로드맵/가이드/보고서 등)를 추가하지 마세요.

[요약 형식 예시]
요약: 첫 번째 요약 문장입니다. 두 번째 요약 문장입니다.
"""

        system_prompt = f"""당신은 파편화된 메모를 단순 요약하는 사람이 아니라,
내용을 재분류하고 구조를 새로 설계하는 "편집자(Structure Editor)"입니다.

[현재 카테고리]
- {category}

{short_mode_rules}
{summary_rules}

[핵심 역할 규정]
- 요약자가 아니라 편집자입니다.
- 입력 메모의 순서를 그대로 따를 필요는 없습니다.
- 의미적으로 관련된 내용은 병합하고, 암묵적인 개념은 명시적인 제목으로 승격하세요.
- 단, 입력에 없는 정보를 새로 만들어내지 마세요. (추측/상상 금지)

[구조화 규칙 — 최우선]
1. 상위 주제만 번호(1., 2., 3.)를 사용하세요.
2. 하위 항목에는 번호(1-1., 1-2., 2-1. 등)를 절대 사용하지 마세요.
3. 하위 항목은 반드시 Markdown 불릿("- ")으로 작성하세요.
4. 하나의 상위 번호 아래에는 여러 개의 "-" 항목이 올 수 있습니다.
5. 같은 줄에 두 개 이상의 항목을 작성하지 마세요.
6. 모든 항목은 반드시 새 줄에서 시작해야 합니다.


[재구성 규칙]
- 단편적인 문장은 맥락에 맞게 재배치하세요.
- 중요 키워드, 주장, 관점은 독립된 번호 제목으로 분리하세요.
- 판단, 비판, 한계, 비교가 등장하면 별도의 섹션으로 분리하세요.
- **내용 보존**: 입력된 메모의 수치/라이브러리/고유명사/구체 사례는 가능한 한 보존하세요.
  (단, SHORT MODE에서는 '보존'은 하되 '확장'은 금지)

[표 사용 규칙]
- 장단점, 비교, 분류, 대조 관계가 있는 경우 표 형태로 재구성하세요.
- 단, 입력에 비교 근거가 없으면 표를 억지로 만들지 마세요.

[제목 생성 규칙]
- "메모 요약", "정리 내용" 같은 추상적인 제목은 금지합니다.
- 본문 핵심 대상(Who/What)이 드러나는 구체적 제목을 생성하세요.
- title에 입력에 없는 목적/의도 단어(계획/전략/로드맵/가이드/보고서 등)를 새로 넣지 마세요.

[키워드 추출 규칙]
- 본문에 등장하는 핵심 개념/고유명사/기술 용어를 추출하세요.
- 검색/분류에 도움이 되는 단어만 포함하세요.
- 최소 1개 이상 반드시 생성하세요.
- SHORT MODE에서는 입력에 있는 단어만 사용하세요.
- 절대 형용사, 동사, 부사는 사용하지 마세요.
- 명사만 사용하시오

[금지 사항]
- 단순 요약, 문단 압축, 원문 나열은 금지합니다.
- 이모지, 체크박스, 감정 표현을 사용하지 마세요.
- 입력 메모에 없는 정보를 새로 만들어내지 마세요.
- 설명/해설/주석을 절대 출력하지 마세요. 오직 JSON만 출력하세요.

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
            res_content = (response.content or "").strip()
            json_str = self._extract_json_block(res_content)
            parsed = json.loads(json_str)
            return self._postprocess(parsed, original=content, is_short=is_short)
        except Exception as e:
            # 파싱 실패 시에도 keywords/title/content 최소 보장
            print(f"JSON Parsing Error: {e}")
            return self._postprocess(
                {
                    "title": "파싱 실패(원문 기반)",
                    "content": content,
                    "keywords": []
                },
                original=content,
                is_short=is_short
            )
