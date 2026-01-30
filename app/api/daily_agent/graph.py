from typing import TypedDict, Literal
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import re


# ✅ 4개 특화 에이전트 클래스 임포트 (네 구조 기준)
from .types.meeting import MeetingAgent
from .types.note import NoteAgent
from .types.planner import PlannerAgent
from .types.memo import MemoAgent


# -----------------------------
# 1) Graph State
# -----------------------------
class GraphState(TypedDict, total=False):
    user_request: str                  # 사용자가 보낸 원본 텍스트
    preprocessed_request: str          # cleaner_node로 정제된 텍스트
    category: Literal["note", "meeting", "planner", "memo"]
    is_short: bool                     # ✅ 짧은 글(정보 부족) 플래그
    refined_note: str
    title: str
    keywords: list[str]
    user_decision: str
    pdf_path: str
    input_type: Literal["text", "pdf"]


# -----------------------------
# 2) Short 판단 (러프 버전)
# - char_len < 200
# - sentence_count < 5
# - no structure
# - 2-of-3 => is_short=True
# -----------------------------
STRUCTURE_PAT = re.compile(r"(\n)|(^\s*[-*•]\s+)|(^\s*\d+\.\s+)|(:)", re.MULTILINE)

def _sentence_count(text: str) -> int:
    # 아주 러프하게: 문장 경계(.?! or 줄바꿈)로 분리
    parts = re.split(r"[.!?]\s+|\n+", (text or "").strip())
    parts = [p.strip() for p in parts if p.strip()]
    return len(parts)

def _has_structure(text: str) -> bool:
    return bool(STRUCTURE_PAT.search(text or ""))

def _is_short_text(text: str) -> bool:
    t = (text or "").strip()
    char_len = len(t)
    sent_cnt = _sentence_count(t)
    has_struct = _has_structure(t)

    conditions = [
        char_len < 200,
        sent_cnt < 5,
        not has_struct,
    ]
    return sum(conditions) >= 2


# -----------------------------
# 3) Cleaner Node
# (기존 유지: 의미 훼손 없이 정제)
# -----------------------------
def cleaner_node(state: GraphState):
    model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)

    system_prompt = """당신은 요약자나 해설자가 아니라,
후속 처리를 위한 "입력 텍스트 정제 전용 에이전트"입니다.
당신의 목적은 사용자가 입력한 원문을
의미를 전혀 훼손하지 않은 상태로
읽기 쉬운 중립적 문어체 텍스트로 정제하는 것입니다.

[수행해야 할 작업]
1. 주석, 각주, 참조 번호 제거
   - 예: [12], (참고), {출처} 등
2. 명백한 오타 및 띄어쓰기 오류만 수정
3. 과도한 구어체, 커뮤니티 표현을 의미 보존 상태로 문어체로 변환
4. 문장 단위는 유지하되, 의미를 바꾸는 재구성은 하지 말 것

[절대 금지 사항]
- 요약, 압축, 정리, 분류, 구조화
- 판단, 비판, 의견 추가
- 문단 순서 변경
- 입력에 없는 정보 생성

[출력 규칙]
# - 정제된 텍스트 본문만 출력하세요.
- 제목, 번호, 불릿, 설명 문구를 추가하지 마세요.
- 원문이 단문이면 단문 형태를 유지하세요."""

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["user_request"])
    ])

    cleaned_text = (response.content or "").strip()
    return {"preprocessed_request": cleaned_text}


# -----------------------------
# 4) Router Node (4카테고리)
# - 길이로 category 결정 ❌
# - 내용/의도로만 결정 ✅
# - is_short는 여기서 계산해서 state에 저장 ✅
# -----------------------------
def router_node(state: GraphState):
    text = state.get("preprocessed_request") or state.get("user_request", "")

    t = (text or "").strip()

    is_short = _is_short_text(t)

    # ✅ 내용 단서 기반 라우팅 (규칙 기반: 흔들림 최소)
    planner_cues = ["해야", "할 일", "할일", "todo", "TODO", "계획", "마감", "까지", "기한", "우선순위", "예약", "신청", "제출", "완료"]
    meeting_cues = ["회의", "미팅", "논의", "안건", "결정", "합의", "회의록", "참석", "액션아이템", "의사결정", "agenda", "minutes"]
    note_cues = ["정리", "요약", "개념", "공부", "학습", "배운", "헷갈", "이해", "질문", "왜", "어떻게", "링크", "문서", "가이드", "스펙", "설정", "에러", "원인"]

    def hit(cues):
        lt = t.lower()
        return sum(1 for c in cues if c.lower() in lt)

    meeting_score = hit(meeting_cues)
    planner_score = hit(planner_cues)
    note_score = hit(note_cues)

    # 우선순위: meeting > planner > note > memo
    if meeting_score >= 2:
        category = "meeting"
    elif planner_score >= 2:
        category = "planner"
    elif note_score >= 1:
        category = "note"
    else:
        category = "memo"

    return {"category": category, "is_short": is_short}


# -----------------------------
# 5) Agent Nodes (4개)
# - BaseAgent.organize(content, category=..., is_short=...) 호출
# -----------------------------
def meeting_node(state: GraphState):
    text = state.get("preprocessed_request", state["user_request"])
    result = MeetingAgent().organize(text, category="meeting", is_short=state.get("is_short", False))
    return {
        "title": result.get("title", "제목 없음"),
        "refined_note": result.get("content", "정리 실패"),
        "keywords": result.get("keywords", []),
    }

def note_node(state: GraphState):
    text = state.get("preprocessed_request", state["user_request"])
    result = NoteAgent().organize(text, category="note", is_short=state.get("is_short", False))
    return {
        "title": result.get("title", "제목 없음"),
        "refined_note": result.get("content", "정리 실패"),
        "keywords": result.get("keywords", []),
    }

def planner_node(state: GraphState):
    text = state.get("preprocessed_request", state["user_request"])
    result = PlannerAgent().organize(text, category="planner", is_short=state.get("is_short", False))
    return {
        "title": result.get("title", "제목 없음"),
        "refined_note": result.get("content", "정리 실패"),
        "keywords": result.get("keywords", []),
    }

def memo_node(state: GraphState):
    text = state.get("preprocessed_request", state["user_request"])
    result = MemoAgent().organize(text, category="memo", is_short=state.get("is_short", False))
    return {
        "title": result.get("title", "제목 없음"),
        "refined_note": result.get("content", "정리 실패"),
        "keywords": result.get("keywords", []),
    }


# -----------------------------
# 6) Reflect Node (평가 X / 규칙 검사 + 보정)
# - short일 때 과확장 방지
# - keywords 비면 fallback 보강
# -----------------------------
def _fallback_keywords(text: str, max_k: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+|[가-힣]{2,}", text or "")
    seen = set()
    out = []
    for tok in tokens:
        tok = tok.strip()
        if len(tok) < 2:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= max_k:
            break
    return out

import re
from typing import Any

def reflect_node(state: "GraphState"):
    """
    Reflect 노드 (최종 승인 버전)
    역할:
    - (1) 아주 약한 검수: 입력이 충분히 긴데 결과가 거의 비면 retry
    - (2) 이모지 제거
    - (3) keywords: '명사처럼 보이는 것'만 남기고 나머지 제거 (채우지 않음 / fallback 금지)
    - (4) SHORT MODE 과확장 방지: 원문 대비 과도하게 길면 안전 포맷으로 보정
      - 하위 번호(1-1 등) 금지, '-' 불릿만 사용
    """

    refined_note = (state.get("refined_note", "") or "").strip()
    title = (state.get("title", "제목 없음") or "제목 없음").strip()
    keywords = state.get("keywords", []) or []
    src = (state.get("preprocessed_request") or state.get("user_request", "") or "").strip()
    is_short = bool(state.get("is_short", False))
    category = (state.get("category", "memo") or "memo").strip()

    # -------------------------
    # Helpers
    # -------------------------
    def remove_emoji(text: str) -> str:
        if not text:
            return text
        # wide unicode range emoji 제거(과하게 잡아도 reflect 목적에 부합)
        emoji_pattern = re.compile("[" u"\U00010000-\U0010FFFF" "]+", flags=re.UNICODE)
        return emoji_pattern.sub("", text)

    def filter_noun_keywords(keys: list[Any]) -> list[str]:
        """
        명사처럼 보이지 않는 키워드를 전부 제거.
        - 여기서는 '삭제만' 수행 (없으면 그냥 [])
        - 조사/어미/서술형/활용형/의미없는 일반어 제거
        """
        if not keys:
            return []

        banned_suffixes = (
            # 서술/활용형에 자주 붙는 꼬리(보수적으로 제거)
            "하다", "되다", "있다", "없다",
            "합니다", "했습니다", "같습니다", "됩니다",
            "하기", "하기를", "하기에",
            "하는", "한", "할",
            # 조사/격조사/접속/보조
            "으로", "로", "에서", "에게", "께", "보다",
            "은", "는", "이", "가", "을", "를", "의", "와", "과", "도", "만",
        )

        # 검색 가치가 낮은 일반어(원하면 여기 더 늘리면 됨)
        stopwords = {
            "것", "수", "때", "경우", "부분",
            "내용", "기능", "문제", "방안",
            "정도", "사용", "관련", "제안",
        }

        out: list[str] = []
        seen = set()

        for k in keys:
            if not isinstance(k, str):
                continue
            k = k.strip()
            if len(k) < 2:
                continue

            # 끝이 조사/어미/서술형이면 제거
            if k.endswith(banned_suffixes):
                continue

            # 일반 불용어 제거
            if k in stopwords:
                continue

            # 너무 일반적인 숫자/기호 위주 제거(필요시 강화)
            if re.fullmatch(r"[\d\W_]+", k):
                continue

            if k in seen:
                continue
            seen.add(k)
            out.append(k)

        return out

    def safe_short_format(category_: str, src_: str) -> str:
        """
        SHORT MODE 보정용 최소 포맷 (확장 최소 / 하위 번호 금지)
        - 상위는 1. 한 줄
        - 하위는 '-' 불릿만
        """
        src_line = src_ if src_ else "미기재"

        if category_ == "meeting":
            return (
                "## 회의 개요\n"
                "- 일시: 미기재\n"
                "- 참석자: 미기재\n"
                "- 주제/안건: 미기재\n\n"
                "## 논의 내용\n"
                "1. 입력 기반\n"
                f"- {src_line}\n\n"
                "## 결정사항\n"
                "- 미기재\n\n"
                "## 액션 아이템\n"
                "- 미기재\n"
            )

        if category_ == "planner":
            return (
                "## 할 일\n"
                "1. 입력 기반\n"
                f"- {src_line}\n\n"
                "## 비고\n"
                "- 미기재\n"
            )

        if category_ == "note":
            return (
                "## 핵심\n"
                "1. 입력 기반\n"
                f"- {src_line}\n\n"
                "## 추가로 적으면 좋은 것\n"
                "- 관련 맥락/출처/예시(있다면)\n"
            )

        # memo/default
        return (
            "1. 입력 기반\n"
            f"- {src_line}\n"
        )

    # -------------------------
    # (1) 아주 약한 기존 검수: 입력이 충분히 긴데 결과가 비면 retry
    # -------------------------
    user_req = (state.get("user_request", "") or "")
    if len(user_req) > 20 and len(refined_note) < 5:
        return {"user_decision": "retry"}

    # -------------------------
    # (2) 이모지 제거
    # -------------------------
    refined_note = remove_emoji(refined_note)
    title = remove_emoji(title)

    # -------------------------
    # (3) keywords: 명사만 남기고, 비면 그대로 []
    #     - fallback으로 채우지 않음
    # -------------------------
    keywords = filter_noun_keywords(keywords)

    # -------------------------
    # (4) SHORT MODE 과확장 방지: 결과가 원문 대비 너무 길면 안전 포맷으로 보정
    # -------------------------
    if is_short:
        # 경험칙: short에서 원문 대비 과도하게 길면 확장으로 간주
        # - refined_note가 아주 길어지는 케이스 방어
        if len(refined_note) > max(500, 3 * len(src)):
            refined_note = safe_short_format(category, src)

        # title도 너무 길면 컷
        if len(title) > 60:
            title = title[:60] + "…"

    return {
        "refined_note": refined_note,
        "title": title or "제목 없음",
        "keywords": keywords,  # 비어도 그대로 []
        "user_decision": "approve",
    }


# -----------------------------
# 7) Graph 구성
# -----------------------------
workflow = StateGraph(GraphState)

workflow.add_node("cleaner", cleaner_node)
workflow.add_node("router", router_node)
workflow.add_node("meeting_agent", meeting_node)
workflow.add_node("note_agent", note_node)
workflow.add_node("planner_agent", planner_node)
workflow.add_node("memo_agent", memo_node)
workflow.add_node("reflect", reflect_node)

workflow.set_entry_point("cleaner")
workflow.add_edge("cleaner", "router")

workflow.add_conditional_edges(
    "router",
    lambda state: state["category"],
    {
        "meeting": "meeting_agent",
        "note": "note_agent",
        "planner": "planner_agent",
        "memo": "memo_agent",
    }
)

for node in ["meeting_agent", "note_agent", "planner_agent", "memo_agent"]:
    workflow.add_edge(node, "reflect")

workflow.add_conditional_edges(
    "reflect",
    lambda state: "pass" if state["user_decision"] == "approve" else "retry",
    {
        "pass": END,
        "retry": "router"
    }
)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)
