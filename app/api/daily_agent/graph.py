from typing import TypedDict, Literal
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import re
# from langchain_community.document_loaders import PDFPlumberLoader

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
- 정제된 텍스트 본문만 출력하세요.
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

def reflect_node(state: GraphState):
    refined_note = state.get("refined_note", "") or ""
    title = state.get("title", "제목 없음") or "제목 없음"
    keywords = state.get("keywords", []) or []
    src = state.get("preprocessed_request") or state.get("user_request", "")
    is_short = bool(state.get("is_short", False))
    category = state.get("category", "memo")

    # (1) 아주 약한 기존 검수: 입력이 충분히 긴데 결과가 비면 재시도
    if len(state.get("user_request", "")) > 20 and len(refined_note.strip()) < 5:
        return {"user_decision": "retry"}

    # (2) 이모지 제거
    emoji_pattern = re.compile("[" u"\U00010000-\U0010FFFF" "]+", flags=re.UNICODE)
    refined_note = emoji_pattern.sub(r'', refined_note)

    # (3) 키워드 비면 보강
    if not keywords:
        keywords = _fallback_keywords(src, max_k=8)

    # (4) SHORT MODE 과확장 방지: 결과가 원문 대비 너무 길면 안전 포맷으로 축약
    # - retry 대신 '보정' 선택 (안정성 목적)
    if is_short:
        # 경험치: short에서는 원문 대비 3배 이상이 나오면 확장 가능성이 큼
        if len(refined_note) > max(500, 3 * len(src)):
            if category == "meeting":
                refined_note = (
                    "## 회의 개요\n"
                    "- 일시: 미기재\n"
                    "- 참석자: 미기재\n"
                    "- 주제/안건: 미기재\n\n"
                    "## 논의 내용\n"
                    f"1. 입력 기반\n  1-1. {src}\n\n"
                    "## 결정사항\n"
                    "1. 미기재\n\n"
                    "## 액션 아이템\n"
                    "1. 미기재\n"
                )
            elif category == "planner":
                refined_note = (
                    "## Planner\n"
                    "1. 할 일 목록\n"
                    f"  1-1. {src}\n"
                    "    - 기한: 미기재\n"
                    "    - 우선순위: 미기재\n\n"
                    "2. 비고\n"
                    "1. 미기재\n"
                )
            elif category == "note":
                refined_note = (
                    "## 핵심\n"
                    f"1. {src}\n\n"
                    "## 상세 노트\n"
                    f"1. 입력 기반\n  1-1. {src}\n\n"
                    "## 추가로 적으면 좋은 것\n"
                    "1. 관련 맥락/출처/예시(있다면)\n"
                )
            else:
                refined_note = f"1. 입력 기반\n  1-1. {src}\n"

            # title도 너무 길면 짧게
            if len(title) > 60:
                title = title[:60] + "…"

    return {
        "refined_note": refined_note,
        "title": title,
        "keywords": keywords,
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
