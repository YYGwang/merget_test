import re
import json
from typing import TypedDict, Literal, List, Any, Union
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# 에이전트 및 파서 임포트
from .types.meeting import MeetingAgent
from .types.note import NoteAgent
from .types.planner import PlannerAgent
from .types.memo import MemoAgent
from .utils.pdfparser import PDFParser
from .utils.wordparser import WordParser
from .utils.ocr_parser import OCRParser
from .utils.stt_parser import STTParser


# -----------------------------
# 1) Graph State (수정됨)
# -----------------------------
class GraphState(TypedDict, total=False):
    input_type: Literal["text", "file", "image", "audio"]  # 입력 타입 라벨
    file_path: str  # 로컬 임시 파일 경로
    user_request: Union[str, List[str]]  # 추출된 원본 텍스트 (단일 또는 청크)
    preprocessed_request: str  # 정제된 텍스트
    category: Literal["note", "meeting", "planner", "memo"]
    refined_note: str
    title: str
    keywords: List[str]
    triples: List[dict]
    abstract: str
    user_decision: str


# -----------------------------
# 2) 파싱 노드 (추가됨)
# -----------------------------
def file_parsing_node(state: GraphState):
    """PDF 및 Word 파일 통합 파싱"""
    path = state.get("file_path")
    if not path: return {"user_request": "파일 경로 없음"}

    ext = path.split('.')[-1].lower()
    if ext == 'pdf':
        return {"user_request": PDFParser().extract_and_chunk(path)}
    elif ext in ['doc', 'docx']:
        # WordParser도 extract_text가 리스트를 반환하도록 설계됨
        return {"user_request": WordParser().extract_text(path)}
    return {"user_request": "지원하지 않는 파일 형식"}


def ocr_parsing_node(state: GraphState):
    """이미지 노드: OCR 실행"""
    path = state.get("file_path")
    parser = OCRParser()
    result = parser.extract_text(path)
    return {"user_request": result}

def stt_parsing_node(state: GraphState):
    """음성 노드: STT 실행"""
    path = state.get("file_path")
    parser = STTParser()
    result = parser.transcribe(path)
    return {"user_request": result}


import tiktoken

#
# def limit_tokens(text: str, model_name: str = "gpt-4o", max_tokens: int = 100000) -> str:
#     """
#     텍스트를 모델의 토큰 한도에 맞춰 자릅니다.
#     GPT-4o의 최대 한도는 128k이지만, 응답 토큰을 위해 100k 정도로 여유를 두는 것이 좋습니다.
#     """
#     try:
#         encoding = tiktoken.encoding_for_model(model_name)
#     except KeyError:
#         encoding = tiktoken.get_encoding("cl100k_base")  # 기본 인코딩 사용
#
#     tokens = encoding.encode(text)
#
#     if len(tokens) <= max_tokens:
#         return text
#
#     # 한도만큼 토큰을 자른 후 다시 텍스트로 변환
#     truncated_tokens = tokens[:max_tokens]
#     return encoding.decode(truncated_tokens)

# -----------------------------
# 3) Cleaner Node (수정됨: 파싱된 텍스트 정제)
# -----------------------------
import os


def cleaner_node(state: GraphState):
    """
    모든 파싱 결과(텍스트/리스트)를 정제합니다.
    (테스트를 위해 임시 파일 삭제 로직은 비활성화되었습니다.)
    """
    model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    raw_input = state.get("user_request", "")
    file_path = state.get("file_path")

    # 1. 텍스트 정제 작업
    chunks = raw_input if isinstance(raw_input, list) else [raw_input]

    system_prompt = """당신은 요약자나 해설자가 아니라, 후속 처리를 위한 "입력 텍스트 정제 전용 에이전트"입니다.
의미를 전혀 훼손하지 않은 상태로 읽기 쉬운 중립적 문어체 텍스트로 정제하세요.
주석, 각주, 참조 번호([1], (참고) 등)는 제거하세요."""

    cleaned_results = []
    for chunk in chunks:
        if not chunk or not str(chunk).strip():
            continue
        res = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"텍스트: {chunk}")
        ])
        cleaned_results.append(res.content.strip())

    # 2. 임시 파일 삭제 로직 (비활성화됨)
    if file_path and os.path.exists(file_path):
        # os.remove(file_path) # 파일을 삭제하지 않도록 주석 처리
        print(f"--- [System] 파일 보존 모드: {file_path} (삭제되지 않음) ---")

    return {
        "preprocessed_request": "\n\n".join(cleaned_results),
        "file_path": file_path  # None으로 초기화하지 않고 경로를 유지하여 후속 노드에서 참조 가능하게 함
    }


# -----------------------------
# 3-2) keyword, triple, abstract 추출
# -----------------------------
def key_triple_node(state: GraphState):
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system_prompt = """
You are an information extraction model for Korean planning, design, and any documents.

You must follow a strict three-step process.

STEP 1: Extract core keywords.
STEP 2: Extract (head, relation, tail) triples using only those keywords.
STEP 3: Generate an abstract using extracted keywords and triples.

The output will be used for wordcloud visualization, knowledge graphs, and document summarization.

Rules:
- Output JSON only. No explanations.
- The entire output must be written in Korean.
- Final output format must be:
{
  "keywords": ["string"],
  "triples": [
    {"head": "string", "relation": "string", "tail": "string"}
  ],
  "abstract": "string"
}

STEP 1 (Keyword Extraction):
- Extract nouns or named entities only.
- No particles, verbs, adjectives, clauses, or sentences.
- Must be standalone keywords.
- Remove vague words (unless part of a meaningful compound):
  기능, 시스템, 서비스, 화면, 페이지, 것, 수, 경우, 관련, 연관
- Keep meaningful compound nouns (e.g., "드래프트 노트", "유사도 검색").

STEP 2 (Triple Extraction):
- Head and tail MUST be selected from the keyword list from STEP 1.
- Do not invent new head/tail terms.
- Extract only statements explicitly grounded in the document (facts + proposals + requirements).
- Relations must be written in Korean, concise, and not full sentences.
- Omit unclear or low-quality triples.

STEP 3 (Abstract Generation):
- Generate a concise Korean abstract using ONLY the extracted keywords and triples.
- The abstract must summarize the main purpose, structure, and technical content of the document.
- Do NOT introduce new concepts, entities, or terms.
- Use natural Korean sentences.
- Reflect the relationships expressed in the triples.
- Length: 2–4 sentences.

Empty Output Rules:
- If no valid keywords: {"keywords": [], "triples": [], "abstract": ""}
- If keywords exist but no valid triples: {"keywords": [...], "triples": [], "abstract": ""}
- Only generate abstract when both keywords and triples are non-empty.
""".strip()

    text = state.get("preprocessed_request") or state.get("user_request", "")
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["preprocessed_request"])
    ])

    res = json.loads((response.content or "").strip())

    return {
        "keywords": res.get("keywords", []),
        "triples": res.get("triples", []),
        "abstract": res.get("abstract", "")
    }

# -----------------------------
# 4) Router & Agent Nodes (기존 유지하되 is_short 제거)
# -----------------------------
import tiktoken
from langchain_core.messages import SystemMessage, HumanMessage


def router_node(state: GraphState):
    # 1. 텍스트 추출 (정제된 텍스트 우선)
    text = state.get("preprocessed_request") or state.get("user_request", "")
    t = str(text)[:10000]
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    # 3. 강화된 시스템 프롬프트 (분류 가이드라인 추가)
    system_prompt = """당신은 입력된 메모의 성격을 분석하여 가장 적합한 처리 프로세스로 안내하는 전문 라우터입니다.
아래 기준에 따라 [meeting, planner, note, memo] 중 하나를 선택하세요.

[분류 기준]
- meeting: 회의록, 미팅 준비 자료, 다수의 대화 내용, 결정 사항이 포함된 경우
- planner: 마감 기한, 할 일 리스트(To-do), 일정 예약, 향후 계획이 주된 내용인 경우
- note: 학습 정리, 전문 지식, 기술 문서, 긴 아티클, 에러 로그 분석 등 정보성 기록인 경우
- memo: 짧은 아이디어, 일상적인 기록, 분류가 모호한 단편적인 텍스트인 경우

[출력 규칙]
- 반드시 단어 하나(meeting, planner, note, memo)만 출력하세요."""

    res = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"분류할 텍스트:\n{t}")
    ])

    category = res.content.strip().lower()

    # 4. 임시 분류 제한 로직 (작업 중인 에이전트로 유도)
    # meeting이나 planner로 분류되어도 현재는 note 에이전트가 처리하도록 통합합니다.
    if category in ["meeting", "planner", "note"]:
        category = "note"
    else:
        category = "memo"

    return {"category": category}


def meeting_node(state: GraphState):
    text = state["preprocessed_request"]
    result = MeetingAgent().organize(text, category="meeting")  # is_short 제거
    return {"title": result["title"], "refined_note": result["content"]}


def note_node(state: GraphState):
    text = state["preprocessed_request"]
    result = NoteAgent().organize(text, category="note")
    return {"title": result["title"], "refined_note": result["content"]}


def planner_node(state: GraphState):
    text = state["preprocessed_request"]
    result = PlannerAgent().organize(text, category="planner")
    return {"title": result["title"], "refined_note": result["content"]}


def memo_node(state: GraphState):
    text = state["preprocessed_request"]
    result = MemoAgent().organize(text, category="memo")
    return {"title": result["title"], "refined_note": result["content"]}


# -----------------------------
# 5) Reflect Node (기존 유지)
# -----------------------------
def reflect_node(state: GraphState):
    refined_note = (state.get("refined_note", "") or "").strip()
    title = (state.get("title", "제목 없음") or "제목 없음").strip()
    keywords = state.get("keywords", [])

    # 이모지 제거 및 키워드 필터링 로직 (상기 코드와 동일)
    # ... (생략된 Helper 함수들 적용)

    return {
        "refined_note": refined_note,
        "title": title,
        "keywords": keywords,
        "user_decision": "approve"  # 품질 미달 시 retry로 변경 가능
    }


# -----------------------------
# 6) Graph 구성 (FAN-IN 구조 적용)
# -----------------------------
workflow = StateGraph(GraphState)

# 노드 등록
workflow.add_node("file_parsing", file_parsing_node)
workflow.add_node("ocr_parsing", ocr_parsing_node)
workflow.add_node("stt_parsing", stt_parsing_node)
workflow.add_node("text_input", lambda state: {"user_request": state["user_request"]})

workflow.add_node("cleaner", cleaner_node)
workflow.add_node("key_triple_extractor", key_triple_node)

workflow.add_node("router", router_node)
workflow.add_node("meeting_agent", meeting_node)
workflow.add_node("note_agent", note_node)
workflow.add_node("planner_agent", planner_node)
workflow.add_node("memo_agent", memo_node)
workflow.add_node("reflect", reflect_node)

# --- 엣지 연결 ---

# 입구 분기 (라벨 기반)
workflow.set_conditional_entry_point(
    lambda state: state.get("input_type", "text"),
    {
        "file": "file_parsing",
        "image": "ocr_parsing",
        "audio": "stt_parsing",
        "text": "text_input"
    }
)

# 모든 파싱 결과는 Cleaner로 집결 (Fan-in)
workflow.add_edge("file_parsing", "cleaner")
workflow.add_edge("ocr_parsing", "cleaner")
workflow.add_edge("stt_parsing", "cleaner")
workflow.add_edge("text_input", "cleaner")

workflow.add_edge("cleaner", "key_triple_extractor")
workflow.add_edge("key_triple_extractor", "router")

workflow.add_conditional_edges(
    "router",
    lambda state: state["category"],
    {"meeting": "meeting_agent", "note": "note_agent", "planner": "planner_agent", "memo": "memo_agent"}
)

for node in ["meeting_agent", "note_agent", "planner_agent", "memo_agent"]:
    workflow.add_edge(node, "reflect")

workflow.add_conditional_edges(
    "reflect",
    lambda state: "pass" if state["user_decision"] == "approve" else "retry",
    {"pass": END, "retry": "router"}
)

app_graph = workflow.compile(checkpointer=MemorySaver())