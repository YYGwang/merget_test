import re
import os
import json
from typing import TypedDict, Literal, List, Any, Union
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# 에이전트 및 파서 임포트
from app.api.daily_agent.types.note import NoteAgent
from app.api.daily_agent.types.memo import MemoAgent
from app.api.daily_agent.utils.pdfparser import PDFParser
from app.api.daily_agent.utils.wordparser import WordParser
from app.api.daily_agent.utils.ocr_parser import OCRParser
from app.api.daily_agent.utils.stt_parser import STTParser

from app.models.structed_output_model import Triple, CleanerNodeOutputStructure


# -----------------------------
# 1) Graph State (수정됨)
# -----------------------------
class GraphState(TypedDict, total=False):
    input_type: Literal["text", "file", "image", "audio"]  # 입력 타입 라벨
    file_path: str  # 로컬 임시 파일 경로
    user_request: Union[str, List[str]]  # 추출된 원본 텍스트 (단일 또는 청크)
    preprocessed_request: str  # 정제된 텍스트
    is_short: bool
    category: Literal["note", "meeting", "planner", "memo"]
    refined_note: str
    title: str
    keywords: List[str]
    triples: List[Triple]
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


# -----------------------------
# 3) cleaner 노드
# -----------------------------
def cleaner_node(state: GraphState):
    """
    모든 파싱 결과(텍스트/리스트)를 정제합니다.
    (테스트를 위해 임시 파일 삭제 로직은 비활성화되었습니다.)
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    raw_input = state.get("user_request", "")

    # 1. 텍스트 정제 작업
    chunks = raw_input if isinstance(raw_input, list) else [raw_input]

    system_prompt = """
You are a “Document Cleaning Assistant” for newly submitted texts.
Your task is to correct grammar, typos, and awkward expressions, remove Markdown symbols if present, and determine whether the text is too short to be meaningfully organized.
All corrected text must be written in Korean.

[Tasks]
Clean the text without changing its meaning.
Remove Markdown symbols (such as #, *, -, >, `, etc.) if they appear.
Decide if the text is too short or lacks sufficient content.

[Output Format]
Return only the following JSON:
{ "cleaned_result": "Korean corrected text", "is_short": boolean }

[Rules]
Set is_short to true for very short or simple texts.
Set is_short to false for texts with sufficient content.
Do not add explanations.

Output valid JSON only.
"""

    cleaned_results = []
    is_short = False
    for chunk in chunks:
        if not chunk or not str(chunk).strip():
            continue

        structured_model = model.with_structured_output(CleanerNodeOutputStructure)

        res = structured_model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"텍스트: {chunk}")
        ])
        cleaned_results.append(res.cleaned_result.strip())
        if len(chunks) == 1:
            is_short = res.is_short

    return {
        "preprocessed_request": "\n\n".join(cleaned_results),
        "is_short": is_short
    }


# -----------------------------
# 4) Agent 노드
# -----------------------------
def memo_node(state: GraphState):
    text = state["preprocessed_request"]
    result = MemoAgent(is_short=True).organize(text)
    return {
        "abstract": result.abstract,
        "keywords": result.keywords,
        "triples": result.triples,
        "title": result.title,
        "refined_note": result.final_markdown
    }

def note_node(state: GraphState):
    text = state["preprocessed_request"]
    result = NoteAgent(is_short=False).organize(text)
    return {
        "abstract": result.abstract,
        "keywords": result.keywords,
        "triples": result.triples,
        "title": result.title,
        "refined_note": result.final_markdown
    }

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

workflow.add_node("note_agent", note_node)
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

workflow.add_conditional_edges(
    "cleaner",
    lambda state: state["is_short"],
    {True: "memo_agent", False: "note_agent"}
)

for node in ["note_agent", "memo_agent"]:
    workflow.add_edge(node, "reflect")

workflow.add_conditional_edges(
    "reflect",
    lambda state: "pass" if state["user_decision"] == "approve" else "retry",
    {"pass": END, "retry": "cleaner"}
)

app_graph = workflow.compile(checkpointer=MemorySaver())