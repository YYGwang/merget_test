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

import boto3
from urllib.parse import urlparse


# -----------------------------
# 1) Graph State (수정됨)
# -----------------------------
class GraphState(TypedDict, total=False):
    input_type: Literal["text", "file", "image", "audio"]  # 입력 타입 라벨
    content: str  # text or s3_url
    file_path: str
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
# ---------------------------

s3_client = boto3.client('s3')


def download_from_s3(s3_url: str) -> str:
    """S3 URL을 해석하여 로컬 /tmp에 다운로드 후 경로 반환"""
    parsed = urlparse(s3_url)
    bucket = parsed.netloc.split('.')[0]
    key = parsed.path.lstrip('/')
    local_path = os.path.join("/tmp", os.path.basename(key))

    s3_client.download_file(bucket, key, local_path)
    return local_path


# --- GraphState에 file_path가 이미 정의되어 있다고 가정 ---

def file_parsing_node(state: GraphState):
    """PDF 및 Word 파일 통합 파싱"""
    s3_url = state.get("content")  # 프론트에서 준 S3 URL
    if not s3_url: return {"user_request": "S3 URL 없음"}

    # 1. S3에서 로컬 /tmp로 다운로드
    local_path = download_from_s3(s3_url)

    # 2. 다운로드된 로컬 경로의 확장자 확인
    ext = local_path.split('.')[-1].lower()

    # 3. 로컬 경로를 파서에 전달 (기존 파서 로직 그대로 사용)
    if ext == 'pdf':
        result = PDFParser().extract_and_chunk(local_path)
    elif ext in ['doc', 'docx']:
        result = WordParser().extract_text(local_path)
    else:
        result = "지원하지 않는 파일 형식"

    # user_request와 함께 생성된 local_path를 반환 (나중에 삭제하기 위함)
    return {"user_request": result, "file_path": local_path}


async def ocr_parsing_node(state: GraphState) -> dict:
    """이미지 노드: OCR 실행"""
    s3_url = state.get("content")
    if not s3_url: return {"user_request": ""}

    # S3 다운로드
    local_path = download_from_s3(s3_url)

    parser = OCRParser()
    # 단일 파일이므로 리스트로 감싸서 전달
    ocr_results = await parser.extract_texts([local_path])

    if not ocr_results:
        return {"user_request": "", "file_path": local_path}

    combined_text = "\n\n".join(
        item["text"].strip() for item in ocr_results if item["text"].strip()
    )

    return {"user_request": combined_text, "file_path": local_path}


def stt_parsing_node(state: GraphState):
    """음성 노드: STT 실행"""
    s3_url = state.get("content")
    local_path = download_from_s3(s3_url)

    parser = STTParser()
    result = parser.transcribe(local_path)  # 로컬 경로 전달
    return {"user_request": result, "file_path": local_path}

# async def file_parsing_node(state: GraphState):
#     """PDF / Word 파일 통합 파싱 (손글씨 PDF 포함)"""
#     path = state.get("file_path")
#     if not path:
#         return {"user_request": "파일 경로 없음"}
#
#     ext = path.split('.')[-1].lower()
#
#     if ext == "pdf":
#         pdf_parser = PDFParser()
#
#         # 1️⃣ 텍스트 우선 시도
#         text_chunks = pdf_parser.extract_and_chunk(path)
#
#         if text_chunks:
#             # ✅ 일반 텍스트 PDF 또는 혼합 PDF (텍스트 존재)
#             return {"user_request": text_chunks}
#
#         # 2️⃣ 텍스트가 전혀 없으면 → 손글씨-only PDF
#         # PDF를 이미지 묶음으로 취급
#         pages = pdf_parser.extract_text_and_images(path)
#
#         image_paths = []
#         for page in pages:
#             image_paths.extend(page["images"])
#
#         if not image_paths:
#             return {"user_request": "PDF에서 추출 가능한 내용 없음"}
#
#         # OCR은 여기서 직접 하지 않고
#         # state를 바꿔서 OCR 노드로 넘긴다
#         return {
#             "input_type": "image",
#             "file_path": image_paths,
#             "user_request": ""
#         }
#
#     elif ext in ["doc", "docx"]:
#         return {"user_request": WordParser().extract_text(path)}
#
#     return {"user_request": "지원하지 않는 파일 형식"}



# def ocr_parsing_node(state: GraphState):
#     """이미지 노드: OCR 실행"""
#     path = state.get("file_path")
#     parser = OCRParser()
#     result = parser.extract_text(path)
#     return {"user_request": result}


# -----------------------------
# 3) cleaner 노드
# -----------------------------
def cleaner_node(state: GraphState):
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
workflow.add_node(
    "text_input",
    lambda state: {"user_request": state["content"]}
)


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