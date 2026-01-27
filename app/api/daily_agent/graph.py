
from typing import TypedDict, Literal
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import re

# 각 파일에서 특화 에이전트 클래스 임포트
from .types.meeting import MeetingAgent
from .types.study import StudyAgent
from .types.task import TaskAgent
from .types.idea import IdeaAgent
from .types.info import InfoAgent


# 1. 그래프 상태 정의
class GraphState(TypedDict):
    user_request: str           # 사용자가 보낸 순수 원본 텍스트
    preprocessed_request: str   # cleaner_node를 통해 정제된 텍스트 [새로 추가]
    category: Literal["meeting", "study", "task", "idea", "info"]
    refined_note: str
    title: str
    keywords: list[str]
    user_decision: str


def cleaner_node(state: GraphState):
    """
    [전처리 노드] LLM을 사용하여 원본의 노이즈를 제거하고 가독성을 높입니다.
    """
    # 전처리 전용 모델 호출 (gpt-4o-mini 활용)
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
   - 예: "때려 박으며" → "기록하며"
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

    cleaned_text = response.content.strip()

    # 전처리된 결과를 상태에 저장하여 다음 노드와 daily.py로 전달
    return {
        "preprocessed_request": cleaned_text
    }

def router_node(state: GraphState):
    """
    [테스트 모드] 무조건 'study' 카테고리를 반환하도록 고정합니다.
    """
    model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)

    system_prompt = """당신은 사용자의 메모를 단 하나의 목적 기준으로 분류하는
                        "메모 라우팅 전문가"입니다.
                        입력된 텍스트의 표현 방식이 아니라,
                        작성자의 주요 의도와 사용 목적을 기준으로
                        아래 5개 카테고리 중 반드시 하나만 선택하세요.

[카테고리 정의]
- meeting: 회의, 미팅, 협업 논의, 의사결정, 합의 사항
- study: 학습 기록, 강의 요약, 개념 정리, 연구 메모
- task: 해야 할 일, 일정, 마감 기한, 개인 작업 목록
- idea: 아직 정리되지 않은 생각, 기획 아이디어, 발상 메모
- info: 판단이나 의견이 없는 단순 사실 정보, 연락처, 주소, 보관용 데이터

[분류 규칙]
- 여러 성격이 섞여 있더라도 가장 지배적인 목적 하나만 선택하세요.
- 문장 수, 길이, 형식이 아닌 내용의 성격을 기준으로 판단하세요.
- 애매한 경우:
  - 학습/개념 설명 → study
  - 행동이나 실행이 포함 → task
  - 생각의 나열, 가능성 탐색 → idea

[출력 규칙 — 매우 중요]
- 반드시 아래 다섯 단어 중 하나만 출력하세요.
- 다른 단어, 설명, 문장, 공백, 기호를 절대 포함하지 마세요.

meeting
study
task
idea
info"""
    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["user_request"])
    ])
    category = response.content.strip().lower()

    # --- 테스트용 고정 값 ---
    category = "study"
    print(f"\n[DEBUG] Router: Forced to '{category}' for testing purposes.")

    return {"category": category}


# --- 5개 특화 에이전트 노드 ---

# --- 5개 특화 에이전트 노드 수정 ---

def meeting_node(state: GraphState):
    # 전처리된 텍스트(preprocessed_request)를 사용하도록 변경
    result = MeetingAgent().organize(state.get("preprocessed_request", state["user_request"]))
    return {
        "title": result.get("title", "제목 없음"),
        "refined_note": result.get("content", "정리 실패"),
        "keywords": result.get("keywords", []) # 키워드 추가
    }


def study_node(state: GraphState):
    result = StudyAgent().organize(state.get("preprocessed_request", state["user_request"]))
    return {
        "title": result.get("title", "제목 없음"),
        "refined_note": result.get("content", "정리 실패"),
        "keywords": result.get("keywords", []) # 이미 잘 들어있지만 전처리 데이터 활용으로 보강
    }


def task_node(state: GraphState):
    result = TaskAgent().organize(state.get("preprocessed_request", state["user_request"]))
    return {
        "title": result.get("title", "제목 없음"),
        "refined_note": result.get("content", "정리 실패"),
        "keywords": result.get("keywords", []) # 키워드 추가
    }


def idea_node(state: GraphState):
    result = IdeaAgent().organize(state.get("preprocessed_request", state["user_request"]))
    return {
        "title": result.get("title", "제목 없음"),
        "refined_note": result.get("content", "정리 실패"),
        "keywords": result.get("keywords", []) # 키워드 추가
    }


def info_node(state: GraphState):
    result = InfoAgent().organize(state.get("preprocessed_request", state["user_request"]))
    return {
        "title": result.get("title", "제목 없음"),
        "refined_note": result.get("content", "정리 실패"),
        "keywords": result.get("keywords", []) # 키워드 추가
    }

# --- 검수 노드 ---
def reflect_node(state: GraphState):
    refined_note = state.get("refined_note", "")
    title = state.get("title", "제목 없음") # 이전 노드에서 생성한 제목 가져오기
    keywords = state.get("keywords", [])    # 이전 노드에서 생성한 키워드 가져오기

    # 1. 검수 로직 (기존 유지)
    if len(state["user_request"]) > 20 and len(refined_note) < 5:
        return {"user_decision": "retry"}

    # 2. 이모지 정제
    emoji_pattern = re.compile("[" u"\U00010000-\U0010FFFF" "]+", flags=re.UNICODE)
    if emoji_pattern.search(refined_note):
        refined_note = emoji_pattern.sub(r'', refined_note)

    # 3. 데이터 유지 및 승인 반환
    # 여기서 title과 keywords를 다시 return 해주어야 상태가 끝까지 유지됩니다.
    return {
        "refined_note": refined_note,
        "title": title,
        "keywords": keywords,
        "user_decision": "approve"
    }

# 3. 그래프 구성 및 엣지 연결

workflow = StateGraph(GraphState)

# 노드 등록
workflow.add_node("cleaner", cleaner_node)
workflow.add_node("router", router_node)
workflow.add_node("meeting_agent", meeting_node)
workflow.add_node("study_agent", study_node)
workflow.add_node("task_agent", task_node)
workflow.add_node("idea_agent", idea_node)
workflow.add_node("info_agent", info_node)
workflow.add_node("reflect", reflect_node)

# 연결 구성
workflow.set_entry_point("cleaner")
workflow.add_edge("cleaner", "router")

# 4. 조건부 라우팅
workflow.add_conditional_edges(
    "router",
    lambda state: state["category"],
    {
        "meeting": "meeting_agent",
        "study": "study_agent",
        "task": "task_agent",
        "idea": "idea_agent",
        "info": "info_agent"
    }
)

# 모든 에이전트 결과는 검수 노드로
for node in ["meeting_agent", "study_agent", "task_agent", "idea_agent", "info_agent"]:
    workflow.add_edge(node, "reflect")

# 검수 결과에 따른 분기
workflow.add_conditional_edges(
    "reflect",
    lambda state: "pass" if state["user_decision"] == "approve" else "retry",
    {
        "pass": END,
        "retry": "router"
    }
)

# 5. 컴파일
memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)