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
    user_request: str
    category: Literal["meeting", "study", "task", "idea", "info"]
    refined_note: str
    title: str
    user_decision: str


# 2. 노드 함수 정의

def cleaner_node(state: GraphState):
    """전처리 노드: 공백 제거 및 기본적인 텍스트 정제"""
    return {"user_request": state["user_request"].strip()}


def router_node(state: GraphState):
    """
    [테스트 모드] 무조건 'study' 카테고리를 반환하도록 고정합니다.
    """
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    system_prompt = """당신은 메모 분류 전문가입니다. 사용자의 입력을 분석하여 반드시 아래 5개 카테고리 중 하나만 선택하여 단어로 답변하세요.

       - meeting: 회의, 미팅, 면담, 협업 결정사항
       - study: 강의 요약, 학습 내용, 개념 정리, 연구
       - task: 할 일, 마감 기한, 개인적 태스크, 체크리스트
       - idea: 영감, 아이디어 파편, 새로운 기획 구상
       - info: 연락처, 주소, 단순 사실 정보, 보관용 데이터

       출력 형식: 오직 카테고리 단어만 출력 (예: meeting)"""
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

def meeting_node(state: GraphState):
    result = MeetingAgent().organize(state["user_request"])
    return {"title": result.get("title", "제목 없음"), "refined_note": result.get("content", "정리 실패")}


def study_node(state: GraphState):
    result = StudyAgent().organize(state["user_request"])
    return {"title": result.get("title", "제목 없음"), "refined_note": result.get("content", "정리 실패")}


def task_node(state: GraphState):
    result = TaskAgent().organize(state["user_request"])
    return {"title": result.get("title", "제목 없음"), "refined_note": result.get("content", "정리 실패")}


def idea_node(state: GraphState):
    result = IdeaAgent().organize(state["user_request"])
    return {"title": result.get("title", "제목 없음"), "refined_note": result.get("content", "정리 실패")}


def info_node(state: GraphState):
    result = InfoAgent().organize(state["user_request"])
    return {"title": result.get("title", "제목 없음"), "refined_note": result.get("content", "정리 실패")}


# --- 검수 노드 ---
def reflect_node(state: GraphState):
    refined_note = state.get("refined_note", "")

    # 1. 정보 누락 체크
    if len(state["user_request"]) > 20 and len(refined_note) < 5:
        return {"user_decision": "retry"}

    # 2. 이모지 정제
    emoji_pattern = re.compile("[" u"\U00010000-\U0010FFFF" "]+", flags=re.UNICODE)
    if emoji_pattern.search(refined_note):
        cleaned_note = emoji_pattern.sub(r'', refined_note)
        return {"refined_note": cleaned_note, "user_decision": "approve"}

    return {"user_decision": "approve"}


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