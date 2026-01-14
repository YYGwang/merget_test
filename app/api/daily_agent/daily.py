import os

import operator
import re
from datetime import datetime, time, timezone, timedelta
from typing import Annotated, List, TypedDict, Literal

from app.core.security import verify_cognito_token
from app.core.database import get_table
from fastapi import APIRouter, HTTPException, Query, Depends, Path
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from boto3.dynamodb.conditions import Key

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()
router = APIRouter()
CONFIRM_TABLE = get_table("confirm_note")



# --- [데이터 모델 정의] ---

class MemoRequest(BaseModel):
    memo: str

    @field_validator('memo', mode='before')
    @classmethod
    def handle_swagger_newlines(cls, v: any) -> str:
        if isinstance(v, str):
            # 제어 문자 제거 및 실제 줄바꿈을 문자로 치환하여 JSON 파싱 보호
            v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
            return v.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()
        return v

class FinalReport(BaseModel):
    # record_date: str = Field(description="작성 날짜 (YYYY-MM-DD)")
    title: str = Field(description="내용을 관통하는 핵심 제목")
    # category: Literal["계획", "수업자료", "연구 노트", "회의 스크립트", "일반 메모"] = Field(description="메모의 유형")
    refined_text: str = Field(description="유형별 템플릿에 맞춰 문장으로 복원한 본문")
    # todo_list: List[str] = Field(description="추출된 핵심 포인트 및 할 일")

class AgentState(TypedDict):
    user_request: str
    research_plan: str
    reflection_feedback: str
    refined_note: FinalReport
    iteration_count: Annotated[int, operator.add]

# --- [로컬 테스트용 프롬프트 이식] ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
structured_llm = llm.with_structured_output(FinalReport)

PROMPTS = {
    'system_persona': """너는 파편화된 모든 종류의 데이터를 분석하여 논리적인 리포트로 변환하는 '범용 지식 정리 전문가'야.
1. 입력된 텍스트의 주제를 최우선으로 파악한다.
2. 단어 나열식, 불완전한 문장들을 문맥에 맞는 자연스러운 서술형 문장으로 복원한다.
3. 정보의 누락 없이 가독성 있는 섹션으로 구분하여 정리한다.""",
    'task_planning': "메모를 분석하여 핵심 주제를 식별하고 각 항목을 논리적으로 그룹화하여 문장으로 복원할 계획을 세워줘.",
    'reflect': "작성된 계획이 원본 메모의 모든 정보를 포함하는지 확인해. 충분하면 '충분함'을 답해줘."
}

# --- [그래프 노드 구현] ---

def planning_node(state: AgentState):
    print(f"[*] Planning... (시도: {state.get('iteration_count', 0) + 1}/5)")
    prompt = f"{PROMPTS['task_planning']}\n\n[입력 메모]\n{state['user_request']}"
    response = llm.invoke([SystemMessage(content=PROMPTS['system_persona']), HumanMessage(content=prompt)])
    return {"research_plan": response.content}

def reflect_node(state: AgentState):
    if state.get("iteration_count", 0) >= 3: return {"reflection_feedback": "충분함"}
    prompt = f"{PROMPTS['reflect']}\n\n계획: {state['research_plan']}\n원본: {state['user_request']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"reflection_feedback": response.content}

def generate_report_node(state: AgentState):
    print("[*] Generate... 리포트 문장 복원 및 구조화 중")
    # curr_date = datetime.now().strftime('%Y-%m-%d')
    result = structured_llm.invoke([
        # SystemMessage(content=f"{PROMPTS['system_persona']}\n오늘 날짜: {curr_date}"),
        HumanMessage(content=state['user_request'])
    ])
    return {"refined_note": result, "iteration_count": 1}

def decide_next_step(state: AgentState):
    if "충분함" in state["reflection_feedback"] or state["iteration_count"] >= 3: return "generate"
    return "retry"

workflow = StateGraph(AgentState)
workflow.add_node("planning", planning_node)
workflow.add_node("reflect", reflect_node)
workflow.add_node("generate", generate_report_node)
workflow.set_entry_point("planning")
workflow.add_edge("planning", "reflect")
workflow.add_conditional_edges("reflect", decide_next_step, {"generate": "generate", "retry": "planning"})
workflow.set_entry_point('planning')
workflow.add_edge("generate", END)
app_graph = workflow.compile()


def test_agent(raw_input: str):
    print("\n" + "=" * 60)
    print("🚀 LangGraph 정리 에이전트 테스트 시작")
    print("=" * 60)

    # 특수 문자 제거 전처리
    clean_input = re.sub(r'[\x00-\x1F\x7F]', '', raw_input).strip()

    config = {"configurable": {"thread_id": "test_1"}}

    try:
        final_state = app_graph.invoke({
            "user_request": clean_input,
            "category": "",
            "research_plan": "",
            "reflection_feedback": "",
            "iteration_count": 0
        }, config)

        report = final_state["refined_note"]
        print("\n" + "✨ 최종 정리 리포트 " + "✨")
        print(f"📌 제목: {report.title}")
        print(f"📅 날짜: {report.record_date}")
        print(f"📁 유형: {report.category}")
        print("-" * 60)
        print(f"📝 본문 내용:\n{report.refined_text}")
        print("-" * 60)
        print(f"✅ 할 일 리스트: {', '.join(report.todo_list)}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")





# 테이블 정의
DRAFT_TABLE = get_table('draft_note')
CONFIRM_TABLE = get_table('confirm_note')


@router.post("/generate-report/{creation_date}")
async def create_report(
        creation_date: int = Path(..., description="원본 메모를 찾기 위한 고유 타임스탬프 값"),
        uid: str = Depends(verify_cognito_token)
):
    try:
        # 1. draft_table에서 원본 데이터를 조회
        response = DRAFT_TABLE.get_item(
            Key={
                "user_key": uid,
                "creation_date": creation_date
            }
        )

        item = response.get('Item') # Item
        if not item:
            raise HTTPException(status_code=404, detail="원본 메모를 찾을 수 없습니다.")

        # 원본 테이블의 'content' 필드에서 텍스트를 가져옵니다.
        raw_content = item.get('content')
        if not raw_content:
            raise HTTPException(status_code=400, detail="메모 내용이 비어있어 분석할 수 없습니다.")

        # 2. AI 에이전트(LangGraph)에게 분석 요청
        final_state = app_graph.invoke({
            "user_request": raw_content,
            "iteration_count": 0,
            "research_plan": "",
            "reflection_feedback": ""
        })

        refined_note = final_state["refined_note"]

        # 3. 분석된 결과를 confirm_table에 저장
        CONFIRM_TABLE.put_item(Item={
            "user_key": uid,
            "creation_date": creation_date,
            "title": refined_note.title,
            # "category": refined_note.category,
            "content": refined_note.refined_text  # AI가 정제한 본문
        })

        # 4. JSON 형식으로 클라이언트에게 결과 반환 (FastAPI가 자동 변환)
        return {
                "creation_date": creation_date,
                "title": refined_note.title,
                "content": refined_note.refined_text
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="리포트 생성 및 저장 실패")