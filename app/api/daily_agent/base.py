import json
import re
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.models.structed_output_model import AgentNodeOutputStructure

load_dotenv()


class BaseAgent(ABC):
    def __init__(self, is_short=False):
        model = 'gpt-4o-mini' if is_short else 'gpt-4o'
        self.model = ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.structured_model = self.model.with_structured_output(AgentNodeOutputStructure, include_raw=False)

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    # -------------------------
    # Main
    # -------------------------
    def organize(self, content: str) -> AgentNodeOutputStructure:
        content = (content or "").strip()

        try:
            response = self.structured_model.invoke([
                SystemMessage(content=self.get_system_prompt()),
                HumanMessage(content=content)
            ])

            return response

        except Exception as e:
            print(f"JSON Parsing Error: {e}")
            # 에러 발생 시 최소한의 구조만 맞춰서 반환
            return {
                "title": "파싱 실패(원문 기반)",
                "content": content,
                "keywords": []
            }
