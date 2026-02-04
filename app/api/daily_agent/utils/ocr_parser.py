import base64
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.core.config import settings  # .env 관리용 settings


class OCRParser:
    def __init__(self, model_name: str = "gpt-4o"):
        # cleaner_node의 방식처럼 settings의 API 키를 사용하여 모델 초기화
        self.model = ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def extract_text(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            return "이미지 파일 없음"

        base64_image = self._encode_image(image_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": "이미지 내 텍스트를 추출해 주세요."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        )

        try:
            response = self.model.invoke([message])
            return response.content.strip()
        except Exception as e:
            return f"OCR 오류: {e}"