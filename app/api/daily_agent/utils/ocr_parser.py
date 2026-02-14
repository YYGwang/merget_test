import os
import base64
import asyncio
from typing import List, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.core.config import settings


class OCRParser:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = ChatOpenAI(
            model=model_name,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )

    def _encode_image(self, image_path: str) -> str:
        """이미지 파일을 base64 문자열로 인코딩 (동기 함수)"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_message(self, base64_image: str) -> HumanMessage:
        """OCR 요청용 HumanMessage 생성"""
        return HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "이미지에 보이는 텍스트를 그대로 추출하세요. "
                        "해석하거나 요약하지 마세요."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        )

    async def _ocr_one(self, image_path: str) -> dict:
        """
        단일 이미지 OCR 처리
        - base64 인코딩과 모델 호출을 thread로 분리
        - 실패/빈 결과도 메타데이터와 함께 반환
        """
        if not os.path.exists(image_path):
            return {
                "path": image_path,
                "text": "",
                "error": "file_not_found"
            }

        try:
            # 1️⃣ base64 인코딩을 thread로 분리 (이벤트 루프 보호)
            base64_image = await asyncio.to_thread(
                self._encode_image,
                image_path
            )

            message = self._build_message(base64_image)

            # 2️⃣ 모델 호출도 thread로 분리
            response = await asyncio.to_thread(
                self.model.invoke,
                [message]
            )

            text = response.content.strip() if response.content else ""

            if not text:
                return {
                    "path": image_path,
                    "text": "",
                    "error": "empty_ocr_result"
                }

            return {
                "path": image_path,
                "text": text,
                "error": None
            }

        except Exception as e:
            return {
                "path": image_path,
                "text": "",
                "error": f"ocr_exception: {e}"
            }

    async def extract_texts(
        self,
        image_paths: Union[str, List[str]]
    ) -> List[dict]:
        """
        입력:
          - image_paths: str | List[str]

        출력 (항상 리스트, 순서 보존):
          [
            {
              "order": 1,
              "path": "...",
              "text": "...",
              "error": None | str
            },
            ...
          ]
        """

        # 1️⃣ 입력을 무조건 리스트로 통일
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # 2️⃣ 병렬 OCR 실행
        tasks = [
            self._ocr_one(path)
            for path in image_paths
        ]

        raw_results = await asyncio.gather(*tasks)

        # 3️⃣ 결과 정리 (입력 순서 그대로 유지)
        results = []
        for idx, result in enumerate(raw_results, start=1):
            results.append({
                "order": idx,                 # 입력 순서
                "path": result["path"],
                "text": result["text"],
                "error": result["error"]
            })

        return results
