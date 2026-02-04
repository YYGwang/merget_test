import os
from openai import OpenAI
from app.core.config import settings

class STTParser:
    def __init__(self):
        # cleaner_node처럼 settings에서 가져온 키를 명시적으로 주입
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def transcribe(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            return "음성 파일을 찾을 수 없습니다."

        try:
            with open(audio_path, "rb") as audio_file:
                # whisper-1 모델 호출
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            return transcript.strip()
        except Exception as e:
            return f"STT 변환 오류: {str(e)}"