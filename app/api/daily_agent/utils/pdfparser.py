import fitz
import re
import os


class PDFParser:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 400):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_and_chunk(self, file_path: str) -> list[str]:
        if not os.path.exists(file_path):
            return []

        doc = fitz.open(file_path)
        full_text = ""

        for page in doc:
            # "text" 모드는 표나 이미지를 무시하고 '순수 텍스트 흐름'만 가져옵니다.
            page_text = page.get_text("text").strip()

            # 텍스트가 있는 경우에만 추가 (그림/표만 있는 페이지는 무시됨)
            if page_text:
                full_text += page_text + "\n"

        doc.close()

        # 만약 전체 PDF에 텍스트가 하나도 없다면 빈 리스트 반환
        if not full_text.strip():
            return []

        # 기본 정제
        cleaned_text = re.sub(r'\n{3,}', '\n\n', full_text).strip()

        # 청킹 로직
        chunks = []
        if len(cleaned_text) <= self.chunk_size:
            return [cleaned_text]

        start = 0
        while start < len(cleaned_text):
            end = start + self.chunk_size
            chunks.append(cleaned_text[start:end])
            start += (self.chunk_size - self.chunk_overlap)
            if end >= len(cleaned_text):
                break

        return chunks