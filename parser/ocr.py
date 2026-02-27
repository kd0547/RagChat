import os
os.environ['USERPROFILE'] = 'C:/AI'
os.environ['HOME'] = 'C:/AI'
from paddleocr import PaddleOCRVL


class DocumentOCRPreprocessor:
    """PaddleOCR-VL을 이용한 문서 OCR + 구조화 전처리기 (RAG 임베딩 전용)"""

    def __init__(self):
        self.pipeline = PaddleOCRVL()

    def process(self, image_path_or_bytes) -> str:
        # 이미지 → Markdown/JSON/텍스트 추출
        result = self.pipeline.predict(image_path_or_bytes)
        return result['markdown']  # 또는 result['text'], result['structured']

    def process_file(self, file_path: str) -> str:
        # PDF/이미지 파일 지원 로직 추가
        ...



