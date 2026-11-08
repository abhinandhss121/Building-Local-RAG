from pypdf import PdfReader
from pathlib import Path
from typing import List

class DocumentProcessor:
    """Handles document extraction and text chunking."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts all text from a given PDF file."""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def chunk_text(self, text: str) -> List[str]:
        """Splits text into overlapping chunks for better semantic retrieval."""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += self.chunk_size - self.overlap
        return chunks
