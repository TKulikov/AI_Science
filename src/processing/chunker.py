"""
Разбивка документов на чанки с сохранением метаданных.
Чанки сохраняются в data/processed
"""

import json
from pathlib import Path
from typing import Iterator
from dataclasses import dataclass, asdict
from src.processing.loader import RawDocument


CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
OUTPUT_DIR = Path("data/processed")


@dataclass
class Chunk:
    text: str
    doc_path: str
    chunk_index: int
    language: str
    source_type: str
    journal: str | None
    year: int | None
    geography: str
    
    
def split_by_tokens(text: str, size: int, overlap: int) -> Iterator[str]:
    # разбиваем текст на фрагменты по словами
    # каждый фрагмент перекрываем с предыдущим на overlap слов
    
    words = text.split()
    if not words:
        return
    
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        yield " ".join(words[start:end])
        if end == len(words):
            break
        start += size - overlap
        
def chunk_document(doc: RawDocument) -> list[Chunk]:
    # разбиваем RawDocument на список Chunk-ов
    
    return [
        Chunk(
            text=fragment,
            doc_path=str(doc.path),
            chunk_index=i,
            language=doc.language,
            source_type=doc.source_type,
            journal=doc.journal,
            year=doc.year,
            geography=doc.geography,
        )
        for i, fragment in enumerate(split_by_tokens(doc.text, CHUNK_SIZE, CHUNK_OVERLAP))
        if fragment.strip()
    ]

def chunk_documents(docs: list[RawDocument]) -> list[Chunk]:
    # применяем chunk_document ко всему корпусу
    
    result = []
    for doc in docs:
        result.extend(chunk_document(doc))
    return result

 
def save_chunks_to_jsonl(chunks: list[Chunk], output_filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_filename
    
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json_str = json.dumps(asdict(chunk), ensure_ascii=False)
            f.write(json_str + '\n')
    print(f"Done! {len(chunks)} chunks")