"""
Сериализация документов в jsonl

Раньше здесь была нарезка на чанки. Теперь нарезку делает LlamaIndex
при индексации, поэтому здесь мы просто сохраняем каждый документ 
одной строкой jsonl со всеми метаданными
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from src.processing.loader import RawDocument


OUTPUT_DIR = Path("data/processed")


@dataclass
class DocRecord:
    text: str
    doc_path: str
    language: str
    source_type: str
    journal: Optional[str]
    year: Optional[int]
    geography: str
    
    
def document_to_record(doc: RawDocument) -> DocRecord:
    # конвертируем RawDocument в DocRecord
    return DocRecord(
        text=doc.text,
        doc_path=str(doc.path),
        language=doc.language,
        source_type=doc.source_type,
        journal=doc.journal,
        year=doc.year,
        geography=doc.geography,
    )
       
def documents_to_records(docs: list[RawDocument]) -> list[DocRecord]:
    return [
        document_to_record(doc)
        for doc in docs
        if doc.text and doc.text.strip()
    ]

 
def save_records_to_jsonl(records: list[DocRecord], output_filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_filename
 
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            
    print(f"Done! {len(records)} документов")