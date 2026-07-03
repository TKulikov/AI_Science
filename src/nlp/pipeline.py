import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
 
from llama_index.core import Document

@dataclass
class DocRecord:
    text: str
    doc_path: str
    language: str
    source_type: str
    journal: Optional[str]
    year: Optional[int]
    geography: str
    

# Загрузка документов из jsonl
def load_records(jsonl_path: Path) -> list[DocRecord]:
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Файл документов не найден: {jsonl_path}\n"
            f"Сначала запустите: python -m src.processing.process"
        )
        
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d= json.loads(line)
                records.append(DocRecord(
                    text=d["text"],
                    doc_path=d["doc_path"],
                    language=d.get("language", "ru"),
                    source_type=d.get("source_type", "unknown"),
                    journal=d.get("journal"),
                    year=d.get("year"),
                    geography=d.get("geography", "RU"),
                ))
            except (KeyError, json.JSONDecodeError) as e:
                print(f" Пропущена строка {line_num}: {e}")
                
    return records

# Конвертация в LlamaIndex Document
def record_to_document(rec: DocRecord) -> Document:
    """
    Оборачивает DocRecord в LlamaIndex Document
    Метаданные: year + language. doc_path исключён из
    эмбеддинга и LLM-контекста (служебное поле, не несёт смысла)
    """
    return Document(
        text=rec.text,
        metadata={
            "doc_path": rec.doc_path,
            "year": rec.year,
            "language": rec.language,
        },
        excluded_embed_metadata_keys=["doc_path"],
        excluded_llm_metadata_keys=["doc_path"],
    )
    

def load_documents(jsonl_path: Path) -> list[Document]:
    """
    Использование:
        docs = load_documents(Path("data/processed/documents.jsonl"))
    """
    
    records = load_records(jsonl_path)
    print(f"Загружено документов: {len(records)}")
    
    documents = [
        record_to_document(rec)
        for rec in records
        if rec.text and rec.text.strip()
    ]
    print(f"Подготовлено Documents: {len(documents)}")
    
    return documents