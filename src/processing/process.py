"""
главный скрипт парсера

Запуск:
    python -m src.processing.process
    
Обходит data/raw, парсит документы, сохраняет их ЦЕЛИКОМ в
data/processed/documents.jsonl (по документу на строку)
"""

from pathlib import Path

from src.processing.loader import iter_documents
from src.processing.chunker import documents_to_records, save_records_to_jsonl

def process():
    DATA_INPUT_DIR = Path("data/raw")
    
    if not DATA_INPUT_DIR.exists():
        print(f"Директория {DATA_INPUT_DIR} не найдена")
        return

    raw_docs = list(iter_documents(DATA_INPUT_DIR))
    print(f"Успешно загружено документов: {len(raw_docs)}")
    
    records = documents_to_records(raw_docs)
    print(f"Готово к сохранению: {len(records)} документов")
    
    print("Сохранение...\n")
    save_records_to_jsonl(records, "documents.jsonl")
    
if __name__ == "__main__":
    process()