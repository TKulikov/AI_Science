"""
главный скрипт парсера
"""

from pathlib import Path
from src.processing.loader import iter_documents
from src.processing.chunker import chunk_documents, save_chunks_to_jsonl

def process():
    DATA_INPUT_DIR = Path("data/raw")
    
    if not DATA_INPUT_DIR.exists():
        print(f"Директория {DATA_INPUT_DIR} не найдена")
        return

    raw_docs = list(iter_documents(DATA_INPUT_DIR))
    print(f"Успешно загружено документов: {len(raw_docs)}")
    
    all_chunks = chunk_documents(raw_docs)
    print(f"Успешно создано чанков: {len(all_chunks)}")
    
    print("Сохранение...\n")
    save_chunks_to_jsonl(all_chunks, "chunks.jsonl")
    
if __name__ == "__main__":
    process()