"""
Скрипт индексации

Запуск:
    python src/index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.loader import load_documents
from src.graph.store import build_all_indexes


def main():

    DATA_DIR = "data/raw"

    documents = load_documents(DATA_DIR)
    if not documents:
        print("Нет документов для индексации")
        sys.exit(1)

    graph_index, vector_index = build_all_indexes(documents)

    print("Готово!")


if __name__ == "__main__":
    main()