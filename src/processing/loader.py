"""
Загрузка документов через SimpleDirectoryReader

SimpleDirectoryReader сам парсит PDF/DOCX/TXT рекурсивно,
но он не знает про структуру папок, поэтому year подставляем через хук 
file_metadata, а language дописываем после загрузки
"""

import re
from pathlib import Path
from typing import Optional

from llama_index.core import SimpleDirectoryReader, Document


# year из структуры пути
def _parse_year_from_path(path_str: str) -> Optional[int]:
    # ищет 4-значный год в компонентах пути
    for part in Path(path_str).parts:
        m = re.match(r"(19|20)\d{2}", part)
        if m:
            return int(m.group(0))
    return None


def _file_metadata(path_str: str) -> dict:
    # хук
    return {
        "doc_path": path_str,
        "year": _parse_year_from_path(path_str),
    }


# определение языка
CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
LATIN = re.compile(r"[a-zA-Z]")

def detect_language(text: str) -> str:
    cyr = len(CYRILLIC.findall(text))
    lat = len(LATIN.findall(text))
    
    total = cyr + lat
    if total == 0:
        return "ru"
    
    rat = cyr / total
    if rat > 0.7:
        return "ru"
    if rat < 0.3:
        return "en"
    
    # иначе непонятно
    return "mixed"


# загрузка документов
def load_documents(data_dir: str | Path) -> list[Document]:

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Директория не найдена: {data_dir}")

    reader = SimpleDirectoryReader(
        input_dir=str(data_dir),
        recursive=True,
        file_metadata=_file_metadata,
        errors="ignore",
        raise_on_error=False,
    )
    documents = reader.load_data(show_progress=True)

    # пост-обработка
    for doc in documents:
        doc.metadata["language"] = detect_language(doc.text)

        # оставляем только нужное: doc_path, year, language
        keep = {"doc_path", "year", "language"}
        doc.metadata = {k: v for k, v in doc.metadata.items() if k in keep}

        # служебное поле
        doc.excluded_embed_metadata_keys = ["doc_path"]
        doc.excluded_llm_metadata_keys = ["doc_path"]

    print(f"Загружено документов: {len(documents)}")
    return documents