"""
Загрузка документов через SimpleDirectoryReader

SimpleDirectoryReader сам парсит PDF/DOCX/TXT рекурсивно
Возвращает список LlamaIndex Document, готовый для store.build_all_indexes
"""

import logging
import re
from pathlib import Path
from typing import Optional

from llama_index.core import SimpleDirectoryReader, Document

logging.getLogger("pypdf").setLevel(logging.ERROR)


# PDF-ридер с лимитом страниц для больших файлов
import os as _os

_PDF_SIZE_LIMIT_MB = float(_os.getenv("PDF_PAGE_LIMIT_ABOVE_MB", "10"))
_MAX_PDF_PAGES = int(_os.getenv("MAX_PDF_PAGES", "50"))

def _read_pdf_limited(file_path, max_pages: int) -> str:
    """читает первые max_pages страниц PDF"""
    from pypdf import PdfReader
    reader = PdfReader(str(file_path))
    pages = reader.pages[:max_pages]
    return "\n".join((p.extract_text() or "") for p in pages)


# year из структуры пути
def _parse_year_from_path(path_str: str) -> Optional[int]:
    # ищет 4-значный год в компонентах пути
    for part in Path(path_str).parts:
        m = re.match(r"(19|20)\d{2}", part)
        if m:
            return int(m.group(0))
    return None


def _file_metadata(path_str: str) -> dict:
    # Хук SimpleDirectoryReader
    return {
        "doc_path": path_str,
        "year": _parse_year_from_path(path_str),
    }


# Определение языка
_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
_LATIN = re.compile(r"[a-zA-Z]")


def detect_language(text: str) -> str:
    sample = text[:3000]
    cyr = len(_CYRILLIC.findall(sample))
    lat = len(_LATIN.findall(sample))
    total = cyr + lat
    if total == 0:
        return "ru"
    rat = cyr / total
    if rat > 0.7:
        return "ru"
    if rat < 0.3:
        return "en"
    return "mixed"


# entrypoint
def load_documents(data_dir: str | Path) -> list[Document]:
    """
    Загружает все документы из директории

    Шаги:
      1. SimpleDirectoryReader парсит файлы + подставляет year через хук
      2. Дописываем language по тексту
      3. Прячем служебные поля из эмбеддинга и LLM-контекста

    Возвращает список Document для store.build_all_indexes.
    """
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

    documents = []
    skipped = []
    input_files = reader.input_files

    import time

    from tqdm import tqdm
    pbar = tqdm(input_files, desc="Загрузка файлов")
    for fpath in pbar:
        fp = Path(fpath)
        size_mb = fp.stat().st_size / 1e6
        pbar.set_postfix_str(f"{fp.name[:30]} ({size_mb:.1f}МБ)")
        t0 = time.time()

        try:
            if fp.suffix.lower() == ".pdf" and size_mb > _PDF_SIZE_LIMIT_MB:
                text = _read_pdf_limited(fp, _MAX_PDF_PAGES)
                if text.strip():
                    meta = _file_metadata(str(fp))
                    documents.append(Document(text=text, metadata=meta))
                    print(f"Большой PDF ({size_mb:.0f}МБ): "
                          f"прочитаны первые {_MAX_PDF_PAGES} стр. - {fp.name}")
            else:
                docs = reader.load_file(
                    input_file=fpath,
                    file_metadata=_file_metadata,
                    file_extractor=reader.file_extractor,
                    errors="ignore",
                    raise_on_error=False,
                )
                documents.extend(docs)

            dt = time.time() - t0
            if dt > 15:
                print(f"Медленный файл ({dt:.0f}с): {fp.name}")
        except Exception as e:
            skipped.append(f"{fp.name}: {type(e).__name__}")
            continue

    if skipped:
        print(f"Пропущено файлов: {len(skipped)}")

    # пост-обработка
    for doc in documents:
        doc.metadata["language"] = detect_language(doc.text)

        keep = {"doc_path", "year", "language"}
        doc.metadata = {k: v for k, v in doc.metadata.items() if k in keep}

        doc.excluded_embed_metadata_keys = ["doc_path"]
        doc.excluded_llm_metadata_keys = ["doc_path"]

    print(f"Загружено документов: {len(documents)}")
    return documents
