"""
Загрузка документов: предфильтр + родной параллельный load_data

Стратегия:
  1. Быстрый предфильтр, где мы отсеиваем сканы (PDF без текста) и гиганты,
     не парся их целиком
  2. Чистый список файлов в SimpleDirectoryReader.load_data(num_workers),
     который грузит параллельно
  3. Пост-обработка: язык по тексту + чистка метаданных

year подставляется через хук file_metadata, language после загрузки.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

from llama_index.core import SimpleDirectoryReader, Document

logging.getLogger("pypdf").setLevel(logging.ERROR)

# Параметры
NUM_WORKERS = int(os.getenv("LOAD_NUM_WORKERS", "4"))
SCAN_PROBE_PAGES = int(os.getenv("SCAN_PROBE_PAGES", "3"))
SCAN_MIN_CHARS_PER_PAGE = int(os.getenv("SCAN_MIN_CHARS_PER_PAGE", "50"))
MAX_FILE_MB = float(os.getenv("MAX_FILE_MB", "60"))


# year из структуры пути
def _parse_year_from_path(path_str: str) -> Optional[int]:
    for part in Path(path_str).parts:
        m = re.match(r"(19|20)\d{2}", part)
        if m:
            return int(m.group(0))
    return None


def _file_metadata(path_str: str) -> dict:
    return {"doc_path": path_str, "year": _parse_year_from_path(path_str)}


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


# Предфильтр: детектор сканов
def _is_scanned_pdf(file_path: Path) -> bool:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        probe = reader.pages[:SCAN_PROBE_PAGES]
        if not probe:
            return False
        total = sum(len((p.extract_text() or "")) for p in probe)
        return (total / len(probe)) < SCAN_MIN_CHARS_PER_PAGE
    except Exception:
        return False


def _prefilter(files: list[Path]) -> tuple[list[Path], list[str]]:
    keep = []
    skipped = []

    from tqdm import tqdm
    for fp in tqdm(files, desc="Предфильтр"):
        size_mb = fp.stat().st_size / 1e6

        if MAX_FILE_MB > 0 and size_mb > MAX_FILE_MB:
            skipped.append(f"{fp.name}: гигант ({size_mb:.0f}МБ)")
            continue

        if fp.suffix.lower() == ".pdf" and _is_scanned_pdf(fp):
            skipped.append(f"{fp.name}: скан (нет текста)")
            continue

        keep.append(fp)

    return keep, skipped


# entrypoint
def load_documents(data_dir: str | Path) -> list[Document]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Директория не найдена: {data_dir}")

    # Собираем все файлы через reader (он знает поддерживаемые форматы)
    base_reader = SimpleDirectoryReader(
        input_dir=str(data_dir),
        recursive=True,
        file_metadata=_file_metadata,
        errors="ignore",
        raise_on_error=False,
    )
    all_files = [Path(f) for f in base_reader.input_files]
    print(f"Найдено файлов: {len(all_files)}")

    # 1. Предфильтр
    good_files, skipped = _prefilter(all_files)
    print(f"После фильтра: {len(good_files)} годных, {len(skipped)} отсеяно")

    if not good_files:
        print("Нет файлов для загрузки после фильтра")
        return []

    # 2. Родной параллельный load_data на чистом списке
    reader = SimpleDirectoryReader(
        input_files=[str(f) for f in good_files],
        file_metadata=_file_metadata,
        errors="ignore",
        raise_on_error=False,
    )
    documents = reader.load_data(show_progress=True, num_workers=NUM_WORKERS)

    # 3. пост-обработка
    for doc in documents:
        doc.metadata["language"] = detect_language(doc.text)
        keep = {"doc_path", "year", "language"}
        doc.metadata = {k: v for k, v in doc.metadata.items() if k in keep}
        doc.excluded_embed_metadata_keys = ["doc_path"]
        doc.excluded_llm_metadata_keys = ["doc_path"]

    if skipped:
        print(f"Пропущено файлов: {len(skipped)}")

    print(f"Загружено документов: {len(documents)}")
    return documents
