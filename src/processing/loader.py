import logging
import os
import re
from pathlib import Path
from typing import Iterator, Optional

from llama_index.core import SimpleDirectoryReader, Document

logging.getLogger("pypdf").setLevel(logging.ERROR)

BATCH_SIZE = int(os.getenv("LOAD_BATCH_SIZE", "50"))
SCAN_PROBE_PAGES = int(os.getenv("SCAN_PROBE_PAGES", "3"))
SCAN_MIN_CHARS = int(os.getenv("SCAN_MIN_CHARS_PER_PAGE", "50"))
MAX_FILE_MB = float(os.getenv("MAX_FILE_MB", "60"))

_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
_LATIN = re.compile(r"[a-zA-Z]")
_YEAR = re.compile(r"(19|20)\d{2}")


def _year_from_path(path: str) -> Optional[int]:
    for part in Path(path).parts:
        m = _YEAR.match(part)
        if m:
            return int(m.group(0))
    return None


def _metadata(path: str) -> dict:
    return {"doc_path": path, "year": _year_from_path(path)}


def detect_language(text: str) -> str:
    sample = text[:3000]
    cyr = len(_CYRILLIC.findall(sample))
    lat = len(_LATIN.findall(sample))
    total = cyr + lat
    if not total:
        return "ru"
    ratio = cyr / total
    if ratio > 0.7:
        return "ru"
    if ratio < 0.3:
        return "en"
    return "mixed"


def _is_scan(path: Path) -> bool:
    try:
        from pypdf import PdfReader
        pages = PdfReader(str(path)).pages[:SCAN_PROBE_PAGES]
        if not pages:
            return False
        chars = sum(len(p.extract_text() or "") for p in pages)
        return chars / len(pages) < SCAN_MIN_CHARS
    except Exception:
        return False


def _prefilter(files: list[Path]) -> tuple[list[Path], list[str]]:
    from tqdm import tqdm
    keep, skipped = [], []
    for fp in tqdm(files, desc="Фильтрация"):
        size_mb = fp.stat().st_size / 1e6
        if MAX_FILE_MB and size_mb > MAX_FILE_MB:
            skipped.append(f"{fp.name}: {size_mb:.0f}МБ")
            continue
        if fp.suffix.lower() == ".pdf" and _is_scan(fp):
            skipped.append(f"{fp.name}: скан")
            continue
        keep.append(fp)
    return keep, skipped


def list_good_files(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    reader = SimpleDirectoryReader(
        input_dir=str(data_dir), recursive=True,
        file_metadata=_metadata, errors="ignore", raise_on_error=False,
    )
    files = [Path(f) for f in reader.input_files]
    print(f"Найдено: {len(files)}")

    good, skipped = _prefilter(files)
    print(f"Годных: {len(good)}, отсеяно: {len(skipped)}")
    for s in skipped[:15]:
        print(f"  {s}")
    if len(skipped) > 15:
        print(f"  +{len(skipped) - 15}")
    return good


def iter_document_batches(
    data_dir: str | Path, batch_size: int = BATCH_SIZE,
) -> Iterator[list[Document]]:
    files = list_good_files(data_dir)
    if not files:
        return

    total = (len(files) + batch_size - 1) // batch_size
    keep_keys = {"doc_path", "year", "language"}

    for i in range(0, len(files), batch_size):
        chunk = files[i:i + batch_size]
        reader = SimpleDirectoryReader(
            input_files=[str(f) for f in chunk],
            file_metadata=_metadata, errors="ignore", raise_on_error=False,
        )
        docs = reader.load_data(show_progress=False)
        for doc in docs:
            doc.metadata["language"] = detect_language(doc.text)
            doc.metadata = {k: v for k, v in doc.metadata.items() if k in keep_keys}
            doc.excluded_embed_metadata_keys = ["doc_path"]
            doc.excluded_llm_metadata_keys = ["doc_path"]
        print(f"Батч {i // batch_size + 1}/{total}: {len(docs)}")
        yield docs


def load_documents(data_dir: str | Path) -> list[Document]:
    result = []
    for batch in iter_document_batches(data_dir):
        result.extend(batch)
    return result