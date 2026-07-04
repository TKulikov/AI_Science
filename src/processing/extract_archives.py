"""
Распаковка архивов в корпусе

Раскрывает .zip, .rar и склеенные многотомники (.001/.002/...) прямо
в те же папки

Запуск: python src/processing/extract_archives.py --data-dir data/raw
Требует: pip install rarfile
"""
import argparse
import zipfile
from pathlib import Path


def _extract_zip(path: Path, dest: Path) -> int:
    """распаковывает zip. Возвращает число извлечённых файлов"""
    try:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            z.extractall(dest)
            return len(names)
    except Exception as e:
        print(f"zip {path.name}: {e}")
        return 0


def _extract_rar(path: Path, dest: Path) -> int:
    """распаковывает rar. Требует rarfile"""
    try:
        import rarfile
        with rarfile.RarFile(path) as r:
            names = [n for n in r.namelist() if not n.endswith("/")]
            r.extractall(dest)
            return len(names)
    except ImportError:
        print("rar: нужен `pip install rarfile` и unrar/7z в системе")
        return 0
    except Exception as e:
        print(f"rar {path.name}: {e}")
        return 0


def _join_multipart(first: Path) -> Path | None:
    stem = first.with_suffix("")
    parts = sorted(first.parent.glob(f"{first.stem}.[0-9][0-9][0-9]"))
    if not parts:
        return None
    joined = stem.with_suffix(".joined.zip")
    try:
        with open(joined, "wb") as out:
            for p in parts:
                out.write(p.read_bytes())
        return joined
    except Exception as e:
        print(f"  ✗ склейка {first.name}: {e}")
        return None


def extract_all(data_dir: Path) -> None:
    archives = list(data_dir.rglob("*.zip")) + list(data_dir.rglob("*.rar"))
    multipart = list(data_dir.rglob("*.001"))

    print(f"Найдено: {len(archives)} архивов, {len(multipart)} многотомников\n")

    total_files = 0

    for arc in archives:
        dest = arc.parent / f"_extracted_{arc.stem}"
        dest.mkdir(exist_ok=True)
        if arc.suffix.lower() == ".zip":
            n = _extract_zip(arc, dest)
        else:
            n = _extract_rar(arc, dest)
        if n:
            print(f"{arc.name} -> {n} файлов")
            total_files += n

    for first in multipart:
        joined = _join_multipart(first)
        if joined:
            dest = first.parent / f"_extracted_{first.stem}"
            dest.mkdir(exist_ok=True)
            n = _extract_zip(joined, dest)
            joined.unlink(missing_ok=True)  # убираем временный склеенный
            if n:
                print(f"{first.name} (многотомник) -> {n} файлов")
                total_files += n

    print(f"Всего извлечено файлов: {total_files}")
    print("Теперь запустите индексацию, извлечённые документы попадут в корпус.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = ap.parse_args()
    extract_all(args.data_dir)