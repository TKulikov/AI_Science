import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.store import build_all_indexes_streaming, GRAPH_PERSIST_DIR, VECTOR_PERSIST_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = ap.parse_args()

    if not args.data_dir.exists():
        print(f"Директория не найдена: {args.data_dir}")
        sys.exit(1)

    build_all_indexes_streaming(str(args.data_dir))
    print(f"Граф: {GRAPH_PERSIST_DIR}")
    print(f"Вектор: {VECTOR_PERSIST_DIR}")


if __name__ == "__main__":
    main()