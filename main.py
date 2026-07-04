"""
FastAPI-сервер: фронт (index.html) + эндпоинт /api/chat поверх query_engine

Индексы грузятся ОДИН раз при старте, не на каждый запрос

Запуск:
    python main.py
    # открыть http://127.0.0.1:8000
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.query.query_engine import aquery, load_indexes


# индексы держим в состоянии приложения
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Загрузка индексов (граф + вектор)...")
    graph, vector = load_indexes()
    _state["graph"] = graph
    _state["vector"] = vector
    print("Индексы загружены. Сервер готов.")
    yield
    _state.clear()


app = FastAPI(lifespan=lifespan)


class UserMessage(BaseModel):
    text: str


def _format_sources(sources: list[dict]) -> str:
    """
    Превращает список источников в строку для фронта
    """
    if not sources:
        return "—"
    parts = []
    for i, s in enumerate(sources, 1):
        from pathlib import Path
        name = Path(s["doc_path"]).name if s.get("doc_path") else "?"
        year = f" ({s['year']})" if s.get("year") else ""
        parts.append(f"{i}. {name}{year}")
    return "; ".join(parts)


@app.post("/api/chat")
async def chat_endpoint(message: UserMessage):
    result = await aquery(message.text, _state["graph"], _state["vector"])
    d = result.to_dict()  # {ans, src: [...]}
    return {
        "ans": d["ans"],
        "src": _format_sources(d["src"]),  # список → строка для фронта
    }


@app.get("/", response_class=HTMLResponse)
async def get_chat_page():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
