import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()


# Модель для валидации входящего запроса
class UserMessage(BaseModel):
    text: str


# Эндпоинт для обработки сообщений
@app.post("/api/chat")
async def chat_endpoint(message: UserMessage):
    user_text = message.text

    # Здесь должна быть логика вашей LLM и поиск источников (RAG)
    # Имитируем ответ в формате dict {ans: "...", src: "..."}
    llm_answer = f"Вы спросили: '{user_text}'. Это искусственный ответ модели."
    sources = "1. Документация FastAPI; 2. База знаний компании."

    return {"ans": llm_answer, "src": sources}


# Эндпоинт для отдачи фронтенд-страницы
@app.get("/", response_class=HTMLResponse)
async def get_chat_page():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
