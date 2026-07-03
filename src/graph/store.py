import os
from typing import Any
from yandex_ai_studio_sdk import AIStudio
from dotenv import load_dotenv

# Компоненты ядра LlamaIndex
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Settings
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback

# Официальные эмбеддинги Яндекса
from llama_index.embeddings.yandexgpt import YandexGPTEmbedding


# 1. СОЗДАЕМ ОБЕРТКУ ДЛЯ YANDEX AI STUDIO SDK (Генерация)
class YandexAIStudioLlamaIndex(CustomLLM):
    api_key: str
    folder_id: str
    model_name: str = "yandexgpt"
    temperature: float = (
        0.2  # Низкая температура важна для точного извлечения сущностей
    )

    _sdk: Any = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._sdk = AIStudio(folder_id=self.folder_id, auth=self.api_key)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192, num_output=2000, model_name=self.model_name
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        model = self._sdk.models.completions(self.model_name).configure(
            temperature=self.temperature
        )
        result = model.run(prompt)
        text_response = "".join([str(alternative) for alternative in result])
        return CompletionResponse(text=text_response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError()


load_dotenv()
# 2. НАСТРОЙКА КЛЮЧЕЙ И КОНФИГУРАЦИИ
YANDEX_API_KEY = os.getenv("API_KEY")
FOLDER_ID = os.getenv("FOLDER_ID")

# Инициализируем LLM (через официальный SDK Яндекса)
llm_model = YandexAIStudioLlamaIndex(
    api_key=YANDEX_API_KEY,
    folder_id=FOLDER_ID,
    model_name="yandexgpt",  # Для сложных графов лучше использовать базовый yandexgpt, а не lite
)

# Инициализируем Эмбеддинги (через официальный коннектор LlamaIndex)
embed_model = YandexGPTEmbedding(api_key=YANDEX_API_KEY, folder_id=FOLDER_ID)

# Записываем всё в глобальные настройки LlamaIndex
Settings.llm = llm_model
Settings.embed_model = embed_model


# 3. ЗАГРУЗКА ДОКУМЕНТОВ И ПОСТРОЕНИЕ ГРАФА
def create_knowledge_graph(data_dir: str):
    print("Загрузка документов...")
    # Читаем все файлы (pdf, txt, docx) из указанной директории
    documents = SimpleDirectoryReader(data_dir, recursive=True).load_data()

    print("Построение ассоциативного графа (PropertyGraph)...")
    # PropertyGraphIndex автоматически отправляет куски текста в LLM,
    # просит извлечь сущности/связи и строит структуру графа
    index = PropertyGraphIndex.from_documents(documents, show_progress=True)

    # Сохраняем граф локально, чтобы не строить его заново при каждом запуске
    index.storage_context.persist(persist_dir="./storage_graph")
    print("Граф успешно построен и сохранен в папку ./storage_graph")
    return index


# Вызов функции (создайте папку 'my_documents' и положите туда файлы)
if __name__ == "__main__":
    os.makedirs("./my_documents", exist_ok=True)

    # Если граф еще не создан, создаем его
    if not os.path.exists("./storage_graph"):
        index = create_knowledge_graph("./my_documents")
    else:
        # Если уже создан — просто загружаем из памяти
        from llama_index.core import StorageContext, load_index_from_storage

        print("Загрузка существующего графа из памяти...")
        storage_context = StorageContext.from_defaults(persist_dir="./storage_graph")
        index = PropertyGraphIndex.from_storage(storage_context)

    # 4. ЗАПРОСЫ К ГРАФУ АССОЦИАЦИЙ
    # Создаем поисковый движок на базе графа
    query_engine = index.as_query_engine()

    response = query_engine.query(
        "Какие ключевые взаимосвязи между объектами описаны в документах?"
    )
    print("\nОтвет системы:")
    print(response)
