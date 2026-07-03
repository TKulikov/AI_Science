import os
from dotenv import load_dotenv
from yandex_ai_studio_sdk import AIStudio
from llama_index.core import Settings
import wikipedia
from llama_index.core import Document, PropertyGraphIndex
from llama_index.core.indices.property_graph import (
    SimpleLLMPathExtractor,
    SchemaLLMPathExtractor,
    DynamicLLMPathExtractor,
)

load_dotenv()
folder_id = os.getenv("FOLDER_ID")
api_key = os.getenv("API_KEY")
if api_key is None or folder_id is None:
    print("get env key failed")
    raise RuntimeError


def get_wikipedia_content(title):
    try:
        page = wikipedia.page(title)
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation page. Options: {e.options}")
    except wikipedia.exceptions.PageError:
        print(f"Page '{title}' does not exist.")
    return None


from typing import Any
from yandex_ai_studio_sdk import AIStudio  # Официальный SDK Яндекса

from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback


class YandexAIStudioLlamaIndex(CustomLLM):
    api_key: str
    folder_id: str
    model_name: str = "yandexgpt"
    temperature: float = 0.6

    # Внутренний объект официального SDK (не участвует в pydantic-валидации LlamaIndex)
    _sdk: Any = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Инициализируем официальный SDK Яндекса
        self._sdk = AIStudio(
            folder_id=self.folder_id,
            auth=self.api_key,  # SDK сам разберется с типами ключей (API-Key, IAM и т.д.)
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Метаданные, которые LlamaIndex запрашивает у модели."""
        return LLMMetadata(
            context_window=8192, num_output=2000, model_name=self.model_name
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Используем родные методы официального SDK Яндекса
        model = self._sdk.models.completions(self.model_name)
        model = model.configure(temperature=self.temperature)

        # Запускаем генерацию текста
        result = model.run(prompt)

        # Извлекаем текст ответа из генератора альтернатив SDK
        text_response = ""
        for alternative in result:
            text_response += str(alternative)

        return CompletionResponse(text=text_response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError("Стриминг пока не реализован")


# ==========================================
# ИСПОЛЬЗОВАНИЕ В ВАШЕМ ПРОЕКТЕ
# ==========================================


# Создаем обертку над официальным SDK
def test():
    llm = YandexAIStudioLlamaIndex(
        api_key=api_key,
        folder_id=folder_id,
        model_name="yandexgpt",  # можно поменять на yandexgpt-lite
    )

    # Передаем в глобальные настройки LlamaIndex
    Settings.llm = llm

    # Теперь LlamaIndex прозрачно вызывает официальный SDK Яндекса под капотом!
    try:
        response = Settings.llm.complete("Привет! Ты работаешь через официальный SDK?")
        print("Ответ YandexGPT:")
        print(response)
    except Exception as e:
        print(f"Ошибка: {e}")


def llama():

    llm = YandexAIStudioLlamaIndex(
        api_key=api_key,
        folder_id=folder_id,
        model_name="yandexgpt",  # можно поменять на yandexgpt-lite
    )
    Settings.llm = llm
    Settings.chunk_size = 2048
    Settings.chunk_overlap = 20
    wiki_title = "Barack Obama"
    content = get_wikipedia_content(wiki_title)

    if content:
        document = Document(text=content, metadata={"title": wiki_title})
        print(f"Fetched content for '{wiki_title}' (length: {len(content)} characters)")
    else:
        print("Failed to fetch Wikipedia content.")

    kg_extractor = DynamicLLMPathExtractor(
        llm=llm,
        max_triplets_per_chunk=20,
        num_workers=4,
        allowed_entity_types=None,
        allowed_relation_types=None,
        allowed_relation_props=None,
        allowed_entity_props=None,
    )

    dynamic_index_2 = PropertyGraphIndex.from_documents(
        [document],
        llm=llm,
        embed_kg_nodes=False,
        kg_extractors=[kg_extractor],
        show_progress=True,
    )

    dynamic_index_2.property_graph_store.save_networkx_graph(
        name="./DynamicGraph_2.html"
    )
    dynamic_index_2.property_graph_store.get_triplets(
        entity_names=["Barack Obama", "Obama"]
    )[:5]


if __name__ == "__main__":
    llama()
