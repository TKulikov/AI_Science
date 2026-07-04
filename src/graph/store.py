"""
Хранилище графа и векторного индекса на LlamaIndex и yandex-ai-studio-sdk

Весь стек идёт через один API-ключ Яндекса:
  - YandexAIStudioLlamaIndex (обёртка над SDK, с таймаутом)
  - YandexGPTEmbedding (официальный коннектор LlamaIndex)

Два индекса:
  - PropertyGraphIndex это триплеты через DynamicLLMPathExtractor
  - VectorStoreIndex это семантический поиск по эмбеддингам Яндекса

Точка входа: build_all_indexes
"""

import os
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv

from llama_index.core import (
    Settings,
    Document,
    PropertyGraphIndex,
    VectorStoreIndex,
)
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback

from llama_index.embeddings.yandexgpt import YandexGPTEmbedding

from yandex_ai_studio_sdk import AIStudio

# КОНФИГУРАЦИЯ 

load_dotenv()

FOLDER_ID = os.getenv("FOLDER_ID")
API_KEY = os.getenv("API_KEY")

GRAPH_PERSIST_DIR = Path("data/graph_store")
VECTOR_PERSIST_DIR = Path("data/vector_store")

# нарезка документов на nodes при индексации
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 128


# LLM обёртка над официальным yandex-ai-studio-sdk
class YandexAIStudioLlamaIndex(CustomLLM):
    api_key: str
    folder_id: str
    model_name: str = "yandexgpt"
    temperature: float = 0.3
    timeout: float = 60.0 # секунды 

    _sdk: Any = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._sdk = AIStudio(folder_id=self.folder_id, auth=self.api_key)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,
            num_output=2000,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        model = self._sdk.models.completions(self.model_name)
        
        model = model.configure(temperature=self.temperature)
        result = model.run(prompt, timeout=self.timeout)
        
        parts: list[str] = []
        for alt in result:
            text = getattr(alt, "text", None)
            parts.append(text if text is not None else str(alt))
        return CompletionResponse(text="".join(parts))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError("Стриминг не используется при индексации")


# глобальные settings 
def _init_settings() -> YandexAIStudioLlamaIndex:
    llm = YandexAIStudioLlamaIndex(
        api_key=API_KEY,
        folder_id=FOLDER_ID,
        model_name="yandexgpt",
        temperature=0.3,
    )
    Settings.llm = llm
    Settings.embed_model = YandexGPTEmbedding(
        api_key=API_KEY, folder_id=FOLDER_ID)
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
    return llm


# экстрактор триплетов
def _make_path_extractor(llm: YandexAIStudioLlamaIndex) -> DynamicLLMPathExtractor:
    return DynamicLLMPathExtractor(
        llm=llm,
        max_triplets_per_chunk=10, # 20
        num_workers=12, # думаю можно меньше
        allowed_entity_types=None,
        allowed_relation_types=None,
        allowed_relation_props=None,
        allowed_entity_props=None,
    )


# построение графового индекса
def build_graph_index(
    documents: List[Document],
    persist_dir: Path = GRAPH_PERSIST_DIR,
) -> PropertyGraphIndex:
    """
    Строим PropertyGraphIndex. LLM читает каждый node и извлекает триплеты
    при embed_kg_nodes=True узлы графа эмбеддятся

    Документы режем на nodes через SentenceSplitter ПЕРЕД экстрактором 
    Иначе DynamicLLMPathExtractor шлёт документ целиком в
    YandexGPT и упирается в лимит токенов.
    """
    llm = _init_settings()

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Документов: {len(documents)}, nodes: {len(nodes)}")

    index = PropertyGraphIndex(
        nodes=nodes,
        llm=llm,
        embed_kg_nodes=True,
        kg_extractors=[_make_path_extractor(llm)],
        show_progress=True,
    )

    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))

    # HTML-визуализация графа
    try:
        index.property_graph_store.save_networkx_graph(
            name=str(persist_dir / "graph.html")
        )
    except Exception as e:
        print(f"Не удалось сохранить graph.html: {e}")

    return index


# построение векторного индекса
def build_vector_index(
    documents: List[Document],
    persist_dir: Path = VECTOR_PERSIST_DIR,
) -> VectorStoreIndex:
    _init_settings()

    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))

    return index


# TODO: загрузка индексов с диска
# PropertyGraphIndex и VectorStoreIndex
# TODO: уменьшить время работы LLM

# entrypoint
def build_all_indexes(
    documents: List[Document],
) -> tuple[PropertyGraphIndex, VectorStoreIndex]:
    print(f"Документов на входе: {len(documents)}")

    print("Строим граф (LLM извлекает триплеты)...")
    graph_index = build_graph_index(documents)

    print("Строим векторный индекс...")
    vector_index = build_vector_index(documents)

    print("Оба индекса построены и сохранены!")
    return graph_index, vector_index
