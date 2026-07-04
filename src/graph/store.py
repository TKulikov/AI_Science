"""
Хранилище графа и векторного индекса на LlamaIndex и yandex_ai_studio_sdk

Весь стек через один API-ключ Яндекса:
  - YandexAIStudioLlamaIndex (обёртка над SDK)
  - YandexGPTEmbedding (официальный коннектор)

Граф строится по КЛАСТЕРАМ близких фрагментов (clustering.py):
семантически похожие nodes группируются и идут в LLM единым контекстом
Это сокращает число вызовов LLM в разы и даёт лучшие связи между документами

Точка входа: build_all_indexes(documents)
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
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback

# официальный коннектор эмбеддингов Яндекса
from llama_index.embeddings.yandexgpt import YandexGPTEmbedding

from yandex_ai_studio_sdk import AIStudio

from src.graph.clustering import cluster_pipeline

# КОНФИГУРАЦИЯ
load_dotenv()

# .env может не найтись, если скрипт запущен не из корня проекта
if not os.getenv("API_KEY") or not os.getenv("FOLDER_ID"):
    from dotenv import find_dotenv
    _env_path = find_dotenv(usecwd=True)
    if not _env_path:
        _here = Path(__file__).resolve()
        for _parent in _here.parents:
            _candidate = _parent / ".env"
            if _candidate.exists():
                _env_path = str(_candidate)
                break
    if _env_path:
        load_dotenv(_env_path, override=False)

FOLDER_ID = os.getenv("FOLDER_ID")
API_KEY = os.getenv("API_KEY")

GRAPH_PERSIST_DIR = Path("data/graph_store")
VECTOR_PERSIST_DIR = Path("data/vector_store")

# Параметры графа
NUM_WORKERS = int(os.getenv("KG_NUM_WORKERS", "8"))
MAX_TRIPLETS_PER_CHUNK = int(os.getenv("KG_MAX_TRIPLETS", "15"))
EMBED_KG_NODES = os.getenv("KG_EMBED_NODES", "false").lower() == "true"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2048"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))


# 
# LLM обёртка над официальным yandex_ai_studio_sdk
class YandexAIStudioLlamaIndex(CustomLLM):
    api_key: str
    folder_id: str
    model_name: str = "yandexgpt"
    temperature: float = 0.3
    timeout: float = 60.0

    _sdk: Any = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._sdk = AIStudio(folder_id=self.folder_id, auth=self.api_key)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192, num_output=2000, model_name=self.model_name,
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



def _init_settings() -> YandexAIStudioLlamaIndex:
    llm = YandexAIStudioLlamaIndex(
        api_key=API_KEY, folder_id=FOLDER_ID,
        model_name="yandexgpt", temperature=0.3,
    )
    Settings.llm = llm
    Settings.embed_model = YandexGPTEmbedding(api_key=API_KEY, folder_id=FOLDER_ID)
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
    return llm


# экстрактор триплетов
def _make_path_extractor(llm: YandexAIStudioLlamaIndex) -> DynamicLLMPathExtractor:
    return DynamicLLMPathExtractor(
        llm=llm,
        max_triplets_per_chunk=MAX_TRIPLETS_PER_CHUNK,
        num_workers=NUM_WORKERS,
        allowed_entity_types=None,
        allowed_relation_types=None,
        allowed_relation_props=None,
        allowed_entity_props=None,
    )


# построение графа (по кластерам)

def build_graph_index(
    documents: List[Document],
    persist_dir: Path = GRAPH_PERSIST_DIR,
) -> PropertyGraphIndex:
    """
    Поток: документы -> nodes -> кластеризация -> группы -> триплеты (LLM)
    Кластеризация сокращает число вызовов LLM в разы
    """
    llm = _init_settings()

    # 1. Нарезка
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"  Документов: {len(documents)} → nodes: {len(nodes)}")

    # 2. Кластеризация: близкие nodes в группы
    embed_model = Settings.embed_model

    def _embed_batch(texts: list[str]) -> list[list[float]]:
        return embed_model.get_text_embedding_batch(texts, show_progress=False)

    group_nodes = cluster_pipeline(nodes, _embed_batch)

    # 3. Извлечение триплетов из групп
    index = PropertyGraphIndex(
        nodes=group_nodes,
        llm=llm,
        embed_kg_nodes=EMBED_KG_NODES,
        kg_extractors=[_make_path_extractor(llm)],
        show_progress=True,
    )

    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))

    try:
        index.property_graph_store.save_networkx_graph(
            name=str(persist_dir / "graph.html")
        )
    except Exception as e:
        print(f"Не удалось сохранить graph.html: {e}")

    return index

# Векторный индекс
def build_vector_index(
    documents: List[Document],
    persist_dir: Path = VECTOR_PERSIST_DIR,
) -> VectorStoreIndex:
    _init_settings()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    return index


# Загрузка индексов
def load_graph_index(persist_dir: Path = GRAPH_PERSIST_DIR) -> PropertyGraphIndex:
    _init_settings()
    if not persist_dir.exists():
        raise FileNotFoundError(f"Граф не найден: {persist_dir}. Запустите индексацию.")
    storage_ctx = StorageContext.from_defaults(persist_dir=str(persist_dir))
    return load_index_from_storage(storage_ctx)


def load_vector_index(persist_dir: Path = VECTOR_PERSIST_DIR) -> VectorStoreIndex:
    _init_settings()
    if not persist_dir.exists():
        raise FileNotFoundError(f"Векторный индекс не найден: {persist_dir}. Запустите индексацию.")
    storage_ctx = StorageContext.from_defaults(persist_dir=str(persist_dir))
    return load_index_from_storage(storage_ctx)


# entrypoint
def build_all_indexes(
    documents: List[Document],
) -> tuple[PropertyGraphIndex, VectorStoreIndex]:
    
    print(f"Документов на входе: {len(documents)}")
    print("Строим граф (LLM извлекает триплеты по кластерам)...")
    graph_index = build_graph_index(documents)

    print("Строим векторный индекс...")
    vector_index = build_vector_index(documents)
    
    print("Оба индекса построены и сохранены")
    return graph_index, vector_index