"""
Хранилище графа и векторного индекса на LlamaIndex

Стек:
  - LLM в Ollama (Qwen2.5)
  - Embeddings с HuggingFace multilingual-e5 (ru+en)

Граф строится не поштучно, а по КЛАСТЕРАМ близких фрагментов.
Точка входа: build_all_indexes
"""

import os
from pathlib import Path
from typing import List

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

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.graph.clustering import cluster_pipeline

load_dotenv()

GRAPH_PERSIST_DIR = Path("data/graph_store")
VECTOR_PERSIST_DIR = Path("data/vector_store")

# LLM (Ollama / Qwen)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300.0"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

# Эмбеддинги (multilingual-e5-base) 
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda") # "cpu" если нет GPU

# Граф 
NUM_WORKERS = int(os.getenv("KG_NUM_WORKERS", "4")) # параллельные вызовы Ollama
MAX_TRIPLETS_PER_CHUNK = int(os.getenv("KG_MAX_TRIPLETS", "15"))
EMBED_KG_NODES = os.getenv("KG_EMBED_NODES", "false").lower() == "true"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2048"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))


# Фабрики LLM и эмбеддингов
def _make_llm() -> Ollama:
    return Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=OLLAMA_TIMEOUT,
        temperature=OLLAMA_TEMPERATURE,
    )


def _make_embed_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        device=EMBED_DEVICE,
    )


def _init_settings() -> Ollama:
    """настраивает глобальные Settings и возвращает LLM для экстрактора"""
    llm = _make_llm()
    Settings.llm = llm
    Settings.embed_model = _make_embed_model()
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
    return llm


# Экстрактор триплетов
def _make_path_extractor(llm: Ollama) -> DynamicLLMPathExtractor:
    return DynamicLLMPathExtractor(
        llm=llm,
        max_triplets_per_chunk=MAX_TRIPLETS_PER_CHUNK,
        num_workers=NUM_WORKERS,
        allowed_entity_types=None,
        allowed_relation_types=None,
        allowed_relation_props=None,
        allowed_entity_props=None,
    )


# Построение графового индекса
def build_graph_index(
    documents: List[Document],
    persist_dir: Path = GRAPH_PERSIST_DIR,
) -> PropertyGraphIndex:
    """
    строит PropertyGraphIndex по кластерам близких фрагментов

    Поток:
      1. Режем документы на nodes (SentenceSplitter)
      2. Кластеризуем близкие nodes (clustering.cluster_pipeline)
      3. Каждая группа это один вызов LLM для извлечения триплетов
    """
    llm = _init_settings()

    # 1.
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Документов: {len(documents)} nodes: {len(nodes)}")

    # 2.
    embed_model = Settings.embed_model

    def _embed_batch(texts: list[str]) -> list[list[float]]:
        return embed_model.get_text_embedding_batch(texts, show_progress=False)

    group_nodes = cluster_pipeline(nodes, _embed_batch)

    # 3.
    index = PropertyGraphIndex(
        nodes=group_nodes,
        llm=llm,
        embed_kg_nodes=EMBED_KG_NODES,
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


# Построение векторного индекса
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


# Загрузка индексов с диска
def load_graph_index(persist_dir: Path = GRAPH_PERSIST_DIR) -> PropertyGraphIndex:
    _init_settings()
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Граф не найден: {persist_dir}. Сначала запустите index.py"
        )
    storage_ctx = StorageContext.from_defaults(persist_dir=str(persist_dir))
    return load_index_from_storage(storage_ctx)


def load_vector_index(persist_dir: Path = VECTOR_PERSIST_DIR) -> VectorStoreIndex:
    _init_settings()
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Векторный индекс не найден: {persist_dir}. Сначала запустите index.py"
        )
    storage_ctx = StorageContext.from_defaults(persist_dir=str(persist_dir))
    return load_index_from_storage(storage_ctx)


# entrypoint
def build_all_indexes(
    documents: List[Document],
) -> tuple[PropertyGraphIndex, VectorStoreIndex]:
    """
    Возвращает (graph_index, vector_index).
    """
    print(f"Документов на входе: {len(documents)}")

    print("Строим граф (LLM извлекает триплеты)...")
    graph_index = build_graph_index(documents)

    print("Строим векторный индекс...")
    vector_index = build_vector_index(documents)

    print("Оба индекса построены и сохранены")
    return graph_index, vector_index