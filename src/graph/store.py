import gc
import os
from pathlib import Path
from typing import Any, List
from tqdm import tqdm
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
from llama_index.embeddings.yandexgpt import YandexGPTEmbedding
from yandex_ai_studio_sdk import AIStudio
import time

try:
    import asyncio

    asyncio.get_running_loop()
    import nest_asyncio

    nest_asyncio.apply()
except (RuntimeError, ImportError):
    pass

load_dotenv()

if not os.getenv("API_KEY") or not os.getenv("FOLDER_ID"):
    from dotenv import find_dotenv

    _env = find_dotenv(usecwd=True)
    if not _env:
        for parent in Path(__file__).resolve().parents:
            if (parent / ".env").exists():
                _env = str(parent / ".env")
                break
    if _env:
        load_dotenv(_env, override=False)

FOLDER_ID = os.getenv("FOLDER_ID")
API_KEY = os.getenv("API_KEY")

GRAPH_PERSIST_DIR = Path("data/graph_store")
VECTOR_PERSIST_DIR = Path("data/vector_store")

NUM_WORKERS = int(os.getenv("KG_NUM_WORKERS", "8"))
MAX_TRIPLETS = int(os.getenv("KG_MAX_TRIPLETS", "15"))
EMBED_KG_NODES = os.getenv("KG_EMBED_NODES", "false").lower() == "true"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2048"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))
MICRO_BATCH_SIZE = 5


def _check_credentials() -> None:
    if not FOLDER_ID or not API_KEY:
        missing = [
            n for n, v in (("FOLDER_ID", FOLDER_ID), ("API_KEY", API_KEY)) if not v
        ]
        raise RuntimeError(f"Не заданы {', '.join(missing)} в .env")


class YandexLLM(CustomLLM):
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
            context_window=8192, num_output=2000, model_name=self.model_name
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        model = self._sdk.models.completions(self.model_name).configure(
            temperature=self.temperature
        )
        result = model.run(prompt, timeout=self.timeout)
        text = "".join(getattr(alt, "text", str(alt)) for alt in result)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        raise NotImplementedError


def _make_llm() -> YandexLLM:
    _check_credentials()
    return YandexLLM(api_key=API_KEY, folder_id=FOLDER_ID)


def _make_embed() -> YandexGPTEmbedding:
    _check_credentials()
    return YandexGPTEmbedding(api_key=API_KEY, folder_id=FOLDER_ID)


def _init() -> YandexLLM:
    llm = _make_llm()
    Settings.llm = llm
    Settings.embed_model = _make_embed()
    Settings.chunk_size = 1000  # Hardcode
    Settings.chunk_overlap = CHUNK_OVERLAP
    return llm


def _extractor(llm: YandexLLM) -> DynamicLLMPathExtractor:
    return DynamicLLMPathExtractor(
        llm=llm,
        max_triplets_per_chunk=MAX_TRIPLETS,
        num_workers=1,
        allowed_entity_types=None,
        allowed_relation_types=None,
        allowed_relation_props=None,
        allowed_entity_props=None,
    )


def get_yandex_tokens_count(text: str) -> int:
    """Точный подсчет токенов через API Яндекса."""
    try:
        if FOLDER_ID is None or API_KEY is None:
            raise NotImplementedError
        sdk = AIStudio(folder_id=FOLDER_ID, auth=API_KEY)
        # Вызываем встроенный легковесный метод подсчета токенов Яндекса
        # Используем имя любой стандартной текстовой модели, например yandexgpt
        result = sdk.models.completions("yandexgpt").tokenize(text)
        return len(result)
    except Exception:
        # Резервный грубый подсчет на случай сбоя сети (примерно 1 слово = 3-4 токена)
        return len(text.split()) * 4


def _split(documents: List[Document]) -> list:
    from tqdm import tqdm

    splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=CHUNK_OVERLAP)
    nodes = []
    for doc in tqdm(documents, desc="Нарезка", unit="doc", leave=False):
        nodes.extend(splitter.get_nodes_from_documents([doc]))
    return nodes


def build_all_indexes_streaming(
    data_dir: str,
    graph_dir: Path = GRAPH_PERSIST_DIR,
    vector_dir: Path = VECTOR_PERSIST_DIR,
) -> tuple[PropertyGraphIndex, VectorStoreIndex]:
    from src.processing.loader import iter_document_batches

    llm = _init()
    kg_extractor = _extractor(llm)
    graph = PropertyGraphIndex(
        nodes=[],
        llm=llm,
        embed_kg_nodes=EMBED_KG_NODES,
        kg_extractors=[kg_extractor],
        show_progress=False,
    )
    vector = VectorStoreIndex(nodes=[], show_progress=False)

    graph_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)
    # Settings.tokenizer = get_yandex_tokens_count
    n = 0
    for documents in iter_document_batches(data_dir, batch_size=5):
        n += 1
        nodes = _split(documents)
        print(f"Батч {n}: {len(documents)} док, {len(nodes)} nodes")

        # Обрабатываем 757 нод микро-порциями, чтобы видеть реальный прогресс
        print(
            f" Начинаем извлечение трилеров через yandex gpt порциями по {MICRO_BATCH_SIZE}..."
        )

        # Оборачиваем внутреннюю обработку в tqdm, чтобы видеть, как двигаются эти 757 нод

        for j in tqdm(
            range(0, len(nodes), MICRO_BATCH_SIZE), desc=f"Разбор чанков батча {n}"
        ):
            micro_nodes = nodes[j : j + MICRO_BATCH_SIZE]

            # 1. СТАБИЛЬНАЯ ОТПРАВКА В ГРАФ С ЗАЩИТОЙ ОТ ПАДЕНИЯ ПО КВОТАМ
            for single_node in micro_nodes:
                success = False
                retries = 0

                while not success:
                    try:
                        graph.insert_nodes(
                            [single_node],
                            kg_extractors=[kg_extractor],
                            embed_kg_nodes=False,
                            transformations=[],
                        )
                        success = True

                        # Маленькая микро-пауза (0.3 сек) защитит от мгновенного всплеска RPS
                        time.sleep(0.3)

                    except Exception as e:
                        retries += 1
                        # Если поймали ошибку сессии или лимитов токенов Яндекса
                        print(
                            f"\n[Внимание] Яндекс перегружен или исчерпан минутный лимит TPM. Ошибка: {e}"
                        )
                        wait_time = min(30 * retries, 60)  # Спим 30 секунд, затем 60
                        print(
                            f"Ожидаем {wait_time} сек. для сброса лимитов токенов (Попытка {retries})..."
                        )
                        time.sleep(wait_time)

            # 2. ОТПРАВКА В ВЕКТОРНЫЙ ИНДЕКС
            for single_node in micro_nodes:
                success_vector = False
                while not success_vector:
                    try:
                        vector.insert_nodes([single_node], transformations=[])
                        success_vector = True
                        time.sleep(0.1)
                    except Exception as e:
                        print(
                            f"\n[Внимание] Ошибка эмбеддингов Яндекса: {e}. Ждем 30 сек..."
                        )
                        time.sleep(30)

        # Синхронизируем прогресс на диск (база в безопасности)
        graph.storage_context.persist(persist_dir=str(graph_dir))
        vector.storage_context.persist(persist_dir=str(vector_dir))

    del documents, nodes
    gc.collect()
    try:
        graph.property_graph_store.save_networkx_graph(
            name=str(graph_dir / "graph.html")
        )
    except Exception:
        pass

    return graph, vector


def load_graph_index(persist_dir: Path = GRAPH_PERSIST_DIR) -> PropertyGraphIndex:
    _init()
    if not persist_dir.exists():
        raise FileNotFoundError(persist_dir)
    return load_index_from_storage(
        StorageContext.from_defaults(persist_dir=str(persist_dir))
    )


def load_vector_index(persist_dir: Path = VECTOR_PERSIST_DIR) -> VectorStoreIndex:
    _init()
    if not persist_dir.exists():
        raise FileNotFoundError(persist_dir)
    return load_index_from_storage(
        StorageContext.from_defaults(persist_dir=str(persist_dir))
    )


def build_all_indexes(
    documents: List[Document],
) -> tuple[PropertyGraphIndex, VectorStoreIndex]:
    llm = _init()
    nodes = _split(documents)

    graph = PropertyGraphIndex(
        nodes=nodes,
        llm=llm,
        embed_kg_nodes=EMBED_KG_NODES,
        kg_extractors=[_extractor(llm)],
        show_progress=True,
    )
    graph.storage_context.persist(persist_dir=str(GRAPH_PERSIST_DIR))

    vector = VectorStoreIndex(nodes=nodes, show_progress=True)
    vector.storage_context.persist(persist_dir=str(VECTOR_PERSIST_DIR))

    return graph, vector

