"""
Кластеризация nodes перед построением графа

Идея (community-based graph construction, как в Microsoft GraphRAG):
вместо одного вызова LLM на каждый node, мы группируем семантически близкие
фрагменты и отдаём группу в LLM единым контекстом. Это:
  - сокращает число вызовов LLM в разы
  - даёт лучшие связи

Поток:
    nodes -> эмбеддинги -> HDBSCAN -> группы TextNode -> в экстрактор
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from llama_index.core.schema import BaseNode, TextNode

# ПАРАМЕТРЫ КЛАСТЕРИЗАЦИИ
# минимальный размер кластера
MIN_CLUSTER_SIZE = 5
# максимум nodes в одной группе, идущей в LLM. Ограничение контекста Qwen.
# ~20 фрагментов по 2048 токенов влезает в 32k контекст с запасом
MAX_GROUP_SIZE = 20


@dataclass
class NodeGroup:
    cluster_id: int
    nodes: list[BaseNode]

    @property
    def merged_text(self) -> str:
        """склеивает тексты группы в единый контекст для LLM"""
        return "\n\n---\n\n".join(n.get_content() for n in self.nodes)


# Эмбеддинги nodes
def embed_nodes(
    nodes: list[BaseNode],
    embed_fn: Callable[[list[str]], list[list[float]]],
    batch_size: int = 64,
) -> np.ndarray:
    """
    считает эмбеддинги всех nodes батчами
    возвращает матрицу (n_nodes, dim)
    """
    texts = [n.get_content() for n in nodes]
    vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors.extend(embed_fn(batch))
        print(f"эмбеддинги: {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")

    print()
    return np.array(vectors, dtype=np.float32)


# Кластеризация HDBSCAN
def cluster_nodes(
    nodes: list[BaseNode],
    embeddings: np.ndarray,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
) -> list[NodeGroup]:
    """
    HDBSCAN не требует задавать число кластеров заранее (в отличие от k-means)
    и сам определяет шумовые точки (label = -1). Шумовые nodes мы не выбрасываем,
    а собираем в отдельные группы иначе потеряли бы часть корпуса.

    Большие кластеры бьём на подгруппы <= MAX_GROUP_SIZE, чтобы влезть
    в контекст LLM
    """
    import hdbscan

    # нормализуем векторы, т.к. HDBSCAN на косинусной близости работает лучше
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(normalized)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"HDBSCAN: {n_clusters} кластеров, {n_noise} шумовых nodes")

    # группируем nodes по метке кластера
    by_label = {}
    for node, label in zip(nodes, labels):
        by_label.setdefault(int(label), []).append(node)

    groups = []
    for label, group_nodes in by_label.items():
        # крупные кластеры (и шум) бьём на куски <= MAX_GROUP_SIZE
        for i in range(0, len(group_nodes), MAX_GROUP_SIZE):
            chunk = group_nodes[i:i + MAX_GROUP_SIZE]
            groups.append(NodeGroup(cluster_id=label, nodes=chunk))

    return groups


# Слияние групп в TextNode для экстрактора
def groups_to_nodes(groups: list[NodeGroup]) -> list[TextNode]:
    """
    Превращает каждую группу в один TextNode с объединённым текстом
    Этот node пойдёт в DynamicLLMPathExtractor. Один вызов LLM на группу

    Метаданные: сохраняем список исходных doc_path, чтобы триплеты
    можно было проследить до источников
    """
    result  = []
    for g in groups:
        source_paths = sorted({
            n.metadata.get("doc_path", "?")
            for n in g.nodes
            if n.metadata
        })
        result.append(TextNode(
            text=g.merged_text,
            metadata={
                "cluster_id": g.cluster_id,
                "source_paths": source_paths,
                "group_size": len(g.nodes),
            },
            excluded_embed_metadata_keys=["cluster_id", "source_paths", "group_size"],
            excluded_llm_metadata_keys=["cluster_id", "source_paths", "group_size"],
        ))
    return result


# entrypoint
def cluster_pipeline(
    nodes: list[BaseNode],
    embed_fn: Callable[[list[str]], list[list[float]]],
) -> list[TextNode]:
    """
    Полный проходд

    Возвращает список TextNode (по одному на группу), готовых для
    DynamicLLMPathExtractor.
    """
    print(f"Кластеризация {len(nodes)} nodes...")
    embeddings = embed_nodes(nodes, embed_fn)
    groups = cluster_nodes(nodes, embeddings)
    group_nodes = groups_to_nodes(groups)

    print(f"{len(nodes)} nodes -> {len(group_nodes)} групп "
          f"(сокращение вызовов LLM в {len(nodes) / max(len(group_nodes), 1)}x)")
    return group_nodes