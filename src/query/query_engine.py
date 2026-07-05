import datetime
import re
from dataclasses import dataclass
from typing import Optional

from llama_index.core import QueryBundle, PropertyGraphIndex, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)

SIMILARITY_TOP_K = 5
GRAPH_HIT_THRESHOLD = 2

_CYR = re.compile(r"[а-яёА-ЯЁ]")
_LAT = re.compile(r"[a-zA-Z]")
_UNITS = r"(мг/л|мг/дм³|°C|м³/ч|т/сут|%|г/л|мг/кг|А/м2|pH)"

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(rf"(\d+(?:[.,]\d+)?)\s*[-–]\s*(\d+(?:[.,]\d+)?)\s*{_UNITS}", re.I),
        "range",
    ),
    (re.compile(rf"[≤<=]\s*(\d+(?:[.,]\d+)?)\s*{_UNITS}", re.I), "max"),
    (re.compile(rf"[≥>=]\s*(\d+(?:[.,]\d+)?)\s*{_UNITS}", re.I), "min"),
    (re.compile(rf"(\d+(?:[.,]\d+)?)\s*{_UNITS}", re.I), "exact"),
]

_PROP_KW = [
    "концентрация",
    "температура",
    "скорость",
    "давление",
    "производительность",
    "содержание",
    "выход",
    "расход",
    "ph",
    "остаток",
    "извлечение",
    "плотность тока",
]


@dataclass(frozen=True)
class NumericConstraint:
    property_name: str
    min_value: Optional[float]
    max_value: Optional[float]
    unit: Optional[str]

    def matches(self, value: float) -> bool:
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


@dataclass(frozen=True)
class ParsedQuery:
    raw_text: str
    language: str = "ru"
    constraints: tuple[NumericConstraint, ...] = ()
    geography_filter: Optional[str] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None


@dataclass(frozen=True)
class Source:
    doc_path: str
    year: Optional[int]
    snippet: str
    score: Optional[float]


@dataclass(frozen=True)
class QueryResult:
    answer: str
    source: str
    sources: list[Source]
    gap_detected: bool
    constraints_applied: list[dict]

    def to_dict(self) -> dict:
        return {
            "ans": self.answer,
            "src": [
                {
                    "doc_path": s.doc_path,
                    "year": s.year,
                    "snippet": s.snippet,
                    "score": s.score,
                }
                for s in self.sources
            ],
        }


def _detect_language(text: str) -> str:
    cyr, lat = len(_CYR.findall(text)), len(_LAT.findall(text))
    total = cyr + lat
    if total == 0:
        return "ru"
    ratio = cyr / total
    return "ru" if ratio > 0.6 else ("en" if ratio < 0.3 else "mixed")


def _find_prop(text: str, pos: int) -> str:
    ctx = text[max(0, pos - 60) : pos].lower()
    for kw in _PROP_KW:
        if kw in ctx:
            return kw
    return "параметр"


def extract_constraints(text: str) -> list[NumericConstraint]:
    out: list[NumericConstraint] = []
    seen: set[str] = set()
    for pattern, kind in _PATTERNS:
        for m in pattern.finditer(text):
            g = m.groups()
            unit = g[-1]
            key = f"{m.group()}:{unit}"
            if key in seen:
                continue
            seen.add(key)
            prop = _find_prop(text, m.start())
            if kind == "range":
                out.append(
                    NumericConstraint(
                        prop,
                        float(g[0].replace(",", ".")),
                        float(g[1].replace(",", ".")),
                        unit,
                    )
                )
            elif kind == "max":
                out.append(
                    NumericConstraint(prop, None, float(g[0].replace(",", ".")), unit)
                )
            elif kind == "min":
                out.append(
                    NumericConstraint(prop, float(g[0].replace(",", ".")), None, unit)
                )
            else:
                v = float(g[0].replace(",", "."))
                out.append(NumericConstraint(prop, v, v, unit))
    return out


def parse_query(raw_text: str) -> ParsedQuery:
    lang = _detect_language(raw_text)
    low = raw_text.lower()

    geo: Optional[str] = None
    if any(k in low for k in ("отечествен", "российск", "в россии", "рф")):
        geo = "RU"
    elif any(
        k in low
        for k in (
            "зарубеж",
            "мировой практик",
            "мировая практик",
            "иностранн",
            "мировой опыт",
            "foreign",
        )
    ):
        geo = "foreign"

    year_from = year_to = None
    if m := re.search(r"(\d{4})\s*[-–]\s*(\d{4})", raw_text):
        year_from, year_to = int(m.group(1)), int(m.group(2))
    elif m := re.search(r"с\s+(\d{4})", raw_text):
        year_from = int(m.group(1))
    elif m := re.search(r"за последни[её]\s+(\d+)\s+лет", raw_text):
        year_to = datetime.date.today().year
        year_from = year_to - int(m.group(1))

    return ParsedQuery(
        raw_text=raw_text,
        language=lang,
        constraints=tuple(extract_constraints(raw_text)),
        geography_filter=geo,
        year_from=year_from,
        year_to=year_to,
    )


def _build_filters(q: ParsedQuery) -> Optional[MetadataFilters]:
    filters: list[MetadataFilter] = []
    if q.year_from:
        filters.append(
            MetadataFilter(key="year", value=q.year_from, operator=FilterOperator.GTE)
        )
    if q.year_to:
        filters.append(
            MetadataFilter(key="year", value=q.year_to, operator=FilterOperator.LTE)
        )
    if q.language in ("ru", "en"):
        filters.append(
            MetadataFilter(key="language", value=q.language, operator=FilterOperator.EQ)
        )
    if not filters:
        return None
    return MetadataFilters(filters=filters, condition=FilterCondition.AND)


def _search_graph(
    graph_index: PropertyGraphIndex, q: ParsedQuery
) -> list[NodeWithScore]:
    from llama_index.core.indices.property_graph import LLMSynonymRetriever

    retriever = graph_index.as_retriever(
        sub_retrievers=[
            LLMSynonymRetriever(graph_index.property_graph_store, include_text=True)
        ],
    )
    return retriever.retrieve(QueryBundle(query_str=q.raw_text))


def _search_vector(
    vector_index: VectorStoreIndex, q: ParsedQuery
) -> list[NodeWithScore]:
    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=SIMILARITY_TOP_K,
        filters=_build_filters(q),
    )
    return retriever.retrieve(QueryBundle(query_str=q.raw_text))


def _filter_by_constraints(
    nodes: list[NodeWithScore], q: ParsedQuery
) -> list[NodeWithScore]:
    if not q.constraints:
        return nodes
    kept: list[NodeWithScore] = []
    for node in nodes:
        node_constraints = extract_constraints(node.node.text)
        if not node_constraints:
            kept.append(node)
            continue
        ok = False
        for nc in node_constraints:
            for qc in q.constraints:
                if nc.unit == qc.unit:
                    val = nc.max_value if nc.max_value is not None else nc.min_value
                    if val is not None and qc.matches(val):
                        ok = True
                        break
            if ok:
                break
        if ok:
            kept.append(node)
    return kept or nodes


def _synthesize(nodes: list[NodeWithScore], q: ParsedQuery, source: str) -> str:
    from llama_index.core.response_synthesizers import get_response_synthesizer

    synthesizer = get_response_synthesizer(response_mode="compact")
    context = "\n\n---\n\n".join(
        f"[Источник {i + 1} | год: {n.node.metadata.get('year', '?')} "
        f"| {n.node.metadata.get('doc_path', '?')}]\n{n.node.text}"
        for i, n in enumerate(nodes)
    )
    prompt = (
        "Ты — аналитик горно-металлургической отрасли. Отвечай точно, "
        "структурированно, со ссылками на источники.\n\n"
        f"Контекст:\n{context}\n\nВопрос: {q.raw_text}\n\n"
        f"(данные получены из: {source}). Если данных мало — укажи это прямо."
    )
    return str(synthesizer.synthesize(query=prompt, nodes=nodes))


def _constraints_dump(q: ParsedQuery) -> list[dict]:
    return [
        {
            "property": c.property_name,
            "min": c.min_value,
            "max": c.max_value,
            "unit": c.unit,
        }
        for c in q.constraints
    ]


def _nodes_to_sources(
    nodes: list[NodeWithScore], max_snippet: int = 300
) -> list[Source]:
    sources: list[Source] = []
    seen: set[str] = set()
    for n in nodes:
        path = n.node.metadata.get("doc_path", "?")
        if path in seen:
            continue
        seen.add(path)
        text = n.node.text or ""
        sources.append(
            Source(
                doc_path=path,
                year=n.node.metadata.get("year"),
                snippet=text[:max_snippet] + ("…" if len(text) > max_snippet else ""),
                score=getattr(n, "score", None),
            )
        )
    return sources


def query(
    raw_text: str,
    graph_index: PropertyGraphIndex,
    vector_index: VectorStoreIndex,
) -> QueryResult:
    q = parse_query(raw_text)

    graph_nodes = _filter_by_constraints(_search_graph(graph_index, q), q)
    if len(graph_nodes) >= GRAPH_HIT_THRESHOLD:
        return QueryResult(
            answer=_synthesize(graph_nodes, q, "граф"),
            source="graph",
            sources=_nodes_to_sources(graph_nodes),
            gap_detected=False,
            constraints_applied=_constraints_dump(q),
        )

    vector_nodes = _filter_by_constraints(_search_vector(vector_index, q), q)
    if vector_nodes:
        return QueryResult(
            answer=_synthesize(vector_nodes, q, "векторный поиск"),
            source="vector",
            sources=_nodes_to_sources(vector_nodes),
            gap_detected=False,
            constraints_applied=_constraints_dump(q),
        )

    return QueryResult(
        answer=(
            f"По запросу «{raw_text}» данные не найдены. Возможно, эта "
            f"комбинация параметров не исследована или отсутствует в корпусе."
        ),
        source="none",
        sources=[],
        gap_detected=True,
        constraints_applied=[],
    )


async def aquery(
    raw_text: str,
    graph_index: PropertyGraphIndex,
    vector_index: VectorStoreIndex,
) -> QueryResult:
    import asyncio

    return await asyncio.to_thread(query, raw_text, graph_index, vector_index)


def load_indexes() -> tuple[PropertyGraphIndex, VectorStoreIndex]:
    from src.graph.store import load_graph_index, load_vector_index

    return load_graph_index(), load_vector_index()

