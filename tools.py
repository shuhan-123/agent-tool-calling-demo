"""
两个工具：
1. lookup_metric  - 查表：从 metrics_data.json 里查某个维度的幻觉率
2. retrieve_doc   - 查资料：从论文里检索相关段落（基于 FAISS 向量索引）

每次调用都会记录到 agent_log.txt，方便简历里说明"具备可观测性设计"。
"""

import json
import time
from pathlib import Path
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).parent
METRICS_FILE = BASE_DIR / "metrics_data.json"
INDEX_DIR = BASE_DIR / "faiss_index"
LOG_FILE = BASE_DIR / "agent_log.txt"

_metrics_cache = None
_vectorstore_cache = None


def _log(tool_name: str, query: str, success: bool, elapsed: float):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] tool={tool_name} query={query!r} success={success} elapsed={elapsed:.2f}s\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def _load_metrics():
    global _metrics_cache
    if _metrics_cache is None:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            _metrics_cache = json.load(f)
    return _metrics_cache


def _load_vectorstore():
    global _vectorstore_cache
    if _vectorstore_cache is None:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        _vectorstore_cache = FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
    return _vectorstore_cache


@tool
def lookup_metric(dimension: str) -> str:
    """查询某个评测维度的跨模型平均幻觉率。输入应为以下之一：目标存在性、计数与属性、OCR文本识别、文化常识、语言先验。本工具不包含 Qwen2-VL 2B/7B 等单模型指标；凡是比较具体模型、模型规模或总体表现的问题，都不要使用本工具，应使用 retrieve_doc。"""
    start = time.time()
    data = _load_metrics()
    result = data.get(dimension)
    elapsed = time.time() - start
    if result is None:
        _log("lookup_metric", dimension, False, elapsed)
        return f"未找到维度「{dimension}」，可选维度：{list(data.keys())}"
    _log("lookup_metric", dimension, True, elapsed)
    return f"{dimension}：跨模型平均幻觉率 {result['hallucination_rate']}，说明：{result['description']}。注意：该数据不是 Qwen2-VL 2B/7B 等单模型指标；如需比较具体模型或总体表现，请调用 retrieve_doc 检索论文原文。"


@tool
def retrieve_doc(query: str) -> str:
    """在论文原文中检索与问题相关的段落，用于回答关于研究方法、实验设计、结论细节、原因解释、不同模型对比等问题。凡是问题提到 Qwen2-VL 2B、Qwen2-VL 7B、模型规模、总体表现，都应使用本工具。"""
    start = time.time()
    try:
        vs = _load_vectorstore()
        docs = vs.similarity_search(query, k=3)
        elapsed = time.time() - start
        _log("retrieve_doc", query, True, elapsed)
        content = "\n---\n".join(d.page_content for d in docs)
        return content
    except Exception as e:
        elapsed = time.time() - start
        _log("retrieve_doc", query, False, elapsed)
        return f"检索失败：{e}"
