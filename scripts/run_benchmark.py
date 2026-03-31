#!/usr/bin/env python3
"""RAG 검색 벤치마크 — Recall@K, MRR, nDCG 자동 평가 + 통계 검증

사용법:
    # 기존 인덱싱된 케이스로 벤치마크 (스마트 청킹)
    python scripts/run_benchmark.py --case-id <CASE_ID> --ground-truth data/benchmark/ground_truth_sample.json

    # 스마트 vs 고정 크기 청킹 비교 벤치마크
    python scripts/run_benchmark.py --case-id <CASE_ID> --ground-truth data/benchmark/ground_truth.json --chunking both

    # 고정 크기 청킹만
    python scripts/run_benchmark.py --case-id <CASE_ID> --ground-truth data/benchmark/ground_truth.json --chunking fixed

Ground Truth JSON 포맷 (v2):
    {
      "questions": [
        {
          "query_id": "q001",
          "query_text": "검색 질의",
          "query_type": "T1",
          "relevant_chunk_ids": ["chunk_id_1", "chunk_id_2"],
          "relevance_grades": {"chunk_id_1": 3, "chunk_id_2": 2},
          "relevant_docs": [{"filename": "file.pdf", "source_type": "document"}],
          "relevant_keywords": ["키워드1", "키워드2"]
        }
      ]
    }

    query_type:
      T1 — 단순 사실 조회 (factual lookup)
      T2 — 참여자/발신자 필터링 (participant filtering)
      T3 — 날짜/기간 필터링 (temporal filtering)
      T4 — 멀티소스 종합 (cross-source synthesis)
      T5 — 요약/비교/타임라인 (summarization & comparison)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# 프로젝트 루트를 PYTHONPATH에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cases.case_store import CaseNotFoundError, CaseStatus, CaseStore
from src.chunkers.chunker import Chunk, DocumentChunker, FixedSizeChunker
from src.chunkers.metadata_enricher import enrich_chunks
from src.indexing.pipeline import IndexingPipeline
from src.parsers.document_parser import DocumentParser
from src.utils.logger import get_logger
from src.vectorstore.vector_store import VectorStoreService

logger = get_logger(__name__)

# query_type 레이블
QUERY_TYPE_LABELS: dict[str, str] = {
    "T1": "Factual Lookup",
    "T2": "Participant Filter",
    "T3": "Temporal Filter",
    "T4": "Cross-Source",
    "T5": "Summarization",
}


# === 데이터 모델 ===


@dataclass
class QuestionResult:
    """단일 질의 평가 결과"""

    question_id: str
    question: str
    question_type: str
    search_method: str
    chunking_method: str
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    retrieved_count: int
    relevant_found: int
    elapsed_ms: float


@dataclass
class BenchmarkResult:
    """전체 벤치마크 결과"""

    case_id: str
    chunking_method: str
    total_questions: int
    total_chunks: int
    results: list[QuestionResult] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""

    @property
    def avg_recall_at_5(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.recall_at_5 for r in self.results) / len(self.results)

    @property
    def avg_recall_at_10(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.recall_at_10 for r in self.results) / len(self.results)

    @property
    def avg_mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.mrr for r in self.results) / len(self.results)

    @property
    def avg_ndcg_at_10(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.ndcg_at_10 for r in self.results) / len(self.results)

    def by_type(self) -> dict[str, dict[str, float]]:
        """질문 유형별 breakdown"""
        type_groups: dict[str, list[QuestionResult]] = {}
        for r in self.results:
            type_groups.setdefault(r.question_type, []).append(r)

        breakdown: dict[str, dict[str, float]] = {}
        for qtype, items in sorted(type_groups.items()):
            n = len(items)
            breakdown[qtype] = {
                "count": n,
                "avg_recall@5": round(sum(r.recall_at_5 for r in items) / n, 4),
                "avg_recall@10": round(sum(r.recall_at_10 for r in items) / n, 4),
                "avg_mrr": round(sum(r.mrr for r in items) / n, 4),
                "avg_ndcg@10": round(sum(r.ndcg_at_10 for r in items) / n, 4),
            }
        return breakdown

    def by_method(self) -> dict[str, dict[str, float]]:
        """검색 방법별 breakdown"""
        method_groups: dict[str, list[QuestionResult]] = {}
        for r in self.results:
            method_groups.setdefault(r.search_method, []).append(r)

        breakdown: dict[str, dict[str, float]] = {}
        for method, items in sorted(method_groups.items()):
            n = len(items)
            breakdown[method] = {
                "count": n,
                "avg_recall@5": round(sum(r.recall_at_5 for r in items) / n, 4),
                "avg_recall@10": round(sum(r.recall_at_10 for r in items) / n, 4),
                "avg_mrr": round(sum(r.mrr for r in items) / n, 4),
                "avg_ndcg@10": round(sum(r.ndcg_at_10 for r in items) / n, 4),
            }
        return breakdown

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "chunking_method": self.chunking_method,
            "total_questions": self.total_questions,
            "total_chunks": self.total_chunks,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "summary": {
                "avg_recall@5": round(self.avg_recall_at_5, 4),
                "avg_recall@10": round(self.avg_recall_at_10, 4),
                "avg_mrr": round(self.avg_mrr, 4),
                "avg_ndcg@10": round(self.avg_ndcg_at_10, 4),
            },
            "by_search_method": self.by_method(),
            "by_question_type": self.by_type(),
            "details": [
                {
                    "question_id": r.question_id,
                    "question": r.question,
                    "question_type": r.question_type,
                    "search_method": r.search_method,
                    "recall@5": r.recall_at_5,
                    "recall@10": r.recall_at_10,
                    "mrr": r.mrr,
                    "ndcg@10": r.ndcg_at_10,
                    "retrieved": r.retrieved_count,
                    "relevant_found": r.relevant_found,
                    "elapsed_ms": round(r.elapsed_ms, 1),
                }
                for r in self.results
            ],
        }


# === Ground Truth 호환 레이어 ===


def normalize_question(raw: dict[str, Any]) -> dict[str, Any]:
    """v1/v2 Ground Truth 포맷을 통합 내부 형식으로 변환

    v2 필드(query_id, query_text, query_type, relevant_chunk_ids, relevance_grades)를
    우선 사용하고, 없으면 v1 필드(id, question, type)로 fallback.
    """
    return {
        "id": raw.get("query_id") or raw.get("id", ""),
        "question": raw.get("query_text") or raw.get("question", ""),
        "type": raw.get("query_type") or raw.get("type", "unknown"),
        "relevant_chunk_ids": raw.get("relevant_chunk_ids", []),
        "relevance_grades": raw.get("relevance_grades", {}),
        "relevant_docs": raw.get("relevant_docs", []),
        "relevant_keywords": raw.get("relevant_keywords", []),
    }


# === nDCG 계산 ===


def _dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at k"""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)  # log2(rank+1), rank는 1-based
    return dcg


def _ndcg_at_k(relevances: list[float], ideal_relevances: list[float], k: int) -> float:
    """Normalized DCG at k"""
    dcg = _dcg(relevances, k)
    ideal = _dcg(sorted(ideal_relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg / ideal


# === 평가 로직 ===


def _is_relevant(result: dict[str, Any], relevant_doc: dict[str, str]) -> bool:
    """검색 결과가 ground truth 문서와 매칭되는지 판단

    filename 부분 매칭 + source_type 매칭으로 판단.
    relevant_keywords가 있으면 content에 키워드 포함 여부도 확인.
    """
    meta = result.get("metadata", {})
    content = result.get("content", "").lower()

    # filename 매칭 (부분 매칭)
    expected_filename = relevant_doc.get("filename", "").lower()
    actual_filename = str(meta.get("filename", "")).lower()
    actual_pst = str(meta.get("pst_file", "")).lower()

    filename_match = False
    if expected_filename:
        filename_match = (
            expected_filename in actual_filename
            or expected_filename in actual_pst
            or actual_filename in expected_filename
        )

    # source_type 매칭
    expected_type = relevant_doc.get("source_type", "")
    actual_type = str(meta.get("source_type", ""))
    type_match = not expected_type or expected_type == actual_type

    # participant 매칭 (있으면)
    expected_participant = relevant_doc.get("participant", "").lower()
    if expected_participant:
        participants_raw = meta.get("participants", "[]")
        if isinstance(participants_raw, str):
            participants_str = participants_raw.lower()
        else:
            participants_str = str(participants_raw).lower()
        if expected_participant not in participants_str and expected_participant not in content:
            return False

    # subject 매칭 (있으면)
    expected_subject = relevant_doc.get("subject", "").lower()
    if expected_subject:
        actual_subject = str(meta.get("thread_subject", "")).lower()
        if expected_subject not in actual_subject and expected_subject not in content:
            return False

    return filename_match and type_match


def _is_keyword_relevant(result: dict[str, Any], keywords: list[str]) -> bool:
    """키워드 기반 관련성 판단 (fallback)

    relevant_docs 매칭이 안 될 때 키워드로 판단.
    """
    if not keywords:
        return False

    content = result.get("content", "").lower()
    meta = result.get("metadata", {})

    # 메타데이터의 topics도 확인
    topics_raw = meta.get("topics", "[]")
    if isinstance(topics_raw, str):
        topics_str = topics_raw.lower()
    else:
        topics_str = str(topics_raw).lower()

    combined = content + " " + topics_str
    matched = sum(1 for kw in keywords if kw.lower() in combined)

    # 키워드의 절반 이상 매칭되면 관련 있다고 판단
    return matched >= max(1, len(keywords) // 2)


def _get_chunk_relevance_grade(
    result: dict[str, Any],
    question: dict[str, Any],
) -> float:
    """검색 결과의 relevance grade를 반환 (0~3)

    relevant_chunk_ids + relevance_grades가 있으면 chunk_id 매칭으로 판단.
    없으면 doc/keyword 매칭 기반으로 binary(0 or 1) 반환.
    """
    chunk_ids = question.get("relevant_chunk_ids", [])
    grades = question.get("relevance_grades", {})
    meta = result.get("metadata", {})

    # v2: chunk_id 기반 매칭
    if chunk_ids and grades:
        result_chunk_id = meta.get("chunk_id", "")
        if result_chunk_id in grades:
            return float(grades[result_chunk_id])

    # fallback: doc/keyword 기반 binary
    relevant_docs = question.get("relevant_docs", [])
    keywords = question.get("relevant_keywords", [])

    is_rel = any(_is_relevant(result, doc) for doc in relevant_docs)
    if not is_rel and keywords:
        is_rel = _is_keyword_relevant(result, keywords)

    return 1.0 if is_rel else 0.0


def evaluate_question(
    question: dict[str, Any],
    search_results: list[dict[str, Any]],
    search_method: str,
    chunking_method: str,
    elapsed_ms: float,
) -> QuestionResult:
    """단일 질의에 대한 Recall@K, MRR, nDCG@10 계산"""
    relevant_docs = question.get("relevant_docs", [])
    keywords = question.get("relevant_keywords", [])
    chunk_ids = question.get("relevant_chunk_ids", [])
    grades = question.get("relevance_grades", {})

    # 관련 문서 수 (chunk_ids가 있으면 우선)
    n_relevant = len(chunk_ids) if chunk_ids else max(len(relevant_docs), 1)

    # 각 결과의 관련성 및 grade 계산
    relevance_flags: list[bool] = []
    relevance_scores: list[float] = []

    for result in search_results:
        grade = _get_chunk_relevance_grade(result, question)
        relevance_scores.append(grade)
        relevance_flags.append(grade > 0)

    # Recall@K
    relevant_in_5 = sum(relevance_flags[:5])
    relevant_in_10 = sum(relevance_flags[:10])
    recall_at_5 = min(relevant_in_5 / n_relevant, 1.0)
    recall_at_10 = min(relevant_in_10 / n_relevant, 1.0)

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, is_rel in enumerate(relevance_flags):
        if is_rel:
            mrr = 1.0 / (i + 1)
            break

    # nDCG@10
    ideal_grades = sorted(grades.values(), reverse=True) if grades else [1.0] * n_relevant
    ideal_float = [float(g) for g in ideal_grades]
    ndcg_at_10 = _ndcg_at_k(relevance_scores, ideal_float, 10)

    return QuestionResult(
        question_id=question["id"],
        question=question["question"],
        question_type=question.get("type", "unknown"),
        search_method=search_method,
        chunking_method=chunking_method,
        recall_at_5=round(recall_at_5, 4),
        recall_at_10=round(recall_at_10, 4),
        mrr=round(mrr, 4),
        ndcg_at_10=round(ndcg_at_10, 4),
        retrieved_count=len(search_results),
        relevant_found=sum(relevance_flags),
        elapsed_ms=elapsed_ms,
    )


# === 통계 검증 ===


def compute_statistical_tests(
    smart_results: BenchmarkResult,
    fixed_results: BenchmarkResult,
) -> dict[str, Any]:
    """Wilcoxon signed-rank test + Cohen's d 효과 크기 계산

    두 청킹 전략의 paired difference를 검증.
    동일 질의·동일 검색 방법에 대한 메트릭 쌍으로 비교.

    Returns:
        메트릭별 {wilcoxon_stat, p_value, cohens_d, interpretation}
    """
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        logger.warning("scipy 미설치 — 통계 검증 건너뜀 (pip install scipy)")
        return {"error": "scipy not installed"}

    # 동일 (question_id, search_method) 쌍으로 매칭
    smart_by_key: dict[tuple[str, str], QuestionResult] = {}
    for r in smart_results.results:
        smart_by_key[(r.question_id, r.search_method)] = r

    fixed_by_key: dict[tuple[str, str], QuestionResult] = {}
    for r in fixed_results.results:
        fixed_by_key[(r.question_id, r.search_method)] = r

    common_keys = sorted(set(smart_by_key.keys()) & set(fixed_by_key.keys()))
    if len(common_keys) < 5:
        return {"error": f"비교 쌍 부족 ({len(common_keys)}개 < 5)"}

    metrics = ["recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]
    stat_results: dict[str, Any] = {"n_pairs": len(common_keys)}

    for metric in metrics:
        smart_vals = [getattr(smart_by_key[k], metric) for k in common_keys]
        fixed_vals = [getattr(fixed_by_key[k], metric) for k in common_keys]

        diffs = [s - f for s, f in zip(smart_vals, fixed_vals)]

        # Cohen's d
        mean_diff = sum(diffs) / len(diffs)
        std_diff = (sum((d - mean_diff) ** 2 for d in diffs) / max(len(diffs) - 1, 1)) ** 0.5
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

        # 효과 크기 해석
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect = "negligible"
        elif abs_d < 0.5:
            effect = "small"
        elif abs_d < 0.8:
            effect = "medium"
        else:
            effect = "large"

        # Wilcoxon signed-rank test (차이가 모두 0이면 건너뜀)
        non_zero_diffs = [d for d in diffs if d != 0]
        if len(non_zero_diffs) < 2:
            stat_results[metric] = {
                "wilcoxon_stat": None,
                "p_value": 1.0,
                "cohens_d": round(cohens_d, 4),
                "effect_size": effect,
                "note": "모든 쌍이 동일 — 검정 불가",
                "smart_mean": round(sum(smart_vals) / len(smart_vals), 4),
                "fixed_mean": round(sum(fixed_vals) / len(fixed_vals), 4),
            }
            continue

        try:
            w_stat, p_value = scipy_stats.wilcoxon(smart_vals, fixed_vals)
        except Exception as e:
            stat_results[metric] = {"error": str(e)}
            continue

        stat_results[metric] = {
            "wilcoxon_stat": round(float(w_stat), 4),
            "p_value": round(float(p_value), 6),
            "significant": p_value < 0.05,
            "cohens_d": round(cohens_d, 4),
            "effect_size": effect,
            "smart_mean": round(sum(smart_vals) / len(smart_vals), 4),
            "fixed_mean": round(sum(fixed_vals) / len(fixed_vals), 4),
        }

    return stat_results


# === 인덱싱 (고정 크기 청킹) ===


# 파일 확장자별 source_type 매핑
_EXTENSION_SOURCE_TYPE: dict[str, str] = {
    ".pst": "email",
    ".ost": "email",
    ".eml": "email",
    ".msg": "email",
    ".pdf": "document",
    ".docx": "document",
    ".doc": "document",
    ".pptx": "document",
    ".ppt": "document",
    ".xlsx": "document",
    ".xls": "document",
    ".txt": "document",
    ".csv": "document",
}


def _infer_source_type(file_path: Path) -> str:
    """파일 확장자에서 source_type 추론"""
    return _EXTENSION_SOURCE_TYPE.get(file_path.suffix.lower(), "document")


def index_with_fixed_chunker(
    case_id: str,
    collection_name: str,
) -> int:
    """기존 케이스의 파일을 FixedSizeChunker로 재인덱싱

    원본 source_type과 메타데이터를 보존한 채 텍스트만 고정 크기로 분할.
    별도 컬렉션에 저장하여 스마트 청킹과 비교 가능.

    Returns:
        저장된 청크 수
    """
    store = CaseStore()
    case_meta = store.get(case_id)

    # 파일 수집 (IndexingPipeline의 로직 재사용)
    pipeline = IndexingPipeline()
    files = pipeline._collect_files(case_meta)

    if not files:
        logger.warning("인덱싱할 파일이 없습니다")
        return 0

    # 파싱 → 텍스트 추출
    doc_parser = DocumentParser()
    fixed_chunker = FixedSizeChunker(chunk_size=512, chunk_overlap=128)
    all_chunks: list[Chunk] = []

    for file_path in files:
        try:
            source_type = _infer_source_type(file_path)

            if file_path.suffix.lower() in (".pst", ".ost"):
                # PST: 이메일은 email, 채팅은 teams_chat으로 구분
                from src.parsers.pst_parser import PSTParser

                result = PSTParser(file_path).parse()

                # 이메일 → source_type="email"
                for email in result.emails:
                    text = f"From: {email.sender}\nSubject: {email.subject}\n\n{email.body}"
                    chunks = fixed_chunker.chunk(
                        text,
                        metadata={
                            "filename": file_path.name,
                            "sender": email.sender,
                            "subject": email.subject,
                        },
                        source_type="email",
                    )
                    all_chunks.extend(chunks)

                    # 첨부파일 → source_type="attachment"
                    if hasattr(email, "attachments") and email.attachments:
                        for att in email.attachments:
                            try:
                                parsed = doc_parser.parse_bytes(att.content, att.filename)
                                docs = parsed if isinstance(parsed, list) else [parsed]
                                for doc in docs:
                                    att_chunks = fixed_chunker.chunk(
                                        doc.content,
                                        metadata={
                                            "filename": att.filename,
                                            "attachment_filename": att.filename,
                                            **doc.metadata,
                                        },
                                        source_type="attachment",
                                    )
                                    all_chunks.extend(att_chunks)
                            except Exception:
                                pass

                # 채팅 → source_type="teams_chat"
                for chat in result.chats:
                    text = f"{chat.sender}: {chat.body}"
                    chunks = fixed_chunker.chunk(
                        text,
                        metadata={
                            "filename": file_path.name,
                            "sender": chat.sender,
                        },
                        source_type="teams_chat",
                    )
                    all_chunks.extend(chunks)

            else:
                parsed = doc_parser.parse(file_path)
                docs = parsed if isinstance(parsed, list) else [parsed]
                for doc in docs:
                    chunks = fixed_chunker.chunk(
                        doc.content,
                        metadata={"filename": file_path.name, **doc.metadata},
                        source_type=source_type,
                    )
                    all_chunks.extend(chunks)
        except Exception as e:
            logger.warning(f"파일 처리 실패 ({file_path.name}): {e}")

    # 메타데이터 보강
    if all_chunks:
        enrich_chunks(all_chunks, case_id=case_id)

    # 벡터 저장 (별도 컬렉션)
    vector_store = VectorStoreService(collection_name=collection_name)
    stored = 0
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        try:
            stored += vector_store.add_chunks(batch)
        except Exception as e:
            logger.warning(f"벡터 저장 실패 (batch {i // batch_size}): {e}")

    logger.info(f"고정 크기 인덱싱 완료: {stored}개 청크 → {collection_name}")
    return stored


# === 벤치마크 실행 ===


def run_benchmark(
    case_id: str,
    ground_truth: dict[str, Any],
    collection_name: str,
    chunking_method: str,
    search_methods: list[str] | None = None,
    n_results: int = 10,
) -> BenchmarkResult:
    """단일 컬렉션에 대해 벤치마크 실행

    Args:
        case_id: 케이스 ID
        ground_truth: Ground Truth JSON
        collection_name: ChromaDB 컬렉션 이름
        chunking_method: "smart" 또는 "fixed"
        search_methods: 테스트할 검색 방법 (기본: vector, bm25, hybrid)
        n_results: 검색 결과 수

    Returns:
        BenchmarkResult
    """
    if search_methods is None:
        search_methods = ["vector", "bm25", "hybrid"]

    raw_questions = ground_truth.get("questions", [])
    questions = [normalize_question(q) for q in raw_questions]
    store = VectorStoreService(collection_name=collection_name)
    stats = store.get_stats()

    result = BenchmarkResult(
        case_id=case_id,
        chunking_method=chunking_method,
        total_questions=len(questions),
        total_chunks=stats["total_chunks"],
        started_at=datetime.now().isoformat(),
    )

    print(f"\n{'='*60}")
    print(f"  Benchmark: {chunking_method} chunking | {collection_name}")
    print(f"  Chunks: {stats['total_chunks']} | Questions: {len(questions)}")
    print(f"  Search methods: {', '.join(search_methods)}")
    print(f"{'='*60}\n")

    for q_idx, question in enumerate(questions, 1):
        for method in search_methods:
            start = time.perf_counter()
            try:
                search_results = store.search(
                    query=question["question"],
                    n_results=n_results,
                    search_method=method,
                )
            except Exception as e:
                logger.error(f"검색 실패 ({method}): {e}")
                search_results = []

            elapsed_ms = (time.perf_counter() - start) * 1000

            qr = evaluate_question(
                question=question,
                search_results=search_results,
                search_method=method,
                chunking_method=chunking_method,
                elapsed_ms=elapsed_ms,
            )
            result.results.append(qr)

            status = "HIT" if qr.mrr > 0 else "MISS"
            print(
                f"  [{q_idx:2d}/{len(questions)}] {method:7s} | "
                f"R@5={qr.recall_at_5:.2f} R@10={qr.recall_at_10:.2f} "
                f"MRR={qr.mrr:.2f} nDCG={qr.ndcg_at_10:.2f} | "
                f"{elapsed_ms:6.0f}ms | {status}"
            )

    result.completed_at = datetime.now().isoformat()
    return result


# === 결과 저장 ===


def save_results(
    results: list[BenchmarkResult],
    output_dir: Path,
    stat_tests: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path | None]:
    """벤치마크 결과를 JSON + CSV + LaTeX로 저장

    Returns:
        (json_path, csv_path, latex_path or None)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    json_path = output_dir / f"benchmark_{timestamp}.json"
    json_data: dict[str, Any] = {
        "benchmark_run": timestamp,
        "results": [r.to_dict() for r in results],
    }
    if stat_tests:
        json_data["statistical_tests"] = stat_tests
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV (상세 결과)
    csv_path = output_dir / f"benchmark_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "chunking", "search_method", "question_id", "question_type",
            "recall@5", "recall@10", "mrr", "ndcg@10",
            "retrieved", "relevant_found", "elapsed_ms",
        ])
        for r in results:
            for qr in r.results:
                writer.writerow([
                    qr.chunking_method, qr.search_method, qr.question_id,
                    qr.question_type, qr.recall_at_5, qr.recall_at_10,
                    qr.mrr, qr.ndcg_at_10, qr.retrieved_count, qr.relevant_found,
                    round(qr.elapsed_ms, 1),
                ])

    # LaTeX (비교 모드일 때)
    latex_path = None
    if len(results) >= 2:
        latex_path = output_dir / f"benchmark_{timestamp}.tex"
        latex_content = generate_latex_tables(results, stat_tests)
        latex_path.write_text(latex_content, encoding="utf-8")

    return json_path, csv_path, latex_path


# === LaTeX 테이블 생성 ===


def generate_latex_tables(
    results: list[BenchmarkResult],
    stat_tests: dict[str, Any] | None = None,
) -> str:
    """Paper-ready LaTeX 테이블 생성

    Table 1: 청킹전략 × 검색방법 비교 (6행: 2 chunking × 3 retrieval)
    Table 2: query_type별 breakdown (T1~T5 × 검색방법)
    Table 3: 통계 검증 결과 (있으면)
    """
    lines: list[str] = []
    lines.append("% Auto-generated by AST PoC Benchmark Runner")
    lines.append(f"% Generated at: {datetime.now().isoformat()}")
    lines.append("")

    # Table 1: 청킹 × 검색방법 비교
    lines.append("% === Table 1: Chunking Strategy × Retrieval Method ===")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Retrieval performance by chunking strategy and search method}")
    lines.append("\\label{tab:chunking-comparison}")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("Chunking & Method & Recall@5 & Recall@10 & MRR & nDCG@10 \\\\")
    lines.append("\\midrule")

    for r in results:
        by_method = r.by_method()
        first = True
        for method, stats in sorted(by_method.items()):
            chunking_label = r.chunking_method.capitalize() if first else ""
            lines.append(
                f"{chunking_label} & {method} & "
                f"{stats['avg_recall@5']:.4f} & {stats['avg_recall@10']:.4f} & "
                f"{stats['avg_mrr']:.4f} & {stats['avg_ndcg@10']:.4f} \\\\"
            )
            first = False
        if r != results[-1]:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Table 2: query_type별 breakdown (hybrid 기준)
    lines.append("% === Table 2: Performance by Query Type (Hybrid Search) ===")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Retrieval performance by query type (hybrid search)}")
    lines.append("\\label{tab:query-type-breakdown}")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("Type & Description & \\multicolumn{2}{c}{Smart} & \\multicolumn{2}{c}{Fixed} \\\\")
    lines.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}")
    lines.append(" & & MRR & nDCG@10 & MRR & nDCG@10 \\\\")
    lines.append("\\midrule")

    # hybrid 결과만 추출하여 query_type별 비교
    for qtype in ["T1", "T2", "T3", "T4", "T5"]:
        label = QUERY_TYPE_LABELS.get(qtype, qtype)
        smart_vals = {"avg_mrr": "-", "avg_ndcg@10": "-"}
        fixed_vals = {"avg_mrr": "-", "avg_ndcg@10": "-"}

        for r in results:
            # hybrid 결과만 필터링하여 query_type 계산
            hybrid_items = [
                qr for qr in r.results
                if qr.search_method == "hybrid" and qr.question_type == qtype
            ]
            if hybrid_items:
                n = len(hybrid_items)
                avg_mrr = sum(qr.mrr for qr in hybrid_items) / n
                avg_ndcg = sum(qr.ndcg_at_10 for qr in hybrid_items) / n
                vals = {"avg_mrr": f"{avg_mrr:.4f}", "avg_ndcg@10": f"{avg_ndcg:.4f}"}
                if r.chunking_method == "smart":
                    smart_vals = vals
                else:
                    fixed_vals = vals

        lines.append(
            f"{qtype} & {label} & "
            f"{smart_vals['avg_mrr']} & {smart_vals['avg_ndcg@10']} & "
            f"{fixed_vals['avg_mrr']} & {fixed_vals['avg_ndcg@10']} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # Table 3: 통계 검증 (있으면)
    if stat_tests and "error" not in stat_tests:
        lines.append("% === Table 3: Statistical Significance Tests ===")
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Statistical comparison: Smart vs. Fixed chunking (Wilcoxon signed-rank test)}")
        lines.append("\\label{tab:statistical-tests}")
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\toprule")
        lines.append("Metric & $W$ & $p$-value & Cohen's $d$ & Effect \\\\")
        lines.append("\\midrule")

        metric_labels = {
            "recall_at_5": "Recall@5",
            "recall_at_10": "Recall@10",
            "mrr": "MRR",
            "ndcg_at_10": "nDCG@10",
        }

        for metric_key, label in metric_labels.items():
            if metric_key in stat_tests and "error" not in stat_tests[metric_key]:
                s = stat_tests[metric_key]
                w = s.get("wilcoxon_stat")
                p = s.get("p_value", 1.0)
                d = s.get("cohens_d", 0.0)
                effect = s.get("effect_size", "n/a")

                w_str = f"{w:.2f}" if w is not None else "—"
                p_str = f"{p:.4f}"
                if p < 0.001:
                    p_str = "$<$0.001"
                sig = "*" if p < 0.05 else ""

                lines.append(
                    f"{label} & {w_str} & {p_str}{sig} & {d:.3f} & {effect} \\\\"
                )

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        n_pairs = stat_tests.get("n_pairs", "?")
        lines.append(f"\\\\\\footnotesize{{$n={n_pairs}$ paired observations. * $p<0.05$}}")
        lines.append("\\end{table}")
        lines.append("")

    return "\n".join(lines)


# === 콘솔 출력 ===


def print_summary(results: list[BenchmarkResult]) -> None:
    """콘솔에 요약 출력"""
    for r in results:
        print(f"\n{'='*60}")
        print(f"  {r.chunking_method.upper()} CHUNKING | {r.total_chunks} chunks")
        print(f"{'='*60}")

        print(f"\n  Overall:  Recall@5={r.avg_recall_at_5:.4f}  "
              f"Recall@10={r.avg_recall_at_10:.4f}  "
              f"MRR={r.avg_mrr:.4f}  nDCG@10={r.avg_ndcg_at_10:.4f}")

        print(f"\n  By search method:")
        for method, stats in r.by_method().items():
            print(f"    {method:7s} — R@5={stats['avg_recall@5']:.4f}  "
                  f"R@10={stats['avg_recall@10']:.4f}  "
                  f"MRR={stats['avg_mrr']:.4f}  nDCG={stats['avg_ndcg@10']:.4f}  "
                  f"(n={int(stats['count'])})")

        print(f"\n  By query type:")
        for qtype, stats in r.by_type().items():
            label = QUERY_TYPE_LABELS.get(qtype, qtype)
            print(f"    {qtype} {label:20s} — R@5={stats['avg_recall@5']:.4f}  "
                  f"R@10={stats['avg_recall@10']:.4f}  "
                  f"MRR={stats['avg_mrr']:.4f}  nDCG={stats['avg_ndcg@10']:.4f}  "
                  f"(n={int(stats['count'])})")


def print_paper_table(results: list[BenchmarkResult]) -> None:
    """Paper-ready 6행 비교 테이블 콘솔 출력"""
    if len(results) < 2:
        return

    print(f"\n{'='*78}")
    print("  PAPER TABLE: Chunking × Retrieval Method Comparison")
    print(f"{'='*78}")
    print(f"  {'Chunking':<10} {'Method':<8} {'Recall@5':>10} {'Recall@10':>10} "
          f"{'MRR':>8} {'nDCG@10':>10}")
    print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    for r in results:
        by_method = r.by_method()
        for method, stats in sorted(by_method.items()):
            print(f"  {r.chunking_method:<10} {method:<8} "
                  f"{stats['avg_recall@5']:>10.4f} {stats['avg_recall@10']:>10.4f} "
                  f"{stats['avg_mrr']:>8.4f} {stats['avg_ndcg@10']:>10.4f}")

    print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")


def print_statistical_summary(stat_tests: dict[str, Any]) -> None:
    """통계 검증 결과 콘솔 출력"""
    if "error" in stat_tests:
        print(f"\n  Statistical tests: {stat_tests['error']}")
        return

    print(f"\n{'='*78}")
    print("  STATISTICAL TESTS: Smart vs Fixed (Wilcoxon signed-rank)")
    print(f"  Paired observations: n={stat_tests.get('n_pairs', '?')}")
    print(f"{'='*78}")
    print(f"  {'Metric':<12} {'W':>8} {'p-value':>10} {'Cohen d':>10} {'Effect':>12} {'Sig?':>6}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*6}")

    metric_labels = {
        "recall_at_5": "Recall@5",
        "recall_at_10": "Recall@10",
        "mrr": "MRR",
        "ndcg_at_10": "nDCG@10",
    }

    for key, label in metric_labels.items():
        if key in stat_tests and isinstance(stat_tests[key], dict) and "error" not in stat_tests[key]:
            s = stat_tests[key]
            w = s.get("wilcoxon_stat")
            w_str = f"{w:.2f}" if w is not None else "—"
            p = s.get("p_value", 1.0)
            d = s.get("cohens_d", 0.0)
            effect = s.get("effect_size", "n/a")
            sig = "YES *" if s.get("significant") else "no"
            print(f"  {label:<12} {w_str:>8} {p:>10.4f} {d:>10.3f} {effect:>12} {sig:>6}")

    print()


# === CLI ===


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AST PoC RAG Benchmark — Recall@K, MRR, nDCG 자동 평가 + 통계 검증",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--case-id", required=True, help="평가할 케이스 ID")
    parser.add_argument("--ground-truth", required=True, help="Ground Truth JSON 파일 경로")
    parser.add_argument(
        "--chunking",
        choices=["smart", "fixed", "both"],
        default="smart",
        help="청킹 방법 (smart=스마트4종, fixed=고정512토큰, both=비교)",
    )
    parser.add_argument("--n-results", type=int, default=10, help="검색 결과 수 (기본 10)")
    parser.add_argument(
        "--output-dir",
        default="data/benchmark/results",
        help="결과 저장 디렉토리 (기본 data/benchmark/results)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["vector", "bm25", "hybrid"],
        default=["vector", "bm25", "hybrid"],
        help="테스트할 검색 방법",
    )
    parser.add_argument(
        "--no-latex",
        action="store_true",
        help="LaTeX 테이블 생성 건너뜀",
    )

    args = parser.parse_args()

    # Ground Truth 로드
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"ERROR: Ground Truth 파일 없음: {gt_path}")
        sys.exit(1)

    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    questions = ground_truth.get("questions", [])
    if not questions:
        print("ERROR: Ground Truth에 질문이 없습니다")
        sys.exit(1)

    # GT 포맷 감지
    first_q = questions[0]
    gt_version = "v2" if "query_id" in first_q else "v1"
    print(f"Ground Truth: {len(questions)}개 질문 로드 ({gt_path.name}, {gt_version} format)")

    # 케이스 확인
    store = CaseStore()
    try:
        case_meta = store.get(args.case_id)
    except CaseNotFoundError:
        print(f"ERROR: 케이스를 찾을 수 없습니다: {args.case_id}")
        sys.exit(1)

    if case_meta.status not in (CaseStatus.READY, CaseStatus.ARCHIVED):
        print(f"WARNING: 케이스 상태가 {case_meta.status.value}입니다. 인덱싱이 완료된 케이스를 권장합니다.")

    all_results: list[BenchmarkResult] = []

    # 스마트 청킹 벤치마크
    if args.chunking in ("smart", "both"):
        smart_collection = f"case_{args.case_id}"
        result = run_benchmark(
            case_id=args.case_id,
            ground_truth=ground_truth,
            collection_name=smart_collection,
            chunking_method="smart",
            search_methods=args.methods,
            n_results=args.n_results,
        )
        all_results.append(result)

    # 고정 크기 청킹 벤치마크
    if args.chunking in ("fixed", "both"):
        fixed_collection = f"case_{args.case_id}_fixed"

        # 고정 크기로 재인덱싱
        print(f"\n{'='*60}")
        print(f"  Fixed-size re-indexing → {fixed_collection}")
        print(f"{'='*60}")

        chunk_count = index_with_fixed_chunker(
            case_id=args.case_id,
            collection_name=fixed_collection,
        )
        print(f"  Indexed: {chunk_count} chunks\n")

        result = run_benchmark(
            case_id=args.case_id,
            ground_truth=ground_truth,
            collection_name=fixed_collection,
            chunking_method="fixed",
            search_methods=args.methods,
            n_results=args.n_results,
        )
        all_results.append(result)

    # 통계 검증 (both 모드)
    stat_tests: dict[str, Any] | None = None
    if args.chunking == "both" and len(all_results) == 2:
        stat_tests = compute_statistical_tests(all_results[0], all_results[1])

    # 결과 저장
    output_dir = Path(args.output_dir)
    json_path, csv_path, latex_path = save_results(all_results, output_dir, stat_tests)

    # 콘솔 출력
    print_summary(all_results)

    if len(all_results) >= 2:
        print_paper_table(all_results)

    if stat_tests:
        print_statistical_summary(stat_tests)

    print(f"\n{'='*60}")
    print(f"  Results saved:")
    print(f"    JSON:  {json_path}")
    print(f"    CSV:   {csv_path}")
    if latex_path:
        print(f"    LaTeX: {latex_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
