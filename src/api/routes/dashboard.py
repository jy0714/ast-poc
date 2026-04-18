"""커뮤니케이션 분석 대시보드 엔드포인트 — Analyst UI

- GET /{case_id}  — 케이스 메타데이터 집계 (네트워크, 타임라인, 소스 분포 등)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.cases.case_store import CaseNotFoundError, CaseStatus, CaseStore
from src.chunkers.metadata_enricher import deserialize_metadata_from_chroma
from src.vectorstore.vector_store import VectorStoreService
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

_case_store = CaseStore()


# === 응답 모델 ===


class ParticipantNode(BaseModel):
    """네트워크 그래프 노드"""

    id: str
    message_count: int
    source_types: list[str]


class ParticipantEdge(BaseModel):
    """네트워크 그래프 엣지"""

    source: str
    target: str
    weight: int


class NetworkData(BaseModel):
    """참여자 네트워크"""

    nodes: list[ParticipantNode]
    edges: list[ParticipantEdge]


class TimelinePoint(BaseModel):
    """월별 커뮤니케이션 빈도"""

    month: str
    email_count: int = 0
    teams_chat_count: int = 0
    document_count: int = 0


class TopicItem(BaseModel):
    """토픽 빈도"""

    topic: str
    count: int


class DashboardResponse(BaseModel):
    """대시보드 집계 응답"""

    case_id: str
    case_name: str
    total_chunks: int
    source_type_counts: dict[str, int]
    top_participants: list[ParticipantNode]
    participant_network: NetworkData
    timeline: list[TimelinePoint]
    top_topics: list[TopicItem]


# === 집계 로직 ===


def _parse_month(date_str: str) -> str | None:
    """날짜 문자열에서 YYYY-MM 추출"""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            dt = datetime.strptime(date_str[:26], fmt)
            return dt.strftime("%Y-%m")
        except ValueError:
            continue
    # ISO format with timezone
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.strftime("%Y-%m")
    except (ValueError, TypeError):
        pass
    return None


def aggregate_dashboard(case_id: str, all_metadatas: list[dict[str, Any]]) -> dict[str, Any]:
    """ChromaDB 메타데이터를 대시보드용 집계 데이터로 변환

    Args:
        case_id: 케이스 ID
        all_metadatas: ChromaDB에서 가져온 전체 메타데이터 리스트

    Returns:
        집계 결과 딕셔너리
    """
    source_counts: Counter[str] = Counter()
    participant_counts: Counter[str] = Counter()
    participant_sources: dict[str, set[str]] = defaultdict(set)
    edge_counts: Counter[tuple[str, str]] = Counter()
    month_data: dict[str, Counter[str]] = defaultdict(Counter)
    topic_counts: Counter[str] = Counter()

    for raw_meta in all_metadatas:
        meta = deserialize_metadata_from_chroma(raw_meta)
        source_type = meta.get("source_type", "unknown")
        source_counts[source_type] += 1

        # 참여자 집계 (이메일 + Teams 채팅)
        participants = meta.get("participants", [])
        if isinstance(participants, str):
            try:
                participants = json.loads(participants)
            except (json.JSONDecodeError, ValueError):
                participants = []

        if isinstance(participants, list):
            for p in participants:
                if isinstance(p, str) and p.strip():
                    name = p.strip()
                    participant_counts[name] += 1
                    participant_sources[name].add(source_type)

            # 엣지: 같은 청크에 등장하는 참여자 쌍
            unique_names = sorted(set(
                p.strip() for p in participants
                if isinstance(p, str) and p.strip()
            ))
            for i in range(len(unique_names)):
                for j in range(i + 1, len(unique_names)):
                    edge = (unique_names[i], unique_names[j])
                    edge_counts[edge] += 1

        # 타임라인 집계
        date_str = meta.get("date_range_start") or meta.get("date") or ""
        month = _parse_month(str(date_str))
        if month:
            month_data[month][source_type] += 1

        # 토픽 집계
        topics = meta.get("topics", [])
        if isinstance(topics, str):
            try:
                topics = json.loads(topics)
            except (json.JSONDecodeError, ValueError):
                topics = []
        if isinstance(topics, list):
            for t in topics:
                if isinstance(t, str) and t.strip():
                    topic_counts[t.strip()] += 1

    # 네트워크 노드 (Top 30)
    top_30_participants = participant_counts.most_common(30)
    top_names = {name for name, _ in top_30_participants}

    nodes = [
        ParticipantNode(
            id=name,
            message_count=count,
            source_types=sorted(participant_sources.get(name, set())),
        )
        for name, count in top_30_participants
    ]

    # 네트워크 엣지 (Top 30 참여자 간 연결만)
    edges = [
        ParticipantEdge(source=a, target=b, weight=w)
        for (a, b), w in edge_counts.most_common(100)
        if a in top_names and b in top_names and w >= 2
    ]

    # 타임라인 (월별 정렬)
    timeline = [
        TimelinePoint(
            month=m,
            email_count=counts.get("email", 0),
            teams_chat_count=counts.get("teams_chat", 0),
            document_count=counts.get("document", 0),
        )
        for m, counts in sorted(month_data.items())
    ]

    # Top 20 토픽
    top_topics = [
        TopicItem(topic=t, count=c)
        for t, c in topic_counts.most_common(20)
    ]

    return {
        "source_type_counts": dict(source_counts),
        "top_participants": nodes,
        "participant_network": NetworkData(nodes=nodes, edges=edges),
        "timeline": timeline,
        "top_topics": top_topics,
    }


# === 엔드포인트 ===


@router.get("/{case_id}", response_model=DashboardResponse)
async def get_dashboard(case_id: str):
    """케이스 커뮤니케이션 분석 대시보드 데이터"""
    store = _case_store

    # 케이스 존재 + 상태 확인
    try:
        case_meta = store.get(case_id)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"케이스를 찾을 수 없습니다: {case_id}")

    if case_meta.status not in (CaseStatus.READY, CaseStatus.ARCHIVED):
        raise HTTPException(
            status_code=409,
            detail=f"이 케이스는 아직 분석할 수 없습니다 (상태: {case_meta.status.value}). "
            "인덱싱이 완료된 케이스만 대시보드를 볼 수 있습니다.",
        )

    # ChromaDB에서 케이스 메타데이터만 조회 (단일 공유 컬렉션 + case_id 필터)
    try:
        vector_store = VectorStoreService(case_id=case_id)
        collection = vector_store._get_collection()

        result = collection.get(
            where={"case_id": case_id},
            include=["metadatas"],
        )
        all_metadatas: list[dict[str, Any]] = result.get("metadatas") or []
        total = len(all_metadatas)

    except Exception as e:
        logger.error(f"ChromaDB 조회 실패: {case_id} — {e}")
        raise HTTPException(status_code=500, detail=f"벡터DB 조회 실패: {e}")

    # 집계
    aggregated = aggregate_dashboard(case_id, all_metadatas)

    return DashboardResponse(
        case_id=case_id,
        case_name=case_meta.name,
        total_chunks=total,
        **aggregated,
    )
