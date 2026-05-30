# 향후 개선 항목 (TODO)

2026-04-18 코드 리뷰 시점에 식별되었으나 PoC 일정상 보류된 항목들.
운영 전환 또는 1TB 실데이터 투입 단계에서 재검토 필요.

---

## #13 — ChromaDB → Qdrant/Milvus 마이그레이션 검토

**현황**
- ChromaDB는 단일 노드 + 메모리 기반 HNSW 인덱스.
- 1TB 규모(수억 청크)에서 검색 latency 및 RAM 압박 우려.

**제안**
- Qdrant: 디스크 기반 HNSW + payload 인덱스, 케이스별 필터 성능 우수.
- Milvus: 분산 환경 + GPU 인덱스 옵션.

**고려 사항**
- 현재 코드는 `VectorStoreService`로 추상화되어 있어 백엔드 교체는 가능.
- 마이그레이션 시 chunk_id 호환, 메타데이터 직렬화 형식, BM25 인덱스 위치 등 검토 필요.
- Docker compose에 새 서비스 추가, 운영 모니터링/백업 절차도 신설.

**액션 트리거**
- PoC 종료 후 1TB 실데이터 인덱싱 시 검색 latency가 P95 > 500ms 또는 메모리 사용률이 80% 초과 시 착수.

---

## #15 — Ollama 임베딩 → TEI(Text Embeddings Inference) 전환 검토

**현황**
- Ollama의 `/api/embed`로 bge-m3 호출. 배치 처리 가능하나 처리량 한계 있음.

**제안**
- HuggingFace TEI(`ghcr.io/huggingface/text-embeddings-inference`)는 bge-m3 전용 최적화.
- 동일 GPU(A5000 24GB)에서 Ollama 대비 처리량 2~3배 보고됨 (벤치마크 필요).

**고려 사항**
- `EmbeddingService.embed_texts()` 내부 HTTP 호출 부분만 교체하면 적용 가능.
- TEI 응답 포맷이 다르므로 파싱 로직 수정 필요 (`{"embeddings": [...]}` 형태는 OpenAI 호환).
- 모델 다운로드/캐시 경로, GPU 메모리 분배(LLM과의 동거) 재설계 필요.
- 보안 정책: 임베딩이 항상 로컬에서 동작해야 하므로 컨테이너 격리 유지.

**액션 트리거**
- Phase A 인덱싱 시간이 운영 SLA(예: 1TB / 24시간) 미달 시 착수.
- 또는 GPU 활용률이 80% 미만으로 측정될 때.

---

## 참고: 이미 적용된 개선 (2026-04-18)

| # | 항목 | 위치 |
|---|------|------|
| 1 | BM25 토큰화 캐싱 | `vector_store.py` |
| 2 | `get_stats()` 메모리 폭주 수정 | `vector_store.py` |
| 3 | `_known_ids` upsert 전환 | `vector_store.py` |
| 4 | `case_meta` None 가드 | `pipeline.py` |
| 5 | 워커 initializer로 파서 재사용 | `pipeline.py` |
| 6 | ProcessPoolExecutor chunksize 최적화 | `pipeline.py` |
| 7 | 임베딩 배치 큰 텍스트 truncate | `embedding_service.py` |
| 8 | BM25 검색 결과 캐시 (ChromaDB 조회 제거) | `vector_store.py` |
| 9 | 임계 실패율 abort | `pipeline.py` |
| 10 | chat 히스토리 ERROR 로깅 | `chat.py` |
| 11 | 임베딩 catch-all 제거 | `embedding_service.py` |
| 12 | SQLAlchemy AsyncSession 인프라 | `database.py` |
| 14 | 단일 컬렉션 + case_id 필터 | `vector_store.py` 외 |
