# AuditRAG v4 구현 지시서 (Claude Code용) — v4.1

> **Claude Code에게 줄 첫 지시 (복사해서 사용)**
>
> `docs/v4/CLAUDE_CODE_v4_구현지시서.md`를 끝까지 읽고 §2 원칙을 지키면서 §7의 순서대로 진행해 줘.
> 먼저 0단계(읽기 전용 점검)만 하고 §8 형식으로 보고한 뒤 내 확인을 기다려.
> 참조 코드는 `docs/v4/ref/`에 있고, `pytest docs/v4/ref/test_ref.py`로 동작을 확인할 수 있어.

- 이 문서는 이전에 전달한 `AuditRAG_v4_단순화_ClaudeCode_킥오프.md`(논의용)를 대체한다.
- v3.3 실험 하네스(`AuditRAG_experiments_harness_v3.3.zip`)는 **그대로 적용하지 않는다**(§6.1).
- 논문 기준 버전: `AuditRAG_논문_통합본_v3.5_KR.md`. 이 작업의 산출물로 논문 3·4장을 v4로 개정한다(§6.9).
- 코드 경로는 공개 저장소 `main`(62f4d5a, 2026-06-16) 기준이다. 로컬 브랜치가 앞서 있으면 0단계에서 차이를 보고한다.

### v4.1 변경 사항 (0단계 보고 반영, 2026-10-15)

| # | 항목 | 변경 내용 | 반영 위치 |
|---|---|---|---|
| V1 | `src/` 수정 범위 | 0단계 보고의 "파서는 실험 코드에서 직접 처리, `src/`는 수정하지 않음"은 **이 지시서와 다르다.** v4는 서비스와 실험이 같은 코드를 쓰는 것이 핵심이므로, 파서 수정(PDF 페이지, PPTX 버그, 첨부 순회)과 단위 생성은 모두 `src/`에서 한다. "`src/` 수정 금지"는 v3.3 하네스 시절 규칙이며 폐기한다 | §2, §5.1 |
| V2 | 작업 브랜치 기준 | `v4-simplify`는 `origin/feat/search-quality` 최신 커밋에서 만든다. 이메일 정제·날짜 필터 수정 등이 그 브랜치에만 있기 때문이다. 그 브랜치의 RRF 가중치·하한·검색 범위 조정은 v4에서 제거되므로 가져오지 않아도 된다. 브랜치 커밋 목록을 1단계 보고에 분류해서 적는다(가져옴/v4에서 대체됨) | §7 |
| V3 | 토큰 기준 | XLM-R 대신 **Qwen3 토크나이저**(`Qwen/Qwen3-Embedding-4B`)로 센다. 주 모델 둘이 같은 토크나이저를 쓰기 때문이다. v3.3 `TokenCounter`는 `hf_name`만 바꿔서 쓴다 | §5.1.7 |
| V4 | 파트 상한 | `UNIT_MAX_PART_TOKENS` 1,500 → **2,048**. 0단계 실측(이메일 p95 1,677, PDF 페이지 p99 2,049) 기준으로 이메일 약 5%만 나뉘고, 헤더 + 2,048 + 지시문이 재순위 입력 4,096 안에 들어가서 **재순위 절단이 원칙적으로 생기지 않는다** | §5.1.7, §5.4, §5.9 |
| V5 | XLSX·DOCX | XLSX는 **행 블록이 단위**다(시트 하나가 81만 토큰까지 있어서 시트 단위 판정이 불가능). DOCX는 제목 경계 + 길이로 만든 가상 페이지가 단위다. 파트는 이메일과 긴 PDF 페이지에서만 생긴다 | §5.1.2, §5.1.4 |
| V6 | 미지원 첨부 | xls(`xlrd`)·html·확장자 없음(매직 바이트 판별)은 지원한다. 스캔 PDF 페이지는 기존 OCR 모듈로 처리한다. 이미지·hwp·dwg는 제외하고 건수만 기록한다 | §5.1.4 |
| V7 | 전달 메일 | 제목이 전달 접두어(FW·Fwd·전달·转发·RV·PŘ·WG)로 시작하면 **인용을 지우지 않는다.** 전달된 내용은 이 메일함에 다른 사본이 없는 경우가 많다. 회신의 인용만 지운다 | §5.1.6 |
| V8 | 언어 감지 | `langdetect` → **`lingua-language-detector`**(후보 언어 제한). 0단계에서 langdetect가 한국어 메일 약 950건을 vi·et로 잘못 판정했다. 혼용 정의는 두 번째 언어가 토큰 기준 20% 이상 | §5.1.8 |
| V9 | 파일 이름 | `src/text/tokenize.py` → `src/text/bm25_tokenizer.py`, `src/units/langid.py` → `src/units/lang_detect.py` | §4, §5.2 |
| V10 | 재순위 | 두 백엔드(qwen3·bge) 모두 `RERANK_MAX_LENGTH=4096`으로 같은 상한을 쓴다(R9 공정성). Qwen3는 `logits_to_keep=1`로 마지막 위치 로짓만 계산한다(`ref/qwen3_formats.py` 수정됨) | §5.4 |
| V11 | VRAM 운용 | 검색 실험(R1~R9 실행)은 LLM을 내린 상태에서 한다. 세 모델 동시 적재는 서비스 조건 응답 시간 측정(`latency`)에서만 한다 | §6.7 |
| V12 | 개발·실측 환경 | 이 PC(RTX 2060 12GB, Turing, BF16 미지원)에서는 가짜 모델과 `Qwen3-Reranker-0.6B`(FP16)로 동작만 확인한다. 실측은 A5000에서 한다. Python은 프로젝트 전체를 **3.12 venv 하나**로 통일한다(서비스도 재순위에 torch가 필요하다) | §5.9, §7 |
| V13 | 채팅 | 방 키를 "폴더 경로 + 참여자 목록"에서 **폴더 경로 + 대화 주제**(MAPI ConversationTopic, 없으면 폴더 경로)로 바꾼다. `chat_type`이 첫 메시지 값으로 고정되는 버그를 고친다. 현재 코퍼스에 채팅이 없으므로 단위 테스트로만 검증한다 | §5.1.4 |
| V14 | 평가 자산 | 질의 100개·판정 데이터는 없는 것으로 보고 새로 만든다(논문 4장 수치는 원래 가정치). 평가 코퍼스 범위(이 메일함 하나 / 추가 조사 대상자·채팅 포함)는 **6단계 전까지** 사용자가 정한다. 1~5단계는 범위와 관계없이 진행한다 | §6.3, §7 |

---

## 1. 목표

### 1.1 한 줄 요약

**검색 단위 = 증거 단위 = 판정 단위**로 바꾸고, 임베딩·재순위를 다국어 모델로 교체한다. 서비스와 실험이 **같은 검색 함수**를 쓰도록 구조를 단순화한다.

### 1.2 바뀌는 것

| 영역 | 현재(v3.x) | v4 |
|---|---|---|
| 색인 단위 | 4종 청커(문자 수·시간 창·스레드·첨부) | 증거 단위: 이메일=메시지, 채팅=세션, 문서=페이지, 첨부=첨부(페이지). 단위 앞에 메타데이터 헤더 |
| 벡터 저장 | ChromaDB 단일 공유 컬렉션 + `case_id` 필터 | 케이스 폴더별 numpy 행렬(fp16) + 정확 탐색(코사인) |
| 임베딩 | Ollama `bge-m3`(1,024차원) | Ollama `qwen3-embedding:4b` 8비트(2,560차원), 질의에만 영어 지시문 |
| BM25 토큰화 | 전체 텍스트에 Kiwi + 품사·길이 필터 | 문자 체계별 토큰화(`ref/script_tokenizer.py`) |
| 결합 | RRF k=60 + 하한 0.0141, 재순위 없을 때 상위 20 | RRF k=60만(하한·가중치 없음), 검색기별 상위 50 |
| 재순위 | `bge-reranker-v2-m3`(FlagEmbedding), 후보 50 | `Qwen3-Reranker-4B`(transformers, BF16), 후보 30. `bge`는 비교 실험용 백엔드로 유지 |
| 응답 생성 | 개발 `gemma4:e4b` / 운영 `gpt-oss:20b` | `gemma4:12b`(Ollama 4비트) |
| 이메일 정제 | 한·영 인용 헤더(로컬에 있다면) | 현지어 인용 헤더 추가(`ref/email_quote.py`) |
| 언어 정보 | 없음(파일 단위 `langdetect`, 오감지 많음) | 단위별 언어 감지(`lingua-language-detector`) → 메타데이터에만 저장 |
| 실험 | `scripts/run_benchmark.py`(청크 단위 GT) + v3.3 하네스(어댑터 방식) | `experiments_v4/`: 서비스의 `retrieve()`를 직접 호출하는 얇은 스크립트 + `ranx` |

### 1.3 바뀌지 않는 것

- 케이스 라이프사이클과 상태 머신(`src/cases/case_store.py`)
- 파서의 기본 동작(`src/parsers/`). 단, §5.1의 필드 추가는 예외다.
- LLM 라우터와 보안 모드(`src/llm/router.py`)
- 질의 파서(`src/rag/query_parser.py`). 서비스 필터로만 쓰고 실험에서는 쓰지 않는다.
- 토픽 추출(`metadata_enricher.extract_topics`). 대시보드용으로 단위 메타데이터에 유지한다.
- 프론트엔드: API 응답 스키마를 유지해서 수정을 최소화한다(§5.7).

---

## 2. 원칙과 제약

1. **로컬 전용.** 조사 텍스트를 외부 API로 보내지 않는다. 임베딩·재순위·번역·응답 생성은 모두 로컬 모델로 한다. 실험은 `SECURITY_MODE=on`에서만 실행한다.
2. **데이터 커밋 금지.** `data/`, `experiments_v4/data/`, `runs/`, `results/`, `cache/`, 모델 가중치는 커밋하지 않는다. `.gitignore`를 먼저 확인한다.
3. **설정값은 이 문서로 고정한다.** k값, RRF k, 재순위 후보 수, 지시문은 §5에 정한 값을 쓴다.
   - 평가 질의 결과를 보고 바꾸지 않는다.
   - 논문 초안의 † 수치에 맞추려고 조정하지 않는다.
   - 개발 질의는 동작 점검에만 쓴다.
4. **작업 브랜치** `v4-simplify`에서 작업하고, 단계마다 커밋한다. 커밋 메시지는 한국어나 영어 모두 가능하다.
5. **체크포인트 3곳(§7)에서 멈추고 보고한다.** 체크포인트 사이에서는 스스로 판단해서 진행하되, 판단한 내용은 보고에 적는다.
6. **삭제는 체크포인트 3 이후에만.** ChromaDB 코드·의존성·기존 청커·`run_benchmark.py`는 체크포인트 3에서 사용자가 확인한 뒤 삭제한다. 그 전에는 새 경로로 교체만 하고 남겨 둔다.
7. **기존 사용자 데이터는 자동 삭제하지 않는다.** 대상은 `data/vectordb/`, `data/bm25_index/`, `data/processed/`다. 새 색인은 `data/index/`에 만든다.
8. **CLAUDE.md 규칙을 따른다.** 타입 힌트, docstring, ruff를 지키고, 주석은 한국어로 써도 된다.
9. **`src/`를 고친다.** 실험 코드에 파서·단위 생성·검색 로직을 따로 만들지 않는다. 실험은 `src/`의 함수를 호출만 한다. 서비스에서 발견한 버그(PPTX 파싱, PDF 페이지 경계, 첨부-부모 연결)도 `src/`에서 고친다.

---

## 3. 현재 코드에서 확인된 사항(공개 main 기준)

0단계에서 로컬 코드와 대조하고, 다르면 보고한다.

| # | 위치 | 현재 상태 | v4에서 필요한 것 |
|---|---|---|---|
| C1 | `src/vectorstore/vector_store.py` | 단일 공유 컬렉션 `ast_chunks` + `case_id` 필터. 논문 v3.x의 "케이스별 컬렉션" 서술과 다름 | 케이스 폴더별 색인으로 교체(§5.5) |
| C2 | `vector_store._kiwi_tokenize` | 전체 텍스트를 Kiwi로 처리하고 SL ≥3자, 그 외 ≥2자만 남김. kiwipiepy 0.24.0에서는 `payment.`가 `pa`/`yment.`로, `březen.`이 `bře`/`zen.`으로 깨짐 | `script_tokenizer.tokenize`로 교체(§5.2) |
| C3 | `document_parser` PDF | 페이지별로 추출한 뒤 `content`로 합침. `ParsedDocument`에 페이지 목록이 없음 | 페이지 목록 보존(§5.1.4) |
| C4 | `pst_parser` | `result.attachments`가 평탄한 목록이라 부모 메일 연결이 없음. `EmailMessage.attachments`에는 메일별로 있음 | 첨부 단위는 메일을 순회하며 생성(§5.1.5) |
| C5 | `EmailMessage` | `message_id`는 있고 `cc`·`in_reply_to`는 없음(`SourceReference`에는 있음) | 파서에서 채울 수 있으면 추가, 없으면 빈 값 |
| C6 | `chunker.py` | 이메일 헤더 라인 생성기 `_build_email_header_line` 있음. 이메일 인용 정제(`Original Message` 등)는 공개 main에 없음 | 헤더 생성 로직은 재사용. 정제는 로컬 구현이 있으면 확장하고, 없으면 `ref/email_quote.py` 도입 |
| C7 | `engine.search` | 재순위 없을 때 `search_top_k=20` + 상대 점수 필터(평균의 30%). 재순위 있을 때 후보 50 + `rerank_min_score=0.1` | 검색은 `retrieve()`로 통일. 서비스 후처리는 §5.7 |
| C8 | `dashboard.py` | Chroma `collection.get(where=case_id)`로 메타데이터 집계 | `units.jsonl` 메타데이터로 집계 |
| C9 | `.env.prod.example` | `OLLAMA_LLM_MODEL=gpt-oss:20b`, `OLLAMA_EMBED_MODEL=bge-m3`, `RRF_MIN_SCORE=0.0141` | §5.9 값으로 변경 |
| C10 | CLAUDE.md GPU 절 | Phase A·B 순차 전용, 동시 사용 없음 | Phase B에서 임베딩(질의)·재순위·LLM 동시 적재(약 21~22GB)로 갱신 |

---

## 4. 목표 구조

```
src/
├─ units/                      # 신규 — chunkers/ 를 대체
│  ├─ schema.py                # EvidenceUnit, UnitPart 데이터 클래스
│  ├─ builder.py               # 파싱 결과 → 증거 단위(유형별 규칙, ID, 중복 제거)
│  ├─ header.py                # 유형별 헤더 문자열
│  ├─ email_clean.py           # 다국어 인용·서명 정제 (ref/email_quote.py 기반, 기존 정제 함수와 통합)
│  ├─ split.py                 # 긴 단위 → 파트 분할 (토큰 기준)
│  └─ lang_detect.py           # lingua 래퍼 (후보 언어 제한, 혼용 판정)
├─ text/
│  └─ bm25_tokenizer.py        # ref/script_tokenizer.py 이식 (BM25 색인·질의 공용)
├─ index/                      # 신규 — vectorstore/ 를 대체
│  ├─ case_index.py            # 케이스 폴더 색인: 저장·로드·검색·필터·삭제·검증
│  └─ bm25.py                  # rank_bm25 래퍼 (토크나이저 버전 기록)
├─ embeddings/embedding_service.py   # 수정 — Qwen3 질의 지시문, L2 정규화, 프로필별 모델
├─ rag/
│  ├─ retrieval.py             # 신규 — retrieve(): BM25/Dense → RRF → 재순위 (서비스·실험 공용)
│  ├─ reranker.py              # 수정 — 백엔드 qwen3 | bge
│  ├─ engine.py                # 수정 — retrieve() 사용
│  └─ query_parser.py          # 유지 (서비스 필터)
├─ indexing/pipeline.py        # 수정 — 청킹 단계 → 단위 생성, 저장 단계 → case_index
├─ chunkers/                   # 체크포인트 3 이후 삭제 (metadata_enricher.extract_topics 는 units 로 이동)
└─ vectorstore/                # 체크포인트 3 이후 삭제
scripts/
├─ unit_stats.py               # 신규 — 실데이터 단위 통계 (임베딩 없이)
├─ build_profiles.py           # 신규 — 케이스에 색인 프로필 추가 생성 (실험용)
├─ check_embed_parity.py       # 신규 — Ollama 8비트 vs transformers BF16 임베딩 일치 확인
└─ run_benchmark.py            # 체크포인트 3 이후 삭제
experiments_v4/                # 신규 — §6
```

---

## 5. 상세 사양

### 5.1 증거 단위(`src/units/`)

#### 5.1.1 스키마

```python
@dataclass
class UnitPart:
    part_id: str          # f"{unit_id}#{k}", k=1..n
    unit_id: str
    unit_type: str        # "email" | "chat" | "document" | "attachment"
    header: str           # §5.1.3
    text: str             # 정제된 본문 조각 (헤더 제외)
    n_tokens: int         # Qwen3 토크나이저 기준 (§5.1.7)
    meta: dict            # 아래 필드
```

`meta`의 필드는 다음과 같다.

| 구분 | 필드 |
|---|---|
| 공통 | `case_id`, `source_file`(상대 경로), `date`(ISO 8601, 시작 시각), `date_end`, `participants`(list), `lang`, `lang_mixed`(bool), `lang_second`, `topics`(list) |
| 이메일 | `sender`, `recipients`, `cc`, `subject`, `message_id`, `attachment_filenames`, `folder_paths`(중복 제거 시 병합) |
| 채팅 | `conversation_id`, `chat_type`, `n_messages` |
| 문서·첨부 | `filename`, `file_type`, `page`(1부터), `page_count`, `page_kind`(`page`·`slide`·`sheet`·`block`), `author`, `last_modified_by`, `created_date`, `last_modified` |
| 첨부 | `parent_unit_id`, `parent_subject`, `parent_sender`, `parent_date` |

#### 5.1.2 단위 ID(재색인해도 같아야 함)

정답(qrels)이 `unit_id`로 저장되므로, 같은 원본은 항상 같은 ID가 나와야 한다. 결정성 테스트를 필수로 둔다.

| 유형 | ID 규칙 |
|---|---|
| 이메일 | `em:` + sha1(소문자 `message_id`)[:16]. `message_id`가 없으면 sha1(`sender`\|`date`\|`subject`\|본문 앞 200자) |
| 채팅 | `ch:` + sha1(`conversation_id`)[:10] + `:` + 세션 시작 시각(UTC, `YYYYMMDDTHHMMSS`) |
| 문서 | `doc:` + sha1(케이스 기준 상대 경로)[:12] + `:p{n}` (슬라이드 `:p{n}`, DOCX 가상 페이지 `:b{n}`, XLSX 행 블록 `:s{시트}r{시작행}-{끝행}`) |
| 첨부 | `att:` + 부모 이메일 ID의 해시 부분 + `:` + 첨부 순번 + (페이지가 있으면 `:p{n}`) |

- **중복 이메일:** 조사 대상자 여러 명의 PST에 같은 메일이 들어 있으면 같은 `unit_id`가 된다. 처음 것만 남기고 `folder_paths`를 병합한다. 그 아래 첨부도 함께 중복 제거된다. 제거 건수는 단위 통계(§6.8 T1)에 보고한다.

#### 5.1.3 헤더(레이블은 본문 언어와 관계없이 한국어로 고정)

```
[이메일] 2024-03-05 14:22 | 보낸이: 홍길동 | 받는이: 김철수, 이영희 | 참조: 박민수 | 제목: RE: 3월 정산 건 | 첨부: 견적서.xlsx
[채팅] 2024-03-05 14:00~14:47 | 참여자: 홍길동, 김철수 | 방: 정산팀
[문서] 정산기준_v3.docx | 3/12쪽 | 작성자: 홍길동 | 수정일: 2024-02-28
[첨부] 견적서.xlsx | 2/3쪽 | 부모 메일: 2024-03-05 홍길동→김철수 "RE: 3월 정산 건"
```

- 빈 필드는 생략한다.
- 받는이·참여자가 5명을 넘으면 앞 5명 뒤에 `외 n명`을 붙인다.
- 헤더 형식은 체크포인트 2에서 사용자가 확정한다.
- 기존 `_build_email_header_line`의 필드 수집 로직은 재사용한다.

#### 5.1.4 유형별 생성 규칙

| 유형 | 단위 | 규칙 |
|---|---|---|
| 이메일 | 메시지 1통 | 본문을 `email_clean.strip_quoted`로 정제한다(§5.1.6). 정제 결과가 비면 원문을 쓴다. 짧은 회신("OK")도 하나의 단위다 |
| 채팅 | 세션 | 같은 방에서 메시지 간격이 `CHAT_SESSION_GAP_MINUTES`(30)를 넘으면 새 세션. 방 키는 **폴더 경로 + 대화 주제**(MAPI ConversationTopic, 없으면 폴더 경로). 참여자 목록은 방 키에 넣지 않는다(V13). `chat_type`은 세션마다 계산한다. 본문은 `HH:MM 발화자: 내용` 줄 목록 |
| PDF | 페이지 | `ParsedDocument`에 `pages: list[str]`를 추가하고, 기존 `content`는 호환을 위해 유지한다. 빈 페이지도 자리를 지켜서 페이지 번호가 원본과 같게 한다. 텍스트 레이어가 없는 페이지는 기존 OCR 모듈(`src/parsers/ocr/`)로 처리한다(V6) |
| PPTX | 슬라이드 | 슬라이드별 텍스트. `document_parser.py`의 `hasattr(shape, "placeholder_format")` 버그를 고친다(`shape.is_placeholder`로 먼저 확인하거나 `try/except ValueError`). 0단계 실측 147건 중 132건 실패 |
| XLSX·XLS | 행 블록 | 시트를 행 단위로 묶어 **블록 하나를 단위로 만든다**(파트가 아님). 블록은 머리글 행 + 데이터 행으로 `UNIT_MAX_PART_TOKENS` 이하. 머리글 행은 시트 첫 비어 있지 않은 행으로 본다. XLS는 `xlrd`로 읽는다 |
| DOCX | 가상 페이지 | 제목 스타일(Heading 1~3) 경계에서 나누고, 길면 문단 경계에서 `UNIT_MAX_PART_TOKENS` 이하로 다시 나눈다. `page_kind="block"`, 헤더에는 `n/N부`로 표기 |
| TXT·HTML·확장자 없음 | 가상 페이지 | HTML은 태그를 제거한 텍스트(`_strip_html` 재사용). 확장자가 없으면 매직 바이트로 형식을 판별한다(PDF·OLE·ZIP 계열·EML 텍스트). 이후 규칙은 해당 형식을 따른다 |
| 제외 | — | 이미지, HWP, DWG, 판별 불가 형식은 단위를 만들지 않고 형식별 건수만 T1에 기록한다 |
| EML·MSG 파일 | 이메일 | 문서 폴더에 있어도 이메일 단위로 처리한다. **EML 첨부도 파싱한다**(현재는 파일명만 헤더에 붙음) |
| 첨부 | 첨부 1건 | **`result.attachments`가 아니라 `email.attachments`·`chat.attachments`를 순회**해서 부모를 안다. EML 경로는 `email.message`의 MIME 파트를 순회한다. 형식별 규칙(위 표)으로 페이지·블록 단위가 되고, 헤더에 부모 메일 정보가 들어간다 |

#### 5.1.5 첨부와 부모의 연결

- `IndexingPipeline._process_pst`가 지금은 `result.attachments`(평탄한 목록)를 첨부 청커에 넘긴다. v4에서는 `builder.build_from_pst(result)`가 이메일·채팅을 순회하며 첨부 단위를 만든다.
- 평탄한 목록에만 있고 부모를 찾을 수 없는 첨부가 있으면 `parent_unit_id=""`로 만들고 건수를 보고한다.

#### 5.1.6 이메일 정제

- `feat/search-quality` 브랜치의 `chunker.py` `clean_email_body`(서명·면책 문구 포함)를 `src/units/email_clean.py`로 옮기고, `ref/email_quote.py`의 패턴(중국어·체코어·스페인어·독일어·프랑스어 인용 헤더, "wrote:" 변형, 헤더 레이블 2줄 연속 규칙)을 합친다. 두 구현을 따로 두지 않는다.
- **전달 메일 예외(V7):** 제목이 전달 접두어(`FW:` `Fwd:` `전달:` `转发:` `RV:` `PŘ:` `WG:`, 대소문자·전각 콜론 무시)로 시작하면 인용을 지우지 않는다. 서명·면책 문구 제거와 `>` 줄 제거만 한다. 전달 메일 수와 그 길이 분포를 T1에 따로 적는다.
- 0단계에서 확인된 중국어 인용 헤더 샘플(`回复_ DBCN 공장 방문용 초청발 발급 요청.eml` 등 5건)을 테스트 고정 자료로 쓴다. 실데이터 원문은 저장소에 넣지 않고, 구조만 남긴 가명 사본을 `tests/fixtures/`에 만든다.
- `ref/test_ref.py`의 이메일 테스트를 프로젝트 테스트로 옮긴다.

#### 5.1.7 토큰 수와 긴 단위 분할

- **토큰 수:** Qwen3 토크나이저(`Qwen/Qwen3-Embedding-4B`)로 센다(V3). v3.3 `TokenCounter`를 이식하고 `hf_name`만 바꾼다(`ref/from_v33/text.py`). HF 캐시가 없으면 문자 근사로 떨어진 사실을 기록한다.
- **분할 기준:** `UNIT_MAX_PART_TOKENS`(기본 **2,048**, V4)를 넘는 단위는 줄·문단 경계에서 나눠 파트로 만든다. 모든 파트에 같은 헤더를 붙인다. XLSX·DOCX는 애초에 이 상한 이하의 블록 단위로 만들기 때문에 파트가 생기지 않는다.
- **기본값 확정:** 2,048은 체크포인트 2에서 단위 통계를 보고 확정한다.
- **검색과 판정:** 검색은 파트 단위로 점수를 매긴 뒤 단위로 합친다(§5.6). 판정은 단위 전체를 보여준다.

#### 5.1.8 언어 감지

- **도구:** `lingua-language-detector`를 쓴다(V8). 후보 언어는 한국어·영어·중국어·일본어·독일어·체코어·스페인어·프랑스어·폴란드어·베트남어로 제한한다(`LanguageDetectorBuilder.from_languages`). 헤더를 제외한 본문으로 판정한다. `langdetect`는 쓰지 않는다.
- **주 언어:** 본문을 줄 단위로 판정한 뒤, 토큰 수(`bm25_tokenizer` 기준)로 가중한 최다 언어로 정한다. 토큰이 5개 미만이면 `und`로 둔다.
- **혼용:** 두 번째 언어의 토큰 비중이 20% 이상이면 `lang_mixed=True`, `lang_second`에 그 언어를 기록한다.
- **검증:** 0단계에서 vi·et로 잘못 판정된 메일 중 20건을 표본으로 다시 판정해서 1단계 보고에 정확도를 적는다.
- **용도:** 분석에만 쓴다. 검색 경로를 나누는 데 쓰지 않는다.

### 5.2 BM25 토크나이저(`src/text/bm25_tokenizer.py`)

`ref/script_tokenizer.py`를 그대로 이식하고, 테스트도 함께 옮긴다.

- **문자 체계별 처리:**
  - 한글 구간: Kiwi 형태소. 기본은 구두점·기호 태그만 제외하고, `ko_pos_filter=False`다.
  - 한자·가나 구간: 2글자 단위(bigram).
  - 라틴·키릴 구간: 소문자로 바꾸고 발음 구별 기호를 제거한다(`dvořák` → `dvorak`).
  - 코드형 문자열·금액·이메일 주소: 통째 토큰을 추가한다.
- **동일 함수 사용:** 색인과 질의에 같은 함수를 쓴다. 규칙을 바꾸면 `TOKENIZER_VERSION`을 올리고, 색인 메타와 다르면 재색인을 요구한다.
- **BM25 구현:** `rank_bm25.BM25Okapi`(기존 의존성)를 쓴다. 3만 파트에서 질의당 BM25 p50이 300ms를 넘으면 `bm25s`로 교체를 제안한다(교체는 보고 후 진행).

### 5.3 임베딩(`src/embeddings/embedding_service.py`)

| 항목 | 값 |
|---|---|
| 기본 모델 | Ollama `qwen3-embedding:4b-q8_0`. 0단계에서 태그 존재를 확인하고, 없으면 사용 가능한 8비트 태그를 보고 |
| 차원 | 2,560(실측 확인), L2 정규화를 우리 쪽에서 다시 수행 |
| 질의 형식 | `ref/qwen3_formats.format_embed_query(query)` → `"Instruct: {EMBED_TASK}\nQuery:{query}"` |
| 문서 형식 | 지시문 없이 `header + "\n\n" + text`(헤더 on 프로필) 또는 `text`(헤더 off) |
| 컨텍스트 | 요청 옵션 `num_ctx=8192`. 파트는 2,048토큰 이하라 잘리지 않아야 한다. 잘림이 감지되면 경고 로그를 남긴다 |
| 비교 프로필 | `bge-m3`(Ollama, 지시문 없음). 실험 R9 전용 |
| 기존 로직 | 배치 실패 시 이분할 재시도, 실패 단위 JSONL 보존은 유지 |

- `EMBED_TASK` 문구는 `ref/qwen3_formats.py` 초안을 쓰고, 체크포인트 2에서 사용자가 확정한다.
- `scripts/check_embed_parity.py`는 샘플 단위 30개와 질의 10개에 대해 Ollama 8비트 임베딩과 transformers BF16 `Qwen/Qwen3-Embedding-4B`(last-token 풀링, 정규화) 임베딩의 코사인 유사도를 계산한다. 평균 0.99 이상이면 통과다. HF 가중치를 사내에 반입할 수 없으면 이 단계를 건너뛰고 보고한다.

### 5.4 재순위(`src/rag/reranker.py`)

- **백엔드:** `RERANK_BACKEND = qwen3 | bge`.
  - `qwen3`: `ref/qwen3_formats.Qwen3Reranker`를 이식한다. transformers ≥4.51, BF16, `attn_implementation="sdpa"`. Windows에서는 flash-attn을 쓰지 않는다.
  - `bge`: 기존 FlagEmbedding 구현을 유지한다(R9 전용).
- **입력:** `header + "\n\n" + text`(헤더 on 프로필) 또는 `text`(헤더 off). 단위의 대표 파트(§5.6)를 쓴다. 지시문은 `RERANK_INSTRUCTION`(영어, 체크포인트 2에서 확정)이다.
- **길이:**
  - `RERANK_MAX_LENGTH`는 4,096이고, **qwen3·bge 두 백엔드에 같은 값**을 쓴다(V10, R9 공정성). FlagEmbedding의 기본값(512)을 그대로 두지 않는다.
  - 파트 상한이 2,048이므로 원칙적으로 잘리지 않는다. 그래도 잘린 건수를 집계해서 보고한다(논문 4.6절).
- **메모리:** Qwen3는 `logits_to_keep=1`로 마지막 위치 로짓만 계산한다(V10). 이렇게 하지 않으면 배치×길이×어휘(약 15만) 크기의 로짓 때문에 수 GB를 더 쓴다.
- **배치·점수:** `RERANK_BATCH_SIZE` 기본값은 4이고, OOM이 나면 2로 줄인다. 점수는 P(yes)이며 0~1 범위다.
- **개발 PC(RTX 2060, Turing):** BF16을 지원하지 않으므로 `Qwen3-Reranker-0.6B`를 FP16으로 써서 동작만 확인한다. `RERANK_MODEL`·`RERANK_DTYPE` 설정으로 바꾼다(V12).
- **적재:** 프로세스당 1회만 적재한다(싱글턴). 첫 호출 지연은 응답 시간 측정에서 제외한다.

### 5.5 케이스 색인(`src/index/case_index.py`)

```
data/index/{case_id}/
├─ manifest.json            # schema_version, tokenizer_version, unit_rules_version, n_units, n_parts, created_at, source_files(해시)
├─ units.jsonl              # UnitPart 한 줄씩 (행 순서 = 행렬 행 순서)
├─ bm25/hdr.pkl, nohdr.pkl  # 토큰화 코퍼스 + tokenizer_version
└─ profiles/{profile}/
   ├─ meta.json             # profile, header(bool), embed_model, embed_dim, normalized, created_at
   └─ dense.npy             # float16 [n_parts, dim]
```

- **프로필:**

  | 프로필 | 용도 |
  |---|---|
  | `hdr-qwen3` | 서비스 기본 |
  | `nohdr-qwen3` | 실험 전용 |
  | `hdr-bgem3` | 실험 R9 전용 |

  서비스는 `SERVICE_INDEX_PROFILE`(기본 `hdr-qwen3`) 하나만 만든다. 실험 프로필은 `scripts/build_profiles.py`로 추가한다.
- **검색:**
  - Dense: 질의 벡터 · 행렬(float32로 나눠 계산) → `argpartition`으로 상위 후보를 뽑는다.
  - 필터(날짜 범위, `unit_type`, 참여자 포함): `units.jsonl` 메타에서 미리 만든 배열로 불리언 마스크를 적용한다.
- **메모리:** 케이스 색인은 필요할 때 로드하고, 최근 2개 케이스까지 캐시한다. 3만 파트 × 2,560 × 2바이트 ≈ 150MB다.
- **증분 색인:** 새 파트를 `units.jsonl`과 `dense.npy`에 이어 붙이고, BM25는 해당 케이스 전체를 다시 만든다.
- **검증:**
  - 설정된 임베딩 모델·차원이 프로필 메타와 다르면 `EmbeddingDimensionError`(기존 클래스)를 낸다.
  - `tokenizer_version`이 다르면 재색인을 요구하는 오류를 낸다.
- **삭제:** 케이스 삭제 시 `data/index/{case_id}/` 전체를 지운다.
- **원자성:** 쓰기는 임시 파일에 한 뒤 교체(rename)한다. 강제 종료 후 반쯤 쓰인 파일이 남지 않게 한다. Chroma 무결성 검사 로직은 필요 없어진다.

### 5.6 검색 함수(`src/rag/retrieval.py`) — 서비스와 실험이 공용으로 쓴다

```python
@dataclass(frozen=True)
class RetrievalConfig:
    profile: str = "hdr-qwen3"
    use_bm25: bool = True
    use_dense: bool = True
    rerank: bool = True
    reranker: str = "qwen3"        # "qwen3" | "bge"
    k_bm25: int = 50
    k_dense: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 30
    depth: int = 100

@dataclass
class UnitHit:
    unit_id: str
    rank: int                      # 1부터
    score: float
    best_part_id: str
    stage_ranks: dict[str, int]    # {"bm25": r, "dense": r, "rrf": r, "rerank": r} (없으면 키 생략)

@dataclass
class RetrievalResult:
    hits: list[UnitHit]
    timings_ms: dict[str, float]   # bm25, embed_query, dense, rrf, rerank, total
    n_rerank_truncated: int

def retrieve(case_id: str, query: str, cfg: RetrievalConfig,
             filters: dict | None = None) -> RetrievalResult: ...
```

**알고리즘**

1. **필터:** `filters`가 있으면 파트 마스크를 만든다. 실험에서는 항상 `None`이다.
2. **BM25:** 질의를 `tokenize`로 처리하고 파트 점수를 계산한다(점수 > 0만 사용). 파트 순위를 단위로 합칠 때는 단위별 첫 등장 파트를 대표 파트로 쓰고, 단위 순위 목록을 만든다.
3. **Dense:** 질의를 임베딩한 뒤 파트 순위를 같은 방식으로 단위로 합친다.
4. **단일 검색기**(`use_bm25` 또는 `use_dense` 하나만 켠 경우): 그 목록을 `depth`까지 반환한다. k가 아니라 depth까지 가져온다.
5. **RRF:** 검색기별 상위 50 단위에 대해 `score(u) = Σ 1/(rrf_k + rank)`(rank는 1부터)를 계산한다.
   - 동점이면 두 검색기 중 더 좋은 순위, 그다음 `unit_id` 순으로 정렬한다.
   - 대표 파트는 RRF 기여가 더 큰 검색기의 대표 파트로 정한다.
6. **재순위:** RRF 상위 `rerank_candidates`(30) 단위를 재순위 점수 내림차순으로 정렬한다. 동점이면 RRF 순서를 따른다. 그 뒤에 RRF 31위 이하를 원래 순서대로 붙여 `depth`까지 반환한다.
7. **점수 하한·가중치·상대 필터는 쓰지 않는다.**

### 5.7 서비스 연결(엔진·API·대시보드)

- **`engine.RAGEngine.search`**
  - `retrieve()`를 호출하고, 상위 `RERANK_TOP_N`(5)개 단위를 `SourceReference`로 변환한다. `content`에는 헤더와 대표 파트 본문을 넣는다.
  - 재순위를 끈 요청은 `RetrievalConfig(rerank=False)`로 처리한다.
  - `SourceReference` 필드는 유지한다. 새 정보(`unit_id`, `page`, `lang`)는 선택 필드로 추가한다.
- **서비스 후처리**
  - `min_source_length`는 유지한다.
  - `_apply_relative_score_filter`는 기본 호출을 끈다(코드는 체크포인트 3에서 정리).
  - `rerank_min_score` 기본값은 0.0이다. Qwen3의 P(yes) 척도에서 다시 정하기 전까지 끈다.
  - 이 후처리는 평가 대상이 아니다. 실험은 `retrieve()` 결과를 그대로 평가한다.
- **API 라우트**
  - `cases.py`: 케이스 삭제 시 `case_index.delete(case_id)`를 호출한다.
  - `settings.py`: 임베딩 모델 변경 시 Chroma 대신 케이스 프로필 메타로 차원을 검증하고, 다르면 재색인 필요 상태를 반환한다.
  - `dashboard.py`: `units.jsonl` 메타로 참여자·토픽·타임라인을 집계한다. 응답의 `total_chunks` 키는 프론트엔드 호환을 위해 유지하되, 값은 단위 수로 하고 docstring에 명시한다.
- **질의 파서:** 서비스 경로에서만 필터를 만든다.

### 5.8 인덱싱 파이프라인(`src/indexing/pipeline.py`)

- **유지하는 골격:** 진행률·취소·멀티프로세싱·실패율 중단·로그 저장은 그대로 둔다.
- **바꾸는 단계:**
  - 파싱 → **단위 생성**(`units.builder`, 정제·언어 감지·토픽 포함) → 임베딩(서비스 프로필) → `case_index` 저장 → BM25 생성.
  - Chroma 저장 호출과 `rebuild_bm25`는 `case_index`로 교체한다.
- **단계 표시:** `IndexingPhase`의 `CHUNKING` 값은 프론트엔드 호환을 위해 그대로 두고, 의미만 "단위 생성"으로 바꾼다. 라벨 변경은 선택이다.
- **처리 캐시:** `data/processed/` 캐시는 단위 JSONL로 바꾸고, 캐시 키에 `unit_rules_version`을 넣는다.

### 5.9 설정·환경(`src/utils/config.py`, `.env.*.example`, CLAUDE.md)

**추가·변경할 설정**

```
# Units
UNIT_MAX_PART_TOKENS=2048
CHAT_SESSION_GAP_MINUTES=30
EMAIL_CLEAN_ENABLED=true

# Index
INDEX_DIR=./data/index
SERVICE_INDEX_PROFILE=hdr-qwen3

# Embedding
OLLAMA_EMBED_MODEL=qwen3-embedding:4b-q8_0
EMBED_DIM=2560
EMBED_NUM_CTX=8192
EMBED_QUERY_TASK="Given an internal audit investigation query, retrieve emails, chat messages, and documents in any language that contain evidence relevant to the query"

# Search
SEARCH_K_BM25=50
SEARCH_K_DENSE=50
RRF_K=60

# Reranker
RERANK_ENABLED=true
RERANK_BACKEND=qwen3
RERANK_MODEL=Qwen/Qwen3-Reranker-4B
RERANK_MAX_LENGTH=4096       # qwen3·bge 공통
RERANK_DTYPE=bf16            # 개발 PC(Turing)는 fp16 + Qwen/Qwen3-Reranker-0.6B
RERANK_BATCH_SIZE=4
RERANK_CANDIDATES=30
RERANK_TOP_N=5
RERANK_MIN_SCORE=0.0
RERANK_INSTRUCTION="Judge whether this email, chat session, or document page contains evidence relevant to the internal audit investigation query. The document may be in any language."

# LLM
OLLAMA_LLM_MODEL=gemma4:12b
OLLAMA_KEEP_ALIVE=-1          # 실험 중 모델이 내려가지 않게
OLLAMA_MAX_LOADED_MODELS=2    # 임베딩 + LLM 동시 적재
```

**삭제할 설정:** `RRF_MIN_SCORE`, `SEARCH_TOP_K`, `RERANK_TOP_K_CANDIDATES`. Chroma 관련 설정은 체크포인트 3 이후 삭제한다.

**Python 환경(V12)**
- 프로젝트 전체(서비스 + `experiments_v4/`)를 **Python 3.12 venv 하나**로 통일한다. 지금 `.venv`(3.14)에는 torch·transformers·FlagEmbedding이 없어서 재순위가 실제로 돌지 않는다. 서비스도 재순위에 torch가 필요하다.
- `pyproject.toml`에 `torch`(CUDA 빌드는 설치 안내에 명시), `transformers>=4.51`, `lingua-language-detector`, `xlrd`, `ranx`(dev 또는 `experiments` extra)를 추가하고, `langdetect`·`chromadb`는 체크포인트 3에서 제거한다.
- 오프라인 반입 절차(Ollama `~/.ollama/models` 복사, HF snapshot 복사, `HF_HUB_OFFLINE=1`)를 `docs/guides/`에 적는다.

**문서 갱신**
- `.env.dev.example`도 같은 모델 체계를 쓰되, 개발 PC(RTX 2060 12GB)에서는 `RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B`, `RERANK_DTYPE=fp16`을 기본값으로 둔다.
- CLAUDE.md의 GPU 절에는 Phase B 동시 적재 구성을 적는다. 재순위 BF16 약 8GB + 임베딩 8비트 약 4~5GB + LLM 4비트 약 8GB = 약 21~22GB이며, 실측값으로 갱신한다.
- CLAUDE.md의 아키텍처 요약·주요 파일 표·테스트 표도 v4로 갱신한다.

### 5.10 테스트

| 구분 | 내용 |
|---|---|
| **신규** | `test_units_builder`: ID 결정성, 중복 메일 제거, 채팅 30분 세션·방 키, 첨부 부모 연결(PST·EML 둘 다), 헤더 형식, 파트 분할, XLSX 행 블록(머리글 반복), DOCX 가상 페이지, PDF 빈 페이지 번호 유지 |
| | `test_document_parser`(보강): PPTX 비플레이스홀더 도형 회귀 테스트, 확장자 없는 파일 판별, XLS·HTML |
| | `test_email_clean`: `ref` 테스트 이식 + 중국어 가명 고정 자료 + 전달 메일 예외 |
| | `test_bm25_tokenizer`: `ref` 테스트 이식 + `FDB21-000963` 형식 코드 |
| | `test_lang_detect`: 한·영·중 혼용, 짧은 글 `und`, 혼용 20% 경계 |
| | `test_case_index`: 저장·로드·검색·필터·삭제·차원 불일치·토크나이저 버전 불일치·원자적 쓰기 |
| | `test_retrieval`: 가짜 검색기로 RRF 계산값 확인, 파트→단위 합치기, 재순위 후 꼬리 붙이기, 단일 검색기 depth |
| | `test_reranker_qwen3`: 모델을 모킹해서 입력 형식, 절단 집계, 점수 순서 확인 |
| | `test_embedding_service`: 질의에만 지시문이 붙는지, 정규화 확인 |
| **수정** | `test_rag_engine`, `test_indexing_pipeline`, `test_dashboard_api`, `test_cases_api`, `test_chat_api`, `test_e2e_pipeline` |
| **삭제(체크포인트 3 이후)** | `test_chunker`, `test_vector_store` |
| **가짜 모델** | `AST_FAKE_MODELS=1`이면 해시 기반 가짜 임베더와 토큰 겹침 기반 가짜 재순위기를 쓴다. 단위·통합 테스트와 실험 selftest에서 GPU 없이 돈다 |

- **완료 기준:** `pytest tests/ -q` 전체 통과, `ruff check src/ tests/ experiments_v4/` 통과.

---

## 6. 실험 하네스 v4(`experiments_v4/`)

### 6.1 v3.3 하네스 처리

어제 전달한 v3.3 하네스는 **저장소에 넣지 않는다.** 청크→단위 매핑, 절삭·청킹 비교, 볼록 결합, Recall@2k처럼 v4에서 없앤 실험을 전제로 만들어졌고, 어댑터 계층도 v4에서는 필요 없다. 실험이 서비스의 `retrieve()`를 직접 호출하기 때문이다.

필요한 함수만 옮긴다(`ref/from_v33/README.md`).
- `stats.py`: `bootstrap_ci`, `holm`, `quadratic_weighted_kappa(n_levels=3)`, `percentile`
- `text.py`: `TokenCounter`
- `pooling.py`: `import_judgments`, `agreement`의 로직

### 6.2 구조

```
experiments_v4/
├─ README.md                # 실행 순서 요약
├─ config.yaml              # 실행 목록·k값·경로 (§6.4)
├─ .gitignore               # data/ runs/ results/ cache/
├─ data/                    # (gitignore) queries.jsonl, qrels.txt, judgments/*.csv
├─ ev4/
│  ├─ __main__.py           # CLI (§6.5)
│  ├─ io.py                 # queries/qrels/run(TREC) 입출력
│  ├─ corpus.py             # 단위 통계 → results/T1_corpus.md
│  ├─ run.py                # retrieve() 호출 → runs/{RUN}.trec, runs/{RUN}.timing.jsonl
│  ├─ pool.py               # 상위 10 합집합 → 판정 시트
│  ├─ translate.py          # 로컬 gemma4:12b 번역(판정 참고용, 캐시)
│  ├─ judgments.py          # 판정 CSV → qrels, 일치도 (v3.3 이식)
│  ├─ evaluate.py           # ranx 지표·CI·검정·분해 → results/*.md
│  ├─ latency.py            # 응답 시간·VRAM·생성 시간
│  ├─ failures.py           # 실패 분석 시트
│  ├─ env.py                # 환경 기록 → results/env.json
│  └─ stats.py              # v3.3 이식
└─ tests/test_ev4.py        # 가짜 모델로 전체 흐름 selftest
```

### 6.3 데이터 형식

**`data/queries.jsonl`**

```json
{"qid": "q001", "text": "3월 정산 건에서 결재를 건너뛴 정황", "type": "semantic", "split": "test", "query_lang": "ko", "note": ""}
```

- **`type`:** 3유형이다. 초안은 `exact`(인명·번호·금액), `semantic`(의미·정황·회피 표현), `relation`(관계·맥락)이며, 정의는 체크포인트 2에서 사용자가 확정한다.
- **`split`:** `test`(평가 60~80개)와 `dev`(20개)로 나눈다. dev는 동작 점검에만 쓴다.
- **질의 작성:** 사용자(감사인)가 직접 하고, 어떤 시스템 결과도 보기 전에 확정한다. 기존 질의·판정 자산은 없으므로 새로 만든다(V14).
- **현지어 질의:** 결정적 증거가 한국어·영어 외 언어인 질의를 넣는다. 평가 코퍼스에서 실제로 확인된 언어로만 만든다(현재 메일함 기준 중국어). 목표는 평가 질의의 10~15%이고, 최소 5개다. 이에 못 미치면 언어별 결과는 사례 수준으로만 보고한다.

**판정 시트 `data/judgments/pool_{A|B}.csv`**(UTF-8 BOM, Excel 호환)

| 열 | 내용 |
|---|---|
| `qid`, `query_text`, `query_type` | 질의 |
| `unit_id`, `unit_type`, `date`, `lang` | 단위 |
| `header` | §5.1.3 헤더 |
| `text` | 단위 전체(파트 이어붙임). 6,000자를 넘으면 자르고 `…(이하 생략, 전체: unit_id로 조회)` 표시 |
| `translation_ko` | `lang`이 `ko`·`en`이 아니면 gemma4:12b 번역, 아니면 빈칸 |
| `grade` | 판정자가 입력. 0 무관 / 1 관련 / 2 결정적 |
| `annotator`, `note` | 판정자, 메모 |

- **블라인드:** 어느 실행에서 찾았는지는 시트에 넣지 않는다. 출처는 `runs/pool_provenance.jsonl`에 따로 둔다.
- **정렬:** 질의 안에서 단위 순서를 무작위로 섞는다(seed 고정).
- **판정 분담:** A는 전체를 판정한다. B는 질의 유형별 층화로 질의 25%를 골라 해당 질의 전체를 판정한다(비율은 사용자 확정).
- **중재:** 불일치 단위는 `adjudication.csv`로 만들고, 중재 등급을 넣어 다시 가져온다.

**`data/qrels.txt`**(최종, TREC 형식): `qid 0 unit_id grade`

### 6.4 실행 목록(`config.yaml`)

```yaml
case_id: "<CASE_ID>"
depth: 100
pool_depth: 10
seed: 20261015
runs:
  R1: {profile: nohdr-qwen3, use_bm25: true,  use_dense: false, rerank: false}
  R2: {profile: hdr-qwen3,   use_bm25: true,  use_dense: false, rerank: false}
  R3: {profile: nohdr-qwen3, use_bm25: false, use_dense: true,  rerank: false}
  R4: {profile: hdr-qwen3,   use_bm25: false, use_dense: true,  rerank: false}
  R5: {profile: nohdr-qwen3, use_bm25: true,  use_dense: true,  rerank: false}
  R6: {profile: hdr-qwen3,   use_bm25: true,  use_dense: true,  rerank: false}
  R7: {profile: nohdr-qwen3, use_bm25: true,  use_dense: true,  rerank: true, reranker: qwen3}
  R8: {profile: hdr-qwen3,   use_bm25: true,  use_dense: true,  rerank: true, reranker: qwen3}
  R9: {profile: hdr-bgem3,   use_bm25: true,  use_dense: true,  rerank: true, reranker: bge}
pool_runs: [R1, R2, R3, R4, R5, R6, R7, R8, R9]
comparisons:            # 주 지표 nDCG@10, 한 계열로 Holm 보정
  ladder_hdr:   [[R2, R6], [R4, R6], [R6, R8]]
  ladder_nohdr: [[R1, R5], [R3, R5], [R5, R7]]
  header:       [[R1, R2], [R3, R4], [R5, R6], [R7, R8]]
  model:        [[R9, R8]]
latency_runs: [R6, R8, R9]
```

- k값(50/50/60/30)은 `RetrievalConfig` 기본값을 쓰고, config에서 덮어쓰지 않는다.
- R9의 BM25는 헤더 on 토큰화를 쓰므로 R8과 같다. 차이는 임베딩·재순위 모델뿐이다.

### 6.5 명령 흐름

```bash
# 0) 프로필 생성 (서비스 기본 hdr-qwen3 은 인덱싱 시 이미 생성)
python -m scripts.build_profiles <CASE_ID> --profiles nohdr-qwen3 hdr-bgem3

# 1) 단위 통계
python -m ev4 corpus

# 2) dev 동작 점검 (모델 선정 아님: 결과가 비거나 오류가 없는지, 시간·VRAM 예비 측정)
python -m ev4 run --split dev --runs R2 R4 R6 R8 R9

# 3) 본 실행
python -m ev4 run --split test --runs all

# 4) 판정 시트 → (감사인 판정) → 가져오기 → 일치도 → 중재 → 최종 qrels
python -m ev4 pool --split test
python -m ev4 judgments import data/judgments/pool_A.csv data/judgments/pool_B.csv
python -m ev4 judgments agreement
python -m ev4 judgments finalize          # 중재 반영 → data/qrels.txt

# 5) 평가·응답 시간·실패 분석·환경 기록
python -m ev4 evaluate
python -m ev4 latency
python -m ev4 failures
python -m ev4 env

# selftest (GPU·실데이터 없이)
AST_FAKE_MODELS=1 python -m ev4 selftest
```

### 6.6 평가(`evaluate.py`)

- **지표**(`ranx`, 단위 기준, 실행 깊이 100):
  - `recall@10`, `ndcg@10`, `mrr@10`(ranx 기본 `rel_lvl=1`, 등급 ≥1을 관련으로 봄)
  - `recall@10-l2`(결정적 증거 재현율)
  - `recall@100`(참고)
  - `ndcg`는 등급을 선형 이득(1, 2)으로 쓴다.
- **주의(ranx 동작 확인 결과):** `-l2` 지표는 등급 2 단위가 없는 질의를 0으로 계산한다. `recall@10-l2`는 **등급 2 단위가 1개 이상인 질의만** 골라서 계산하고, 대상 질의 수를 함께 표시한다.
- **신뢰구간:** 질의 재표집 bootstrap 1,000회로 95% 구간을 낸다(`stats.bootstrap_ci`).
- **유의성 검정:**
  - `ranx.compare(..., stat_test="fisher", n_permutations=10000)`의 p값을 쓴다.
  - §6.4 `comparisons` 11개에 Holm 보정을 한다. 주 지표는 nDCG@10이다.
  - 다른 지표는 기술 통계로만 보고한다.
- **분해:**
  - 질의 유형별(3유형).
  - 증거 언어별: 질의의 등급 2 단위 언어로 `ko`/`en`/`other`를 정한다. 여러 언어면 `other`가 하나라도 있을 때 `other`, 아니면 다수결이다.
  - 교차 언어 여부: `query_lang ≠ 증거 언어`.
  - 분해 표에는 R2·R4·R6·R8·R9와 각 그룹의 질의 수 n을 함께 낸다.
- **출력:** `results/tables.md`(§6.8의 T1~T7), `results/per_query.csv`, `results/comparisons.csv`

### 6.7 응답 시간·VRAM·실패 분석

- **VRAM 운용(V11):** `run`(R1~R9)은 LLM을 내린 상태(`ollama stop gemma4:12b`)에서 실행한다. 검색 품질은 LLM과 무관하고, 재순위 활성 메모리 여유를 확보하기 위해서다. 세 모델 동시 적재는 `latency`에서만 한다(서비스 조건).
- **응답 시간**(`latency.py`):
  - 대상은 `latency_runs`이고, 평가 질의 전체를 쓴다. 측정 전 세 모델(임베딩·재순위·LLM)을 모두 적재한다.
  - 사전 실행 3개 후, 질의마다 실행 순서를 무작위로 바꿔 5회 반복한다.
  - `timings_ms` 단계별 p50/p95와 재순위 절단 비율을 보고한다.
- **생성 시간:** R8 상위 5개 단위를 컨텍스트로 넣어 `gemma4:12b` 응답을 생성한다. 첫 토큰까지 시간, 전체 시간, 초당 토큰 수의 p50/p95를 낸다(평가 질의 전체, 1회).
- **VRAM:** 측정 중 `nvidia-smi --query-gpu=memory.used` 최대치와 재순위 프로세스의 `torch.cuda.max_memory_allocated`를 기록한다. `ollama ps`·`ollama show`로 모델 태그와 양자화도 기록한다.
- **실패 분석**(`failures.py`):
  - 대상: R8에서 등급 2 단위가 상위 10에 없는 질의.
  - 행: 놓친 단위마다 한 행. 열은 qid, 유형, 질의, unit_id, 단위 유형, 언어, 토큰 수, R2·R4·R6·R7·R8에서의 순위(100위 밖이면 `>100`), `category`(빈칸), `note`.
  - `category` 선택지: 토큰화(정확 일치 실패) / 교차 언어 / 표현 불일치 / 긴 단위 희석 / 재순위 강등 / 헤더 영향 / 판정 이견 / 기타. 분류는 사용자가 한다.

### 6.8 산출물 표

| ID | 내용 | 생성 |
|---|---|---|
| T1 | 단위 통계: 유형별 단위·파트 수, 언어 분포, 토큰 길이(중앙값·p95·p99·최대), 분할된 단위 수, 중복 메일 제거 수, 정제로 줄어든 문자 비율(언어별), 부모 없는 첨부 수 | `corpus` |
| T2 | 사다리 표: R1~R8 × 지표(평균 [95% CI]) | `evaluate` |
| T3 | 헤더 효과: 4쌍의 차이, p(Holm) | `evaluate` |
| T4 | 질의 유형별 nDCG@10·Recall@10 | `evaluate` |
| T5 | 증거 언어별·교차 언어 | `evaluate` |
| T6 | 모델 비교 R8 vs R9(전체·언어별) | `evaluate` |
| T7 | 응답 시간·VRAM·생성 시간 | `latency` |
| F1 | 실패 분석 시트와 분류 집계 | `failures` |
| E1 | 환경: GPU, 드라이버, CUDA, OS, Python, 라이브러리 버전, Ollama 버전·모델 태그, HF 모델 리비전, 토크나이저 버전 | `env` |

### 6.9 논문 반영 위치(v3.5 → v4)

| 논문 위치 | 바뀌는 내용 | 근거 산출물 |
|---|---|---|
| 1.2절 기여 | "양식 인지 청킹" → "증거 단위 색인과 메타데이터 헤더" | — |
| 3.3절 | 4종 청커 → 단위 정의·헤더·분할·정제 | §5.1, T1 |
| 3.4절 | 점수 하한·검색 범위 50·품사 필터 삭제, 재순위 후보 30, 지시문 원문, `max_length`와 절단 비율 | §5.2~5.6, T7 |
| 3.5절 | ChromaDB → 케이스 폴더 색인 | §5.5 |
| [표 2]·3.2절 | 모델 태그, VRAM 실측 | E1, T7 |
| 4.1절 | [표 3] 단위·언어 분포, 질의 3유형·현지어 질의 수, 등급 0/1/2, 판정 일치도 | T1, `agreement` |
| 4.2~4.3절 | 실행 9회, ranx, 실험 환경 | §6.4, E1 |
| 4.4~4.7절 | 사다리·헤더·유형·언어·모델 비교·응답 시간 | T2~T7 |
| 4.8절 | 절삭 실험 → 실패 분석 | F1 |

---

## 7. 작업 순서와 체크포인트

| 단계 | 내용 | 완료 기준 |
|---|---|---|
| **0** | 읽기 전용 점검. `git status`·`git log`로 공개 main(62f4d5a)과 차이 확인. §3 C1~C10을 로컬 코드와 대조. 이메일 정제 함수 유무, `.env` 실제 값, 기존 테스트 통과 여부(`pytest -q` 결과), Ollama 태그(`qwen3-embedding:4b-q8_0`, `gemma4:12b`) 존재, HF 가중치(`Qwen/Qwen3-Reranker-4B`, 선택적으로 `Qwen/Qwen3-Embedding-4B`)를 사내에 반입할 수 있는지 확인 | §8 형식 보고 |
| ⏸ **체크포인트 1** | 사용자가 차이점과 진행 여부 확인 | |
| **1** | Python 3.12 venv 구성. `origin/feat/search-quality`에서 `v4-simplify` 브랜치 생성(V2). `src/units/`(스키마·빌더·헤더·정제·분할·언어 감지), `src/text/bm25_tokenizer.py`, 파서 수정(PDF `pages`·빈 페이지 자리, PPTX 버그, EML 첨부 파싱, 첨부 순회, xls·html·확장자 판별, 스캔 페이지 OCR, 채팅 방 키·`chat_type`), 테스트 | 신규 테스트 통과, 기존 테스트 깨지지 않음. 보고에 feat 브랜치 커밋 분류 포함 |
| **2** | `scripts/unit_stats.py <case_id>`: 실데이터로 단위만 생성(임베딩 없음) → T1 초안. 정제 전후 샘플 20건(언어별, 전달 메일 포함)을 로컬 파일로 저장해서 사용자가 확인할 수 있게 함. OCR 처리 시간과 OCR 페이지 수도 보고 | T1 초안 |
| ⏸ **체크포인트 2** | 사용자 결정: 헤더 형식, `UNIT_MAX_PART_TOKENS`, 질의 3유형 정의, 현지어 질의 수, B 판정 비율, `EMBED_QUERY_TASK`·`RERANK_INSTRUCTION` 문구, 서비스 기본 프로필(헤더 on) | |
| **3** | 임베딩·재순위 교체. 개발 PC에서는 가짜 모델과 `Qwen3-Reranker-0.6B`(FP16)로 동작 확인. `check_embed_parity.py`와 세 모델 동시 적재 VRAM 확인은 **A5000에서** 실행(6단계 시작 시 먼저 수행해도 됨) | 개발 PC 동작 확인. A5000 항목은 실행 일정 보고 |
| **4** | `case_index`·`retrieve()`·엔진·API·대시보드·설정·파이프라인 연결. 가짜 모델 e2e | `pytest tests/` 전체 통과, `ruff` 통과 |
| **5** | `experiments_v4/` 구현과 selftest | `AST_FAKE_MODELS=1 python -m ev4 selftest` 통과(T1~T7 표 생성까지) |
| ⏸ **체크포인트 3** | 변경 요약(파일·줄 수), 삭제 예정 목록(Chroma·청커·`run_benchmark.py`·관련 테스트·설정·docker-compose `chromadb` 서비스·`chromadb` 의존성), CLAUDE.md 갱신안 보고. 사용자 확인 후 삭제와 병합 | |
| **6** | (전제: 평가 코퍼스 범위 확정, V14) A5000 실데이터: 임베딩 일치·VRAM 확인 → 케이스 재색인(서비스 프로필) → 실험 프로필 생성 → dev 점검 → test 실행(LLM 내림) → 판정 시트 → (사용자·감사인 판정) → 평가·응답 시간·실패 분석·환경 기록 | `results/` 전체, 판정 대기 시 그 시점까지 결과 보고 |

---

## 8. 보고 형식

단계마다 아래 순서로 짧게 보고한다.

1. **한 일:** 변경·추가한 파일과 요지(파일당 한 줄)
2. **확인 결과:** 테스트 수와 통과 여부, 측정값(있으면)
3. **이 문서와 다르게 한 것과 이유:** 없으면 "없음"
4. **사용자가 결정할 것:** 없으면 "없음"
5. **다음 단계**

0단계 보고에는 아래 항목을 반드시 포함한다.
- 로컬과 공개 main의 차이 요약
- C1~C10 대조표
- Ollama 태그·HF 가중치 반입 가능 여부
- 기존 테스트 결과
- 예상 작업량(단계별)

---

## 부록: 참조 코드(`docs/v4/ref/`)

| 파일 | 내용 | 상태 |
|---|---|---|
| `script_tokenizer.py` | 문자 체계별 BM25 토크나이저 | 테스트 통과(kiwipiepy 0.24.0) |
| `email_quote.py` | 다국어 인용 제거 | 테스트 통과 |
| `qwen3_formats.py` | Qwen3 임베딩 질의 형식, 재순위 템플릿·점수 계산 클래스 | 형식 테스트 통과, 모델 실행은 3단계에서 확인 |
| `test_ref.py` | 위 세 모듈 테스트 14개 | `pytest docs/v4/ref/test_ref.py -q` |
| `from_v33/` | v3.3 하네스에서 옮길 `stats.py`·`text.py`·`pooling.py`와 이식 안내 | 원본 그대로, 이식 시 수정 |
