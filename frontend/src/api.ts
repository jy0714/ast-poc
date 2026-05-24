/** 백엔드 API 클라이언트 */

const BASE = '';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// === Admin: Cases ===

export interface CaseData {
  case_id: string;
  name: string;
  description: string;
  status: string;
  pst_paths: string[];
  doc_paths: string[];
  total_documents: number;
  total_chunks: number;
  created_at: string;
  updated_at: string;
  error_message: string;
}

export const casesApi = {
  create: (name: string, description: string, pstPaths: string[], docPaths: string[]) =>
    request<CaseData>('/api/admin/cases/', {
      method: 'POST',
      body: JSON.stringify({ name, description, pst_paths: pstPaths, doc_paths: docPaths }),
    }),

  list: (status?: string) =>
    request<CaseData[]>(`/api/admin/cases/${status ? `?status=${status}` : ''}`),

  get: (id: string) => request<CaseData>(`/api/admin/cases/${id}`),

  updateSources: (id: string, pstPaths: string[], docPaths: string[]) =>
    request<CaseData>(`/api/admin/cases/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ pst_paths: pstPaths, doc_paths: docPaths }),
    }),

  delete: (id: string) => request<{ message: string }>(`/api/admin/cases/${id}`, { method: 'DELETE' }),

  archive: (id: string) =>
    request<CaseData>(`/api/admin/cases/${id}/archive`, { method: 'POST' }),
};

// === Admin: Indexing ===

export interface IndexingProgress {
  case_id: string;
  status: string;
  phase: string;
  total_files: number;
  processed_files: number;
  total_chunks: number;
  progress_percent: number;
  elapsed: string;
  errors: string[];
}

export const indexingApi = {
  start: (caseId: string) =>
    request<{ message: string; case_id: string }>('/api/admin/indexing/start', {
      method: 'POST',
      body: JSON.stringify({ case_id: caseId }),
    }),

  stop: (caseId: string) =>
    request<{ message: string }>(`/api/admin/indexing/stop/${caseId}`, { method: 'POST' }),

  progress: (caseId: string) =>
    request<IndexingProgress>(`/api/admin/indexing/progress/${caseId}`),

  increment: (caseId: string, docPaths: string[]) =>
    request<{ message: string }>(`/api/admin/indexing/increment/${caseId}`, {
      method: 'POST',
      body: JSON.stringify({ doc_paths: docPaths }),
    }),
};

// === Analyst: Chat ===

export interface ChatSource {
  content: string;
  source_type: string;
  filename: string;
  date: string;
  participants: string[];
  subject: string;
  relevance_score: number;
  search_method: string;
  // 이메일 전용 — 비-이메일에서는 빈 값
  sender?: string;
  recipients?: string[];
  cc?: string[];
  attachments?: string[];
  message_id?: string;
  in_reply_to?: string;
  // Office/PDF 작성자·수정자 추적
  author?: string;
  last_modified_by?: string;
  created_date?: string;
  last_modified?: string;
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
  security_mode: boolean;
  case_id: string;
  // 출처 인용 검증 (할루시네이션 감지)
  citation_count?: number;
  invalid_citations?: number[];
  uncited_response?: boolean;
}

export interface CaseInfo {
  case_id: string;
  name: string;
  description: string;
  total_documents: number;
  total_chunks: number;
}

export const chatApi = {
  query: (
    caseId: string,
    message: string,
    securityMode: boolean,
    filters?: Record<string, unknown>,
    signal?: AbortSignal,
  ) =>
    request<ChatResponse>('/api/analyst/chat/', {
      method: 'POST',
      body: JSON.stringify({ case_id: caseId, message, security_mode: securityMode, filters }),
      signal,
    }),

  cases: () => request<CaseInfo[]>('/api/analyst/chat/cases'),

  /** SSE 스트리밍 - EventSource 대신 fetch 사용 (POST 필요)
   *
   * signal로 중단 가능. abort 시 reader.read()가 AbortError를 던지고,
   * 서버는 연결 끊김(is_disconnected)을 감지해 토큰 생성을 멈춘다.
   */
  stream: async function* (
    caseId: string,
    message: string,
    securityMode: boolean,
    signal?: AbortSignal,
  ) {
    const res = await fetch(`${BASE}/api/analyst/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ case_id: caseId, message, security_mode: securityMode }),
      signal,
    });

    if (!res.ok || !res.body) throw new Error(`Stream error: ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim();
          if (data === '[DONE]') return;
          try {
            yield JSON.parse(data);
          } catch {
            // skip
          }
        }
      }
    }
  },
};

// === Analyst: Dashboard ===

export interface ParticipantNode {
  id: string;
  message_count: number;
  source_types: string[];
}

export interface ParticipantEdge {
  source: string;
  target: string;
  weight: number;
}

export interface TimelinePoint {
  month: string;
  email_count: number;
  teams_chat_count: number;
  document_count: number;
}

export interface TopicItem {
  topic: string;
  count: number;
}

export interface DashboardData {
  case_id: string;
  case_name: string;
  total_chunks: number;
  source_type_counts: Record<string, number>;
  top_participants: ParticipantNode[];
  participant_network: {
    nodes: ParticipantNode[];
    edges: ParticipantEdge[];
  };
  timeline: TimelinePoint[];
  top_topics: TopicItem[];
}

export const dashboardApi = {
  get: (caseId: string) => request<DashboardData>(`/api/analyst/dashboard/${caseId}`),
};
