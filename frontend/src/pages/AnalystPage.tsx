import { useState, useEffect, useRef } from 'react';
import { chatApi, type CaseInfo, type ChatSource, type TokenUsage } from '../api';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: ChatSource[];
  loading?: boolean;
  uncited?: boolean;  // LLM이 출처 인용 없이 답변 (할루시네이션 위험)
  tokenUsage?: TokenUsage;  // 토큰 사용량 (비용 산정용)
}

export default function AnalystPage() {
  const [cases, setCases] = useState<CaseInfo[]>([]);
  const [selectedCase, setSelectedCase] = useState<string>('');
  const [secureMode, setSecureMode] = useState(true);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  // 진행 중인 스트리밍 요청의 AbortController. 매 전송마다 새로 생성.
  const abortRef = useRef<AbortController | null>(null);

  // 케이스 목록 로드
  useEffect(() => {
    chatApi.cases().then(setCases).catch(() => {});
  }, []);

  // 언마운트 시 진행 중인 요청 중단 (메모리/연결 누수 방지)
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  // 스크롤 자동 이동
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || !selectedCase || isStreaming) return;

    setInput('');
    setError('');
    setMessages(prev => [...prev, { role: 'user', content: text }]);

    // 어시스턴트 메시지 (로딩)
    const assistantIdx = messages.length + 1;
    setMessages(prev => [...prev, { role: 'assistant', content: '', loading: true }]);
    setIsStreaming(true);

    // 이전 controller가 남아있으면 정리하고 항상 새로 생성 (연속 빠른 중단 대응)
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    let fullContent = '';
    let sources: ChatSource[] = [];
    let uncited = false;
    let tokenUsage: TokenUsage | undefined;

    try {
      for await (const event of chatApi.stream(selectedCase, text, secureMode, controller.signal)) {
        if (event.type === 'token') {
          fullContent += event.content;
          setMessages(prev => {
            const updated = [...prev];
            updated[assistantIdx] = { role: 'assistant', content: fullContent, loading: true };
            return updated;
          });
        } else if (event.type === 'sources') {
          sources = event.sources;
          uncited = event.uncited_response === true;
        } else if (event.type === 'token_usage') {
          tokenUsage = {
            input_tokens: event.input_tokens,
            output_tokens: event.output_tokens,
            total_tokens: event.total_tokens,
            source: event.source,
          };
        } else if (event.type === 'error') {
          setError(event.message);
        }
      }

      // 스트리밍 정상 완료
      setMessages(prev => {
        const updated = [...prev];
        updated[assistantIdx] = {
          role: 'assistant',
          content: fullContent || '응답을 생성하지 못했습니다.',
          sources,
          loading: false,
          uncited,
          tokenUsage,
        };
        return updated;
      });
    } catch (e: unknown) {
      // 의도적 중단(AbortError)과 네트워크/서버 에러를 구분
      const aborted = e instanceof DOMException && e.name === 'AbortError';
      if (aborted) {
        // 현재까지 모은 내용 유지 + 중단 표시. fallback 호출 안 함.
        setMessages(prev => {
          const updated = [...prev];
          const stoppedContent = fullContent
            ? `${fullContent}\n\n[응답이 중단되었습니다]`
            : '[응답이 중단되었습니다]';
          updated[assistantIdx] = {
            role: 'assistant',
            content: stoppedContent,
            sources,
            loading: false,
          };
          return updated;
        });
      } else {
        // 네트워크/스트리밍 실패 → 동기 방식 fallback (같은 signal 전달)
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        try {
          const result = await chatApi.query(
            selectedCase, text, secureMode, undefined, controller.signal,
          );
          setMessages(prev => {
            const updated = [...prev];
            updated[assistantIdx] = {
              role: 'assistant',
              content: result.answer,
              sources: result.sources,
              loading: false,
              uncited: result.uncited_response === true,
              tokenUsage: result.token_usage,
            };
            return updated;
          });
          setError('');
        } catch (e2: unknown) {
          // fallback도 중단됐으면 중단 표시, 아니면 오류
          const fallbackAborted = e2 instanceof DOMException && e2.name === 'AbortError';
          setMessages(prev => {
            const updated = [...prev];
            updated[assistantIdx] = {
              role: 'assistant',
              content: fallbackAborted
                ? '[응답이 중단되었습니다]'
                : `오류: ${e2 instanceof Error ? e2.message : String(e2)}`,
              loading: false,
            };
            return updated;
          });
          if (fallbackAborted) setError('');
        }
      }
    } finally {
      // 이 요청의 controller가 아직 현재 것이면 정리
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
      setIsStreaming(false);
      inputRef.current?.focus();
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const caseName = cases.find(c => c.case_id === selectedCase)?.name;

  return (
    <div className="h-[calc(100vh-52px)] flex flex-col">
      {/* 상단 바: 케이스 선택 + 보안 모드 */}
      <div className="bg-white border-b border-gray-200 px-4 py-2 flex items-center gap-4">
        <select
          className="border border-gray-300 rounded px-3 py-1.5 text-sm flex-1 max-w-xs"
          value={selectedCase}
          onChange={e => {
            setSelectedCase(e.target.value);
            setMessages([]);
          }}
        >
          <option value="">케이스 선택...</option>
          {cases.map(c => (
            <option key={c.case_id} value={c.case_id}>
              {c.name} ({c.total_documents}문서, {c.total_chunks}청크)
            </option>
          ))}
        </select>

        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <div
            className={`relative w-10 h-5 rounded-full transition-colors ${secureMode ? 'bg-green-500' : 'bg-gray-300'}`}
            onClick={() => setSecureMode(!secureMode)}
          >
            <div
              className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform ${secureMode ? 'translate-x-5' : 'translate-x-0.5'}`}
            />
          </div>
          <span className={secureMode ? 'text-green-700 font-medium' : 'text-gray-500'}>
            {secureMode ? '보안 ON' : '보안 OFF'}
          </span>
        </label>

        {caseName && (
          <span className="text-xs text-gray-400 ml-auto">
            {caseName}
          </span>
        )}
      </div>

      {/* 메시지 영역 */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.length === 0 && selectedCase && (
          <div className="text-center text-gray-400 mt-20">
            <p className="text-lg">질문을 입력하세요</p>
            <p className="text-sm mt-1">이 케이스의 문서에서 답변을 찾아드립니다</p>
          </div>
        )}

        {!selectedCase && (
          <div className="text-center text-gray-400 mt-20">
            <p className="text-lg">케이스를 먼저 선택하세요</p>
            <p className="text-sm mt-1">인덱싱이 완료된 케이스만 표시됩니다</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-2xl rounded-lg px-4 py-3 ${
              msg.role === 'user'
                ? 'bg-blue-600 text-white'
                : 'bg-white border border-gray-200 text-gray-800'
            }`}>
              {/* 메시지 내용 */}
              <div className="whitespace-pre-wrap text-sm leading-relaxed">
                {msg.content}
                {msg.loading && <span className="inline-block w-1.5 h-4 bg-gray-400 animate-pulse ml-0.5 align-middle" />}
              </div>

              {/* 출처 인용 없음 경고 (할루시네이션 위험) */}
              {!msg.loading && msg.uncited && (
                <div className="mt-2 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-1.5">
                  ⚠ 이 응답에는 출처 인용이 포함되어 있지 않습니다. 내용을 직접 검증해 주세요.
                </div>
              )}

              {/* 출처 */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-100">
                  <p className="text-xs font-medium text-gray-500 mb-2">출처 ({msg.sources.length}건)</p>
                  <div className="space-y-1.5">
                    {msg.sources.map((s, j) => (
                      <SourceCard key={j} source={s} />
                    ))}
                  </div>
                </div>
              )}

              {/* 토큰 사용량 (비용 산정용) */}
              {!msg.loading && msg.tokenUsage && (
                <div className="mt-2 text-[11px] text-gray-400">
                  토큰: 입력 {msg.tokenUsage.input_tokens.toLocaleString()} · 출력{' '}
                  {msg.tokenUsage.output_tokens.toLocaleString()} · 합계{' '}
                  {msg.tokenUsage.total_tokens.toLocaleString()}
                  {msg.tokenUsage.source === 'estimated' && ' (추정치)'}
                </div>
              )}
            </div>
          </div>
        ))}

        {error && (
          <div className="text-center text-red-500 text-sm">{error}</div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* 입력 바 */}
      <div className="border-t border-gray-200 bg-white px-4 py-3">
        <div className="max-w-3xl mx-auto flex gap-2">
          <textarea
            ref={inputRef}
            className="flex-1 border border-gray-300 rounded-lg px-4 py-2.5 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            placeholder={selectedCase ? '질문을 입력하세요... (Enter로 전송, Shift+Enter 줄바꿈)' : '케이스를 먼저 선택하세요'}
            rows={1}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={!selectedCase || isStreaming}
          />
          {isStreaming ? (
            <button
              className="bg-red-500 text-white px-5 py-2.5 rounded-lg text-sm font-medium hover:bg-red-600"
              onClick={handleStop}
            >
              답변 멈추기
            </button>
          ) : (
            <button
              className="bg-blue-600 text-white px-5 py-2.5 rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={handleSend}
              disabled={!selectedCase || !input.trim()}
            >
              전송
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/** 출처 카드 컴포넌트 */
function SourceCard({ source }: { source: ChatSource }) {
  const [expanded, setExpanded] = useState(false);

  const typeLabel: Record<string, string> = {
    email: '이메일',
    teams_chat: 'Teams 채팅',
    document: '문서',
    attachment: '첨부파일',
  };

  return (
    <div
      className="bg-gray-50 rounded p-2 text-xs cursor-pointer hover:bg-gray-100 transition-colors"
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center gap-2">
        <span className="px-1.5 py-0.5 bg-gray-200 rounded text-[10px] font-medium">
          {typeLabel[source.source_type] || source.source_type}
        </span>
        {source.filename && <span className="text-gray-600 truncate">{source.filename}</span>}
        {source.subject && <span className="text-gray-500 truncate">- {source.subject}</span>}
        <span className="ml-auto text-gray-400">{(source.relevance_score * 100).toFixed(0)}%</span>
      </div>
      {expanded && (
        <div className="mt-2 text-gray-600 whitespace-pre-wrap leading-relaxed">
          {source.content}
          {source.source_type === 'email' ? (
            <div className="mt-2 space-y-0.5 text-gray-500">
              {source.sender && <div>보낸 사람: {source.sender}</div>}
              {source.recipients && source.recipients.length > 0 && (
                <div>받는 사람: {source.recipients.join(', ')}</div>
              )}
              {source.cc && source.cc.length > 0 && (
                <div>참조: {source.cc.join(', ')}</div>
              )}
              {source.date && <div>날짜: {source.date}</div>}
              {source.attachments && source.attachments.length > 0 && (
                <div>첨부파일: {source.attachments.join(', ')}</div>
              )}
              {source.message_id && (
                <div className="text-gray-400 truncate">Message-ID: {source.message_id}</div>
              )}
              {source.in_reply_to && (
                <div className="text-gray-400 truncate">In-Reply-To: {source.in_reply_to}</div>
              )}
            </div>
          ) : (
            <div className="mt-2 space-y-0.5 text-gray-500">
              {source.author && <div>작성자: {source.author}</div>}
              {source.last_modified_by && source.last_modified_by !== source.author && (
                <div>마지막 수정: {source.last_modified_by}</div>
              )}
              {source.created_date && <div>생성일: {source.created_date}</div>}
              {source.last_modified && source.last_modified !== source.created_date && (
                <div>수정일: {source.last_modified}</div>
              )}
              {source.date && !source.created_date && (
                <div className="text-gray-400">날짜: {source.date}</div>
              )}
              {source.participants.length > 0 && (
                <div className="text-gray-400">참여자: {source.participants.join(', ')}</div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
