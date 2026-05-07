import { useState, useEffect, useCallback } from 'react';
import { casesApi, indexingApi, type CaseData, type IndexingProgress } from '../api';

/** 상태 배지 색상 */
function statusColor(s: string) {
  switch (s) {
    case 'created': return 'bg-gray-200 text-gray-700';
    case 'indexing': return 'bg-blue-100 text-blue-700';
    case 'ready': return 'bg-green-100 text-green-700';
    case 'archived': return 'bg-yellow-100 text-yellow-700';
    case 'error': return 'bg-red-100 text-red-700';
    default: return 'bg-gray-100 text-gray-600';
  }
}

export default function AdminPage() {
  const [cases, setCases] = useState<CaseData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // 새 케이스 폼
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [newDocPaths, setNewDocPaths] = useState('');
  const [newPstPaths, setNewPstPaths] = useState('');

  // 인덱싱 진행률
  const [progress, setProgress] = useState<Record<string, IndexingProgress>>({});

  const loadCases = useCallback(async () => {
    try {
      const data = await casesApi.list();
      setCases(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => { loadCases(); }, [loadCases]);

  // 인덱싱 중인 케이스 진행률 폴링
  useEffect(() => {
    const indexingCases = cases.filter(c => c.status === 'indexing');
    if (indexingCases.length === 0) return;

    const TERMINAL_PHASES = new Set(['completed', 'error', 'cancelled']);
    const interval = setInterval(async () => {
      for (const c of indexingCases) {
        try {
          const p = await indexingApi.progress(c.case_id);
          setProgress(prev => ({ ...prev, [c.case_id]: p }));
          if (TERMINAL_PHASES.has(p.phase)) {
            loadCases();
          }
        } catch { /* ignore */ }
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [cases, loadCases]);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setLoading(true);
    setError('');
    try {
      await casesApi.create(
        newName.trim(),
        newDesc.trim(),
        newPstPaths.split('\n').map(s => s.trim().replace(/^["']+|["']+$/g, '').trim()).filter(Boolean),
        newDocPaths.split('\n').map(s => s.trim().replace(/^["']+|["']+$/g, '').trim()).filter(Boolean),
      );
      setNewName(''); setNewDesc(''); setNewDocPaths(''); setNewPstPaths('');
      await loadCases();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const handleIndex = async (caseId: string) => {
    try {
      await indexingApi.start(caseId);
      await loadCases();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleStop = async (caseId: string) => {
    try {
      await indexingApi.stop(caseId);
      await loadCases();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleArchive = async (caseId: string) => {
    try {
      await casesApi.archive(caseId);
      await loadCases();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleDelete = async (caseId: string) => {
    if (!confirm('이 케이스를 삭제하시겠습니까?')) return;
    try {
      await casesApi.delete(caseId);
      await loadCases();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Admin - 케이스 관리</h2>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-2 rounded text-sm">
          {error}
          <button className="ml-2 underline" onClick={() => setError('')}>닫기</button>
        </div>
      )}

      {/* 새 케이스 생성 */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 space-y-3">
        <h3 className="font-semibold text-gray-700">새 케이스 생성</h3>
        <div className="grid grid-cols-2 gap-3">
          <input
            className="border border-gray-300 rounded px-3 py-2 text-sm"
            placeholder="케이스명"
            value={newName}
            onChange={e => setNewName(e.target.value)}
          />
          <input
            className="border border-gray-300 rounded px-3 py-2 text-sm"
            placeholder="설명 (선택)"
            value={newDesc}
            onChange={e => setNewDesc(e.target.value)}
          />
        </div>
        <textarea
          className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
          placeholder="문서 폴더 경로 (줄바꿈 구분)"
          rows={2}
          value={newDocPaths}
          onChange={e => setNewDocPaths(e.target.value)}
        />
        <textarea
          className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
          placeholder="PST 파일 경로 (줄바꿈 구분)"
          rows={2}
          value={newPstPaths}
          onChange={e => setNewPstPaths(e.target.value)}
        />
        <button
          className="bg-blue-600 text-white px-4 py-2 rounded text-sm hover:bg-blue-700 disabled:opacity-50"
          onClick={handleCreate}
          disabled={loading || !newName.trim()}
        >
          생성
        </button>
      </div>

      {/* 케이스 목록 */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-gray-700">케이스 목록</h3>
          <button
            className="text-sm text-blue-600 hover:underline"
            onClick={loadCases}
          >
            새로고침
          </button>
        </div>

        {cases.length === 0 ? (
          <p className="text-gray-500 text-sm">등록된 케이스가 없습니다.</p>
        ) : (
          <div className="space-y-2">
            {cases.map(c => (
              <div key={c.case_id} className="bg-white rounded-lg border border-gray-200 p-4">
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-gray-800">{c.name}</span>
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${statusColor(c.status)}`}>
                        {c.status}
                      </span>
                    </div>
                    {c.description && <p className="text-sm text-gray-500">{c.description}</p>}
                    <div className="text-xs text-gray-400 space-x-3">
                      <span>ID: {c.case_id.slice(0, 8)}</span>
                      <span>문서: {c.total_documents}개</span>
                      <span>청크: {c.total_chunks}개</span>
                      {c.doc_paths.length > 0 && <span>경로: {c.doc_paths.join(', ')}</span>}
                    </div>
                    {c.error_message && (
                      <p className="text-xs text-red-500">오류: {c.error_message}</p>
                    )}

                    {/* 인덱싱 진행률 */}
                    {c.status === 'indexing' && progress[c.case_id] && (
                      <div className="mt-2">
                        <div className="flex items-center gap-2 text-xs text-blue-600">
                          <span>{progress[c.case_id].phase}</span>
                          <span>{progress[c.case_id].processed_files}/{progress[c.case_id].total_files} 파일</span>
                          <span>{progress[c.case_id].total_chunks} 청크</span>
                          <span>{progress[c.case_id].elapsed}</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
                          <div
                            className="bg-blue-600 h-1.5 rounded-full transition-all"
                            style={{ width: `${progress[c.case_id].progress_percent}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>

                  {/* 액션 버튼 */}
                  <div className="flex gap-1.5 shrink-0">
                    {(c.status === 'created' || c.status === 'ready' || c.status === 'error') && (
                      <button
                        className="px-3 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700"
                        onClick={() => handleIndex(c.case_id)}
                      >
                        인덱싱
                      </button>
                    )}
                    {c.status === 'indexing' && (
                      <button
                        className="px-3 py-1 bg-orange-500 text-white text-xs rounded hover:bg-orange-600"
                        onClick={() => handleStop(c.case_id)}
                      >
                        중지
                      </button>
                    )}
                    {c.status === 'ready' && (
                      <button
                        className="px-3 py-1 bg-yellow-500 text-white text-xs rounded hover:bg-yellow-600"
                        onClick={() => handleArchive(c.case_id)}
                      >
                        보관
                      </button>
                    )}
                    <button
                      className="px-3 py-1 bg-red-500 text-white text-xs rounded hover:bg-red-600"
                      onClick={() => handleDelete(c.case_id)}
                    >
                      삭제
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
