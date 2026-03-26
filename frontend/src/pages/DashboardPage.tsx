import { useState, useEffect, useRef, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell,
} from 'recharts';
import * as d3 from 'd3';
import { chatApi, dashboardApi, type CaseInfo, type DashboardData, type ParticipantNode, type ParticipantEdge } from '../api';

// === Color constants ===

const SOURCE_COLORS: Record<string, string> = {
  email: '#3b82f6',
  teams_chat: '#8b5cf6',
  document: '#10b981',
  attachment: '#f59e0b',
  unknown: '#6b7280',
};

const PIE_COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#6b7280'];

// === Network Graph (D3 Force Simulation) ===

interface NetworkGraphProps {
  nodes: ParticipantNode[];
  edges: ParticipantEdge[];
}

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  message_count: number;
  source_types: string[];
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  weight: number;
}

function NetworkGraph({ nodes, edges }: NetworkGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = svgRef.current.clientWidth || 700;
    const height = 500;

    svg.attr('viewBox', `0 0 ${width} ${height}`);

    const simNodes: SimNode[] = nodes.map(n => ({ ...n }));
    const simLinks: SimLink[] = edges
      .filter(e => simNodes.some(n => n.id === e.source) && simNodes.some(n => n.id === e.target))
      .map(e => ({ source: e.source, target: e.target, weight: e.weight }));

    const maxCount = Math.max(...simNodes.map(n => n.message_count), 1);
    const radiusScale = d3.scaleSqrt().domain([1, maxCount]).range([6, 28]);
    const maxWeight = Math.max(...(simLinks.map(l => l.weight).concat([1])));
    const widthScale = d3.scaleLinear().domain([1, maxWeight]).range([1, 6]);

    const simulation = d3.forceSimulation(simNodes)
      .force('link', d3.forceLink<SimNode, SimLink>(simLinks).id(d => d.id).distance(120))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide<SimNode>().radius(d => radiusScale(d.message_count) + 4));

    const g = svg.append('g');

    // Zoom
    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.3, 3])
        .on('zoom', (event) => g.attr('transform', event.transform))
    );

    // Edges
    const link = g.append('g')
      .selectAll('line')
      .data(simLinks)
      .join('line')
      .attr('stroke', '#d1d5db')
      .attr('stroke-width', d => widthScale(d.weight))
      .attr('stroke-opacity', 0.6);

    // Nodes
    const node = g.append('g')
      .selectAll<SVGCircleElement, SimNode>('circle')
      .data(simNodes)
      .join('circle')
      .attr('r', d => radiusScale(d.message_count))
      .attr('fill', d => {
        if (d.source_types.includes('email') && d.source_types.includes('teams_chat')) return '#6366f1';
        if (d.source_types.includes('email')) return SOURCE_COLORS.email;
        if (d.source_types.includes('teams_chat')) return SOURCE_COLORS.teams_chat;
        return '#6b7280';
      })
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .attr('cursor', 'pointer')
      .on('click', (_event, d) => setSelectedNode(prev => prev === d.id ? null : d.id))
      .call(
        d3.drag<SVGCircleElement, SimNode>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null; d.fy = null;
          })
      );

    node.append('title').text(d => `${d.id} (${d.message_count})`);

    // Labels
    const label = g.append('g')
      .selectAll('text')
      .data(simNodes)
      .join('text')
      .text(d => d.id.length > 12 ? d.id.slice(0, 12) + '...' : d.id)
      .attr('font-size', 10)
      .attr('text-anchor', 'middle')
      .attr('dy', d => radiusScale(d.message_count) + 14)
      .attr('fill', '#374151')
      .attr('pointer-events', 'none');

    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as SimNode).x!)
        .attr('y1', d => (d.source as SimNode).y!)
        .attr('x2', d => (d.target as SimNode).x!)
        .attr('y2', d => (d.target as SimNode).y!);
      node.attr('cx', d => d.x!).attr('cy', d => d.y!);
      label.attr('x', d => d.x!).attr('y', d => d.y!);
    });

    return () => { simulation.stop(); };
  }, [nodes, edges]);

  const selectedInfo = selectedNode
    ? nodes.find(n => n.id === selectedNode)
    : null;

  const selectedEdges = selectedNode
    ? edges.filter(e => e.source === selectedNode || e.target === selectedNode)
      .sort((a, b) => b.weight - a.weight)
    : [];

  return (
    <div className="relative">
      <svg ref={svgRef} className="w-full border rounded-lg bg-white" style={{ height: 500 }} />
      {selectedInfo && (
        <div className="absolute top-4 right-4 bg-white border rounded-lg shadow-lg p-4 w-64">
          <div className="flex justify-between items-center mb-2">
            <h4 className="font-semibold text-sm truncate">{selectedInfo.id}</h4>
            <button onClick={() => setSelectedNode(null)} className="text-gray-400 hover:text-gray-600 text-xs">X</button>
          </div>
          <p className="text-xs text-gray-500 mb-1">Messages: {selectedInfo.message_count}</p>
          <p className="text-xs text-gray-500 mb-2">
            Channels: {selectedInfo.source_types.map(s =>
              <span key={s} className="inline-block px-1.5 py-0.5 rounded text-white text-[10px] mr-1"
                style={{ backgroundColor: SOURCE_COLORS[s] || '#6b7280' }}>{s}</span>
            )}
          </p>
          {selectedEdges.length > 0 && (
            <>
              <p className="text-xs font-medium text-gray-700 mb-1">Connections:</p>
              <ul className="text-xs text-gray-500 max-h-32 overflow-y-auto">
                {selectedEdges.slice(0, 10).map(e => (
                  <li key={`${e.source}-${e.target}`} className="flex justify-between">
                    <span className="truncate">{e.source === selectedNode ? e.target : e.source}</span>
                    <span className="text-gray-400 ml-2">{e.weight}</span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// === Summary Card ===

function StatCard({ label, value, color }: { label: string; value: number | string; color: string }) {
  return (
    <div className="bg-white rounded-lg border p-4">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>{typeof value === 'number' ? value.toLocaleString() : value}</p>
    </div>
  );
}

// === Main Dashboard Page ===

export default function DashboardPage() {
  const [cases, setCases] = useState<CaseInfo[]>([]);
  const [selectedCase, setSelectedCase] = useState('');
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Load available cases
  useEffect(() => {
    chatApi.cases().then(setCases).catch(() => {});
  }, []);

  // Load dashboard data when case changes
  const loadDashboard = useCallback(async (caseId: string) => {
    if (!caseId) { setData(null); return; }
    setLoading(true);
    setError('');
    try {
      const result = await dashboardApi.get(caseId);
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load dashboard');
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadDashboard(selectedCase); }, [selectedCase, loadDashboard]);

  // Prepare pie chart data
  const pieData = data
    ? Object.entries(data.source_type_counts).map(([name, value]) => ({ name, value }))
    : [];

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-gray-800">Communication Dashboard</h2>
        <select
          value={selectedCase}
          onChange={e => setSelectedCase(e.target.value)}
          className="border rounded-lg px-3 py-2 text-sm bg-white min-w-[260px]"
        >
          <option value="">-- Select case --</option>
          {cases.map(c => (
            <option key={c.case_id} value={c.case_id}>
              {c.name} ({c.total_chunks.toLocaleString()} chunks)
            </option>
          ))}
        </select>
      </div>

      {/* Loading / Error / Empty */}
      {loading && (
        <div className="flex items-center justify-center py-20">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
          <span className="ml-3 text-gray-500">Loading dashboard...</span>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-4 text-sm">{error}</div>
      )}

      {!loading && !error && !data && (
        <div className="text-center py-20 text-gray-400">
          Select a case to view communication analysis
        </div>
      )}

      {/* Dashboard Content */}
      {data && !loading && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="Total Chunks" value={data.total_chunks} color="text-gray-800" />
            <StatCard label="Email" value={data.source_type_counts.email || 0} color="text-blue-600" />
            <StatCard label="Teams Chat" value={data.source_type_counts.teams_chat || 0} color="text-purple-600" />
            <StatCard label="Documents" value={data.source_type_counts.document || 0} color="text-emerald-600" />
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Source Type Distribution */}
            <div className="bg-white rounded-lg border p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Source Distribution</h3>
              {pieData.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={50}
                      outerRadius={90}
                      dataKey="value"
                      label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`}
                      labelLine={false}
                    >
                      {pieData.map((entry, i) => (
                        <Cell key={entry.name} fill={SOURCE_COLORS[entry.name] || PIE_COLORS[i % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-gray-400 text-sm text-center py-10">No data</p>
              )}
            </div>

            {/* Timeline Chart */}
            <div className="bg-white rounded-lg border p-4 lg:col-span-2">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Monthly Communication Timeline</h3>
              {data.timeline.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={data.timeline}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="email_count" name="Email" stackId="a" fill={SOURCE_COLORS.email} />
                    <Bar dataKey="teams_chat_count" name="Teams Chat" stackId="a" fill={SOURCE_COLORS.teams_chat} />
                    <Bar dataKey="document_count" name="Document" stackId="a" fill={SOURCE_COLORS.document} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-gray-400 text-sm text-center py-10">No timeline data</p>
              )}
            </div>
          </div>

          {/* Network Graph */}
          <div className="bg-white rounded-lg border p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">
              Participant Network
              <span className="text-xs text-gray-400 ml-2 font-normal">
                (drag nodes, scroll to zoom, click node for details)
              </span>
            </h3>
            {data.participant_network.nodes.length > 0 ? (
              <NetworkGraph
                nodes={data.participant_network.nodes}
                edges={data.participant_network.edges}
              />
            ) : (
              <p className="text-gray-400 text-sm text-center py-10">No participant data available</p>
            )}
          </div>

          {/* Bottom Row: Participants + Topics */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Top Participants */}
            <div className="bg-white rounded-lg border p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Top Participants</h3>
              {data.top_participants.length > 0 ? (
                <div className="space-y-2 max-h-80 overflow-y-auto">
                  {data.top_participants.map((p, i) => (
                    <div key={p.id} className="flex items-center gap-3 text-sm">
                      <span className="text-gray-400 w-6 text-right">{i + 1}</span>
                      <div className="flex-1 min-w-0">
                        <div className="truncate font-medium text-gray-700">{p.id}</div>
                        <div className="flex gap-1 mt-0.5">
                          {p.source_types.map(s => (
                            <span key={s} className="text-[10px] px-1.5 py-0.5 rounded text-white"
                              style={{ backgroundColor: SOURCE_COLORS[s] || '#6b7280' }}>{s}</span>
                          ))}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-24 bg-gray-100 rounded-full h-2">
                          <div
                            className="h-2 rounded-full bg-blue-500"
                            style={{ width: `${(p.message_count / data.top_participants[0].message_count) * 100}%` }}
                          />
                        </div>
                        <span className="text-gray-500 w-10 text-right">{p.message_count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-400 text-sm text-center py-10">No participant data</p>
              )}
            </div>

            {/* Top Topics */}
            <div className="bg-white rounded-lg border p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Top Topics</h3>
              {data.top_topics.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {data.top_topics.map(t => (
                    <span
                      key={t.topic}
                      className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full bg-gray-100 text-sm text-gray-700 border"
                    >
                      {t.topic}
                      <span className="text-xs text-gray-400">({t.count})</span>
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-gray-400 text-sm text-center py-10">No topic data</p>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
