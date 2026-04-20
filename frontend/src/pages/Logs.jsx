import { useEffect, useState } from "react";
import { logsAPI } from "../services/api";
import { ScrollText, Activity, Mic, Database } from "lucide-react";

export default function Logs() {
  const [logs, setLogs] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([logsAPI.getHistory(50), logsAPI.getStats()])
      .then(([h, s]) => { setLogs(h); setStats(s); })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-screen">
      <div className="text-fintech-accent animate-pulse font-mono text-sm">Loading logs...</div>
    </div>
  );

  return (
    <div className="p-8 space-y-6 overflow-y-auto h-screen">
      <div>
        <h2 className="text-2xl font-bold text-white">Query Logs</h2>
        <p className="text-slate-400 text-sm mt-1">All queries, responses, and system metrics</p>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: "Total Queries", value: stats.total_queries, icon: Activity, color: "text-fintech-accent" },
            { label: "RAG Queries", value: stats.rag_queries, icon: Database, color: "text-fintech-green" },
            { label: "Voice Queries", value: stats.voice_queries, icon: Mic, color: "text-fintech-gold" },
            { label: "Avg Latency", value: `${stats.avg_latency_ms}ms`, icon: ScrollText, color: "text-fintech-200" },
          ].map(({ label, value, icon: Icon, color }) => (
            <div key={label} className="glass rounded-xl p-4">
              <div className={`${color} mb-2`}><Icon size={16} /></div>
              <p className="text-2xl font-bold text-white">{value}</p>
              <p className="text-xs text-slate-500 mt-0.5">{label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Log table */}
      <div className="glass rounded-2xl overflow-hidden">
        <div className="px-6 py-4 border-b border-fintech-600/20">
          <h3 className="text-white font-semibold text-sm">Recent Queries</h3>
        </div>
        {logs.length === 0 ? (
          <div className="p-8 text-center text-slate-500 text-sm">No queries yet. Start chatting!</div>
        ) : (
          <div className="divide-y divide-fintech-600/10">
            {logs.map((log, i) => (
              <div key={i} className="px-6 py-4 hover:bg-fintech-700/20 transition-colors">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-slate-200 truncate">{log.query}</p>
                    <p className="text-xs text-slate-500 mt-1 line-clamp-2">{log.answer}</p>
                  </div>
                  <div className="flex-shrink-0 text-right space-y-1">
                    <div className="flex items-center gap-2 justify-end">
                      {log.rag_used && <span className="text-xs text-fintech-green bg-fintech-green/10 px-2 py-0.5 rounded-full">RAG</span>}
                      {log.voice_input && <span className="text-xs text-fintech-gold bg-fintech-gold/10 px-2 py-0.5 rounded-full">Voice</span>}
                    </div>
                    <p className="text-xs text-slate-600 font-mono">{log.latency_ms}ms</p>
                    <p className="text-xs text-slate-600">{new Date(log.timestamp).toLocaleTimeString()}</p>
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
