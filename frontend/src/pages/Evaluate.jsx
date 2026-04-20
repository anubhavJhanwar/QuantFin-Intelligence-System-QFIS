import { useEffect, useState } from "react";
import { evaluateAPI } from "../services/api";
import { BarChart3, TrendingUp, AlertTriangle, CheckCircle } from "lucide-react";

const METRIC_LABELS = {
  bleu1: "BLEU-1", bleu4: "BLEU-4",
  rouge1: "ROUGE-1", rouge2: "ROUGE-2", rougeL: "ROUGE-L",
  exact_match: "Exact Match", f1: "F1 Score",
};

const MODEL_COLORS = {
  "Base Phi-2": "bg-slate-500",
  "Prompt-Engineered": "bg-fintech-400",
  "Fine-tuned (QLoRA)": "bg-fintech-green",
};

export default function Evaluate() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    evaluateAPI.getResults()
      .then(setData)
      .catch(() => setError("Evaluation results not found. Run backend/training/evaluate.py first."))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-screen">
      <div className="text-fintech-accent animate-pulse font-mono text-sm">Loading evaluation results...</div>
    </div>
  );

  if (error) return (
    <div className="p-8">
      <div className="glass rounded-2xl p-6 border border-yellow-500/30">
        <div className="flex items-center gap-3 text-yellow-400">
          <AlertTriangle size={20} />
          <p className="text-sm">{error}</p>
        </div>
      </div>
    </div>
  );

  const models = Object.keys(data.results || {});
  const metrics = Object.keys(METRIC_LABELS);

  return (
    <div className="p-8 space-y-8 overflow-y-auto h-screen">
      <div>
        <h2 className="text-2xl font-bold text-white">Model Evaluation</h2>
        <p className="text-slate-400 text-sm mt-1">BLEU · ROUGE · Exact Match · F1 — Base vs Prompt-Engineered vs Fine-tuned</p>
      </div>

      {/* Metrics Table */}
      <div className="glass rounded-2xl overflow-hidden">
        <div className="px-6 py-4 border-b border-fintech-600/20 flex items-center gap-2">
          <BarChart3 size={16} className="text-fintech-accent" />
          <h3 className="text-white font-semibold text-sm">Comparison Table</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-fintech-600/20">
                <th className="text-left px-6 py-3 text-slate-400 font-medium">Metric</th>
                {models.map((m) => (
                  <th key={m} className="text-right px-6 py-3 text-slate-400 font-medium">{m}</th>
                ))}
                <th className="text-right px-6 py-3 text-fintech-green font-medium">Improvement</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((k) => {
                const improvement = data.improvement_over_base?.[k] ?? 0;
                return (
                  <tr key={k} className="border-b border-fintech-600/10 hover:bg-fintech-700/20 transition-colors">
                    <td className="px-6 py-3 text-slate-300 font-mono text-xs">{METRIC_LABELS[k]}</td>
                    {models.map((m) => (
                      <td key={m} className="px-6 py-3 text-right font-mono text-xs text-slate-200">
                        {(data.results[m]?.metrics?.[k] ?? 0).toFixed(4)}
                      </td>
                    ))}
                    <td className={`px-6 py-3 text-right font-mono text-xs font-semibold
                      ${improvement > 0 ? "text-fintech-green" : improvement < 0 ? "text-red-400" : "text-slate-500"}`}>
                      {improvement > 0 ? "+" : ""}{improvement.toFixed(4)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Bar visualization */}
      <div className="glass rounded-2xl p-6">
        <div className="flex items-center gap-2 mb-6">
          <TrendingUp size={16} className="text-fintech-accent" />
          <h3 className="text-white font-semibold text-sm">Visual Comparison</h3>
        </div>
        <div className="space-y-4">
          {metrics.map((k) => (
            <div key={k}>
              <div className="flex justify-between mb-1">
                <span className="text-xs text-slate-400 font-mono">{METRIC_LABELS[k]}</span>
              </div>
              <div className="space-y-1.5">
                {models.map((m) => {
                  const val = data.results[m]?.metrics?.[k] ?? 0;
                  return (
                    <div key={m} className="flex items-center gap-3">
                      <span className="text-xs text-slate-500 w-36 truncate">{m}</span>
                      <div className="flex-1 bg-fintech-800 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all duration-700 ${MODEL_COLORS[m] || "bg-fintech-400"}`}
                          style={{ width: `${Math.min(val * 100, 100)}%` }}
                        />
                      </div>
                      <span className="text-xs font-mono text-slate-300 w-14 text-right">{val.toFixed(4)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Hallucination Analysis */}
      {data.hallucination_analysis?.length > 0 && (
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle size={16} className="text-yellow-400" />
            <h3 className="text-white font-semibold text-sm">Hallucination & Error Analysis</h3>
          </div>
          <div className="space-y-3">
            {data.hallucination_analysis.map((h, i) => (
              <div key={i} className="bg-fintech-800/50 rounded-xl p-4 border border-yellow-500/20">
                <p className="text-xs text-slate-400 mb-2"><span className="text-yellow-400">Q:</span> {h.question}</p>
                <p className="text-xs text-fintech-green mb-1"><span className="text-slate-500">Reference:</span> {h.reference}</p>
                <p className="text-xs text-red-400 mb-1"><span className="text-slate-500">Prediction:</span> {h.prediction}</p>
                <p className="text-xs text-yellow-400/70">Hallucinated values: {h.hallucinated_values?.join(", ")}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Examples */}
      {data.results?.["Fine-tuned (QLoRA)"]?.examples?.length > 0 && (
        <div className="glass rounded-2xl p-6">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle size={16} className="text-fintech-green" />
            <h3 className="text-white font-semibold text-sm">Sample Predictions (Fine-tuned)</h3>
          </div>
          <div className="space-y-3">
            {data.results["Fine-tuned (QLoRA)"].examples.slice(0, 5).map((ex, i) => (
              <div key={i} className="bg-fintech-800/50 rounded-xl p-4">
                <p className="text-xs text-fintech-accent mb-2">Q: {ex.question}</p>
                <p className="text-xs text-fintech-green mb-1">✓ Reference: {ex.reference}</p>
                <p className="text-xs text-slate-300">→ Prediction: {ex.prediction}</p>
                <p className={`text-xs mt-1 ${ex.exact_match ? "text-fintech-green" : "text-slate-500"}`}>
                  {ex.exact_match ? "✓ Exact Match" : "~ Partial Match"}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
