import { Brain, Database, Zap, BarChart3, Mic } from "lucide-react";

const PIPELINE = [
  { step: "01", title: "FinQA Dataset", desc: "1200+ samples cleaned, deduplicated, normalized. Split 70/15/15 with no leakage.", icon: Database, color: "text-fintech-accent" },
  { step: "02", title: "QLoRA Fine-tuning", desc: "Phi-2 (2.7B) fine-tuned with 4-bit NF4 quantization via PEFT. LoRA r=16 on attention layers.", icon: Brain, color: "text-fintech-green" },
  { step: "03", title: "RAG Pipeline", desc: "FAISS vector index with sentence-transformers. Top-3 context retrieval augments every query.", icon: Zap, color: "text-fintech-gold" },
  { step: "04", title: "Evaluation", desc: "BLEU-1/4, ROUGE-1/2/L, Exact Match, F1. Fine-tuned outperforms base on all metrics.", icon: BarChart3, color: "text-fintech-200" },
  { step: "05", title: "Voice Support", desc: "Whisper-tiny for speech-to-text. Hold mic button to ask questions by voice.", icon: Mic, color: "text-red-400" },
];

export default function About() {
  return (
    <div className="p-8 space-y-8 overflow-y-auto h-screen">
      <div>
        <h2 className="text-2xl font-bold text-white">About QFIS</h2>
        <p className="text-slate-400 text-sm mt-1">Financial Question Answering System using QLoRA and RAG</p>
      </div>

      <div className="glass rounded-2xl p-6">
        <h3 className="text-white font-semibold mb-4">System Architecture</h3>
        <div className="space-y-4">
          {PIPELINE.map(({ step, title, desc, icon: Icon, color }) => (
            <div key={step} className="flex gap-4 items-start">
              <div className={`flex-shrink-0 w-10 h-10 rounded-xl bg-fintech-800 flex items-center justify-center ${color}`}>
                <Icon size={18} />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-slate-500">{step}</span>
                  <h4 className="text-sm font-semibold text-white">{title}</h4>
                </div>
                <p className="text-xs text-slate-400 mt-0.5">{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {[
          { label: "Base Model", value: "microsoft/phi-2" },
          { label: "Fine-tuning", value: "QLoRA (4-bit NF4)" },
          { label: "LoRA Rank", value: "r=16, α=32" },
          { label: "Dataset", value: "FinQA (dreamerdeo)" },
          { label: "Vector DB", value: "FAISS (cosine)" },
          { label: "Structured DB", value: "MongoDB" },
          { label: "Backend", value: "FastAPI + uvicorn" },
          { label: "Frontend", value: "React + Tailwind" },
        ].map(({ label, value }) => (
          <div key={label} className="glass rounded-xl p-4">
            <p className="text-xs text-slate-500">{label}</p>
            <p className="text-sm font-mono text-fintech-accent mt-0.5">{value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
