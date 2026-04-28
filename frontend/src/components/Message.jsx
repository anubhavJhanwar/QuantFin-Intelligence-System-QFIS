import ReactMarkdown from "react-markdown";
import { Bot, User, Zap, Database } from "lucide-react";

function AnswerBlock({ label, content, accent }) {
  return (
    <div className={`flex-1 rounded-xl p-3 border ${accent}`}>
      <p className={`text-xs font-semibold mb-1.5 ${accent.includes("green") ? "text-fintech-green" : "text-fintech-accent"}`}>
        {label}
      </p>
      <ReactMarkdown
        components={{
          p: ({ children }) => <p className="mb-1 last:mb-0 text-sm text-slate-200">{children}</p>,
          strong: ({ children }) => <strong className="text-fintech-accent font-semibold">{children}</strong>,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

export default function Message({ msg }) {
  const isUser = msg.role === "user";
  const hasComparison = !isUser && msg.base_answer;

  return (
    <div className={`flex gap-3 message-enter ${isUser ? "flex-row-reverse" : "flex-row"}`}>
      {/* Avatar */}
      <div className={`flex-shrink-0 w-9 h-9 rounded-xl flex items-center justify-center
        ${isUser
          ? "bg-fintech-400 text-white"
          : "bg-gradient-to-br from-fintech-accent/20 to-fintech-400/20 border border-fintech-accent/30 text-fintech-accent"
        }`}>
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>

      {/* Bubble */}
      <div className={`${isUser ? "max-w-[75%] items-end" : "w-full max-w-[90%] items-start"} flex flex-col gap-1`}>

        {isUser ? (
          <div className="px-4 py-3 rounded-2xl text-sm leading-relaxed bg-fintech-400 text-white rounded-tr-sm">
            <p>{msg.content}</p>
          </div>
        ) : hasComparison ? (
          /* Side-by-side comparison */
          <div className="w-full glass rounded-2xl rounded-tl-sm p-3 flex flex-col gap-3">
            <div className="flex gap-3">
              <AnswerBlock
                label="⚡ Fine-tuned (QLoRA)"
                content={msg.content}
                accent="border-fintech-accent/40 bg-fintech-accent/5"
              />
              <AnswerBlock
                label="🔵 Base Phi-2"
                content={msg.base_answer}
                accent="border-fintech-green/30 bg-fintech-green/5"
              />
            </div>
          </div>
        ) : (
          <div className="px-4 py-3 rounded-2xl text-sm leading-relaxed glass text-slate-200 rounded-tl-sm">
            <ReactMarkdown
              components={{
                p: ({ children }) => <p className="mb-1 last:mb-0">{children}</p>,
                strong: ({ children }) => <strong className="text-fintech-accent font-semibold">{children}</strong>,
                code: ({ children }) => (
                  <code className="bg-fintech-800 text-fintech-accent px-1 py-0.5 rounded text-xs font-mono">{children}</code>
                ),
              }}
            >
              {msg.content}
            </ReactMarkdown>
          </div>
        )}

        {/* Metadata */}
        {!isUser && msg.meta && (
          <div className="flex items-center gap-3 px-1">
            {msg.meta.rag_used && (
              <span className="flex items-center gap-1 text-xs text-fintech-green">
                <Database size={10} /> RAG
              </span>
            )}
            {msg.meta.model_type && (
              <span className="flex items-center gap-1 text-xs text-fintech-accent/70">
                <Zap size={10} /> {msg.meta.model_type}
              </span>
            )}
            {msg.meta.latency_ms && (
              <span className="text-xs text-slate-500">{msg.meta.latency_ms}ms</span>
            )}
          </div>
        )}

        {/* Sources */}
        {!isUser && msg.sources && msg.sources.length > 0 && (
          <details className="mt-1 w-full">
            <summary className="text-xs text-fintech-200/60 cursor-pointer hover:text-fintech-200 transition-colors px-1">
              {msg.sources.length} source{msg.sources.length > 1 ? "s" : ""} retrieved
            </summary>
            <div className="mt-2 space-y-2">
              {msg.sources.map((s, i) => (
                <div key={i} className="glass rounded-lg p-3 text-xs text-slate-400">
                  <div className="flex justify-between mb-1">
                    <span className="text-fintech-accent/70 font-mono">Context {i + 1}</span>
                    <span className="text-fintech-green">score: {s.score}</span>
                  </div>
                  <p className="line-clamp-3">{s.context}</p>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
