import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Zap, RefreshCw } from "lucide-react";
import Message from "../components/Message";
import TypingIndicator from "../components/TypingIndicator";
import VoiceButton from "../components/VoiceButton";
import { queryAPI, voiceAPI } from "../services/api";
import { useVoice } from "../hooks/useVoice";

const SUGGESTIONS = [
  "What was Apple's revenue in 2022?",
  "How does operating income differ from net income?",
  "What is the debt-to-equity ratio and why does it matter?",
  "Explain free cash flow in simple terms.",
];

export default function Chat() {
  const [messages, setMessages] = useState([
    {
      id: 0,
      role: "assistant",
      content:
        "Hello! I'm **QFIS**, your AI-powered financial analyst. I'm fine-tuned on the FinQA dataset using QLoRA and enhanced with RAG retrieval.\n\nAsk me anything about financial statements, ratios, or company performance.",
      sources: [],
      meta: null,
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [useRag, setUseRag] = useState(true);
  const [voiceProcessing, setVoiceProcessing] = useState(false);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const { isRecording, audioBlob, startRecording, stopRecording, clearAudio } = useVoice();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Handle voice blob
  useEffect(() => {
    if (!audioBlob) return;
    const process = async () => {
      setVoiceProcessing(true);
      try {
        const result = await voiceAPI.transcribeAndAnswer(audioBlob);
        setMessages((prev) => [
          ...prev,
          { id: Date.now(), role: "user", content: result.transcription, sources: [], meta: null },
          {
            id: Date.now() + 1,
            role: "assistant",
            content: result.answer,
            sources: result.sources || [],
            meta: { model_type: result.model_type, latency_ms: result.latency_ms, rag_used: true },
          },
        ]);
      } catch {
        setMessages((prev) => [
          ...prev,
          { id: Date.now(), role: "assistant", content: "Voice processing failed. Please try again.", sources: [], meta: null },
        ]);
      } finally {
        setVoiceProcessing(false);
        clearAudio();
      }
    };
    process();
  }, [audioBlob, clearAudio]);

  const sendMessage = useCallback(async (text) => {
    const q = (text || input).trim();
    if (!q || loading) return;
    setInput("");
    setMessages((prev) => [
      ...prev,
      { id: Date.now(), role: "user", content: q, sources: [], meta: null },
    ]);
    setLoading(true);
    try {
      const res = await queryAPI.ask(q, useRag);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: "assistant",
          content: res.answer,
          base_answer: res.base_answer,
          sources: res.sources || [],
          meta: { model_type: res.model_type, latency_ms: res.latency_ms, rag_used: res.rag_used },
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          role: "assistant",
          content: "⚠️ Backend error. Make sure the API server is running on port 8000.",
          sources: [],
          meta: null,
        },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }, [input, loading, useRag]);

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([{
      id: 0, role: "assistant",
      content: "Chat cleared. Ask me a new financial question!",
      sources: [], meta: null,
    }]);
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-fintech-600/20 glass">
        <div>
          <h2 className="text-white font-semibold">Financial Q&A</h2>
          <p className="text-xs text-slate-500">Phi-2 · QLoRA · RAG · FinQA</p>
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer select-none">
            <div
              onClick={() => setUseRag((v) => !v)}
              className={`w-9 h-5 rounded-full transition-colors relative ${useRag ? "bg-fintech-400" : "bg-fintech-700"}`}
            >
              <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${useRag ? "translate-x-4" : "translate-x-0.5"}`} />
            </div>
            RAG
          </label>
          <button onClick={clearChat} className="p-2 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-fintech-700/50 transition-colors">
            <RefreshCw size={15} />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-5">
        {messages.map((msg) => (
          <Message key={msg.id} msg={msg} />
        ))}
        {loading && (
          <div className="flex gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-fintech-accent/20 to-fintech-400/20 border border-fintech-accent/30 flex items-center justify-center text-fintech-accent flex-shrink-0">
              <Zap size={16} />
            </div>
            <TypingIndicator />
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Suggestions */}
      {messages.length <= 1 && (
        <div className="px-6 pb-3 flex flex-wrap gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => sendMessage(s)}
              className="text-xs px-3 py-1.5 rounded-full glass text-fintech-200/70 hover:text-fintech-accent hover:border-fintech-accent/40 border border-fintech-600/30 transition-all"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="px-6 pb-6 pt-2">
        <div className="glass rounded-2xl flex items-end gap-3 p-3 glow-blue">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask a financial question..."
            rows={1}
            className="flex-1 bg-transparent text-sm text-slate-200 placeholder-slate-500 resize-none outline-none max-h-32 leading-relaxed"
            style={{ minHeight: "24px" }}
          />
          <div className="flex items-center gap-2 flex-shrink-0">
            <VoiceButton
              isRecording={isRecording}
              isProcessing={voiceProcessing}
              onStart={startRecording}
              onStop={stopRecording}
            />
            <button
              onClick={() => sendMessage()}
              disabled={!input.trim() || loading}
              className="p-3 rounded-xl bg-fintech-400 text-white hover:bg-fintech-300 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200"
            >
              <Send size={16} />
            </button>
          </div>
        </div>
        <p className="text-center text-xs text-slate-600 mt-2">
          Enter to send · Shift+Enter for new line · Hold mic to speak
        </p>
      </div>
    </div>
  );
}
