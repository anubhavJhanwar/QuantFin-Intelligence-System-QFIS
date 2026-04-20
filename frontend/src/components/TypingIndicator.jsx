export default function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 px-4 py-3 rounded-2xl glass w-fit">
      <span className="text-xs text-fintech-200 mr-2 font-mono">QFIS thinking</span>
      <span className="typing-dot" />
      <span className="typing-dot" />
      <span className="typing-dot" />
    </div>
  );
}
