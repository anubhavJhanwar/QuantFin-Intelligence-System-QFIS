import { Mic, MicOff, Loader } from "lucide-react";

export default function VoiceButton({ isRecording, isProcessing, onStart, onStop }) {
  if (isProcessing) {
    return (
      <button disabled className="p-3 rounded-xl bg-fintech-700 text-fintech-accent/50 cursor-not-allowed">
        <Loader size={18} className="animate-spin" />
      </button>
    );
  }

  return (
    <button
      onMouseDown={onStart}
      onMouseUp={onStop}
      onTouchStart={onStart}
      onTouchEnd={onStop}
      className={`p-3 rounded-xl transition-all duration-200 select-none
        ${isRecording
          ? "bg-red-500/20 text-red-400 border border-red-500/50 animate-pulse"
          : "bg-fintech-700 text-fintech-200 hover:bg-fintech-600 hover:text-fintech-accent border border-fintech-600/50"
        }`}
      title={isRecording ? "Release to send" : "Hold to speak"}
    >
      {isRecording ? <MicOff size={18} /> : <Mic size={18} />}
    </button>
  );
}
