import axios from "axios";

const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 300000,  // 5 min — first request loads both models into VRAM
  headers: { "Content-Type": "application/json" },
});

export const queryAPI = {
  ask: (question, useRag = true) =>
    api.post("/api/query", { question, use_rag: useRag }).then((r) => r.data),

  status: () => api.get("/api/query/status").then((r) => r.data),
};

export const voiceAPI = {
  transcribeAndAnswer: (audioBlob) => {
    const form = new FormData();
    form.append("audio", audioBlob, "recording.wav");
    return api
      .post("/api/voice/transcribe", form, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((r) => r.data);
  },
};

export const evaluateAPI = {
  getResults: () => api.get("/api/evaluate").then((r) => r.data),
};

export const logsAPI = {
  getHistory: (limit = 50) =>
    api.get(`/api/logs?limit=${limit}`).then((r) => r.data),
  getStats: () => api.get("/api/logs/stats").then((r) => r.data),
};

export default api;
