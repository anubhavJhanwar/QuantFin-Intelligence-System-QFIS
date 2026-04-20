/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        fintech: {
          900: "#0a0e1a",
          800: "#0f1629",
          700: "#151e38",
          600: "#1a2647",
          500: "#1e3a5f",
          400: "#2563eb",
          300: "#3b82f6",
          200: "#60a5fa",
          100: "#93c5fd",
          accent: "#00d4ff",
          gold: "#f59e0b",
          green: "#10b981",
          red: "#ef4444",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      animation: {
        "fade-in": "fadeIn 0.3s ease-in-out",
        "slide-up": "slideUp 0.4s ease-out",
        "pulse-slow": "pulse 3s infinite",
        typing: "typing 1.2s steps(3) infinite",
      },
      keyframes: {
        fadeIn: { "0%": { opacity: 0 }, "100%": { opacity: 1 } },
        slideUp: { "0%": { transform: "translateY(20px)", opacity: 0 }, "100%": { transform: "translateY(0)", opacity: 1 } },
        typing: { "0%, 100%": { opacity: 1 }, "50%": { opacity: 0 } },
      },
      backdropBlur: { xs: "2px" },
    },
  },
  plugins: [],
};
