import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the FastAPI backend so the frontend can call
// the real endpoint (POST /api/screen) with a same-origin fetch("/api/...").
// Override the target with VITE_API_TARGET if the backend runs elsewhere.
// Use 127.0.0.1 (not "localhost") so the proxy target resolves to IPv4, matching
// a backend bound to 127.0.0.1. Override with VITE_API_TARGET if needed.
const API_TARGET = process.env.VITE_API_TARGET || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
      },
    },
  },
});
